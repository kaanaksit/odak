import torch
import logging
import itertools
from .util import set_amplitude, generate_complex_field, calculate_amplitude, calculate_phase
from .lens import quadratic_phase_function
from .util import wavenumber
from ..tools import zero_pad, crop_center, generate_2d_gaussian, circular_binary_mask, correlation_2d
from tqdm import tqdm


def propagate_beam(
                   field,
                   k,
                   distance,
                   dx,
                   wavelength,
                   propagation_type='Bandlimited Angular Spectrum',
                   kernel = None,
                   zero_padding = [True, False, True],
                   aperture = 1.,
                   scale = 1,
                   samples = [20, 20, 5, 5]
                  ):
    """
    Definitions for various beam propagation methods mostly in accordence with "Computational Fourier Optics" by David Vuelz.

    Parameters
    ----------
    field            : torch.complex
                       Complex field [m x n].
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    propagation_type : str
                       Type of the propagation.
                       The options are Impulse Response Fresnel, Transfer Function Fresnel, Angular Spectrum, Bandlimited Angular Spectrum, Fraunhofer.
    kernel           : torch.complex
                       Custom complex kernel.
    zero_padding     : list
                       Zero padding the input field if the first item in the list set True.
                       Zero padding in the Fourier domain if the second item in the list set to True.
                       Cropping the result with half resolution if the third item in the list is set to true.
                       Note that in Fraunhofer propagation, setting the second item True or False will have no effect.
    aperture         : torch.tensor
                       Aperture at Fourier domain default:[2m x 2n], otherwise depends on `zero_padding`.
                       If provided as a floating point 1, there will be no aperture in Fourier domain.
    scale            : int
                       Resolution factor to scale generated kernel.
    samples          : list
                       When using `Impulse Response Fresnel` propagation, these sample counts along X and Y will be used to represent a rectangular aperture. First two is for a hologram pixel and second two is for an image plane pixel.

    Returns
    -------
    result           : torch.complex
                       Final complex field [m x n].
    """
    if zero_padding[0]:
        field = zero_pad(field)
    if propagation_type == 'Angular Spectrum':
        result = angular_spectrum(field, k, distance, dx, wavelength, zero_padding[1], aperture = aperture)
    elif propagation_type == 'Bandlimited Angular Spectrum':
        result = band_limited_angular_spectrum(field, k, distance, dx, wavelength, zero_padding[1], aperture = aperture)
    elif propagation_type == 'Impulse Response Fresnel':
        result = impulse_response_fresnel(field, k, distance, dx, wavelength, zero_padding[1], aperture = aperture, scale = scale, samples = samples)
    elif propagation_type == 'Seperable Impulse Response Fresnel':
        result = seperable_impulse_response_fresnel(field, k, distance, dx, wavelength, zero_padding[1], aperture = aperture, scale = scale, samples = samples)
    elif propagation_type == 'Transfer Function Fresnel':
        result = transfer_function_fresnel(field, k, distance, dx, wavelength, zero_padding[1], aperture = aperture)
    elif propagation_type == 'custom':
        result = custom(field, kernel, zero_padding[1], aperture = aperture)
    elif propagation_type == 'Fraunhofer':
        result = fraunhofer(field, k, distance, dx, wavelength)
    elif propagation_type == 'Incoherent Angular Spectrum':
        result = incoherent_angular_spectrum(field, k, distance, dx, wavelength, zero_padding[1], aperture = aperture)
    else:
        logging.warning('Propagation type not recognized')
        assert True == False
    if zero_padding[2]:
        result = crop_center(result)
    return result


def get_propagation_kernel(
                           nu, 
                           nv, 
                           dx = 8e-6, 
                           wavelength = 515e-9, 
                           distance = 0., 
                           device = torch.device('cpu'), 
                           propagation_type = 'Bandlimited Angular Spectrum', 
                           scale = 1,
                           samples = [20, 20, 5, 5]
                          ):
    """
    Get propagation kernel for the propagation type.

    Parameters
    ----------
    nu                 : int
                         Resolution at X axis in pixels.
    nv                 : int
                         Resolution at Y axis in pixels.
    dx                 : float
                         Pixel pitch in meters.
    wavelength         : float
                         Wavelength in meters.
    distance           : float
                         Distance in meters.
    device             : torch.device
                         Device, for more see torch.device().
    propagation_type   : str
                         Propagation type.
                         The options are `Angular Spectrum`, `Bandlimited Angular Spectrum` and `Transfer Function Fresnel`.
    scale              : int
                         Scale factor for scaled beam propagation.
    samples            : list
                         When using `Impulse Response Fresnel` propagation, these sample counts along X and Y will be used to represent a rectangular aperture. First two is for a hologram pixel and second two is for an image plane pixel.


    Returns
    -------
    kernel             : torch.tensor
                         Complex kernel for the given propagation type.
    """                                                      
    logging.warning('Requested propagation kernel size for %s method with %s m distance, %s m pixel pitch, %s m wavelength, %s x %s resolutions, x%s scale and %s samples.'.format(propagation_type, distance, dx, nu, nv, scale, samples))
    if propagation_type == 'Bandlimited Angular Spectrum':
        kernel = get_band_limited_angular_spectrum_kernel(
                                                          nu = nu,
                                                          nv = nv,
                                                          dx = dx,
                                                          wavelength = wavelength,
                                                          distance = distance,
                                                          device = device
                                                         )
    elif propagation_type == 'Angular Spectrum':
        kernel = get_angular_spectrum_kernel(
                                             nu = nu,
                                             nv = nv,
                                             dx = dx,
                                             wavelength = wavelength,
                                             distance = distance,
                                             device = device
                                            )
    elif propagation_type == 'Transfer Function Fresnel':
        kernel = get_transfer_function_fresnel_kernel(
                                                      nu = nu,
                                                      nv = nv,
                                                      dx = dx,
                                                      wavelength = wavelength,
                                                      distance = distance,
                                                      device = device
                                                     )
    elif propagation_type == 'Impulse Response Fresnel':
        kernel = get_impulse_response_fresnel_kernel(
                                                     nu = nu, 
                                                     nv = nv, 
                                                     dx = dx, 
                                                     wavelength = wavelength,
                                                     distance = distance,
                                                     device =  device,
                                                     scale = scale,
                                                     aperture_samples = samples
                                                    )
    elif propagation_type == 'Incoherent Angular Spectrum':
        kernel = get_incoherent_angular_spectrum_kernel(
                                                        nu = nu,
                                                        nv = nv, 
                                                        dx = dx, 
                                                        wavelength = wavelength, 
                                                        distance = distance,
                                                        device = device
                                                       )
    elif propagation_type == 'Seperable Impulse Response Fresnel':
        kernel, _, _, _ = get_seperable_impulse_response_fresnel_kernel(
                                                                        nu = nu,
                                                                        nv = nv,
                                                                        dx = dx,
                                                                        wavelength = wavelength,
                                                                        distance = distance,
                                                                        device = device,
                                                                        scale = scale,
                                                                        aperture_samples = samples
                                                                       )
    else:
        logging.warning('Propagation type not recognized')
        assert True == False
    return kernel


def get_light_kernels(
                      wavelengths,
                      distances,
                      pixel_pitches,
                      resolution = [1080, 1920],
                      resolution_factor = 1,
                      samples = [50, 50, 5, 5],
                      propagation_type = 'Bandlimited Angular Spectrum',
                      kernel_type = 'spatial',
                      device = torch.device('cpu')
                     ):
    """
    Utility function to request a tensor filled with light transport kernels according to the given optical configurations.

    Parameters
    ----------
    wavelengths        : list
                         A list of wavelengths.
    distances          : list
                         A list of propagation distances.
    pixel_pitches      : list
                         A list of pixel_pitches.
    resolution         : list
                         Resolution of the light transport kernel.
    resolution_factor  : int
                         If `Impulse Response Fresnel` propagation is used, this resolution factor could be set larger than one leading to higher resolution light transport kernels than the provided native `resolution`. For more, see odak.learn.wave.get_impulse_response_kernel().
    samples            : list
                         If `Impulse Response Fresnel` propagation is used, these sample counts will be used to calculate the light transport kernel. For more, see odak.learn.wave.get_impulse_response_kernel().
    propagation_type   : str
                         Propagation type. For more, see odak.learn.wave.propagate_beam().
    kernel_type        : str
                         If set to `spatial`, light transport kernels will be provided in space. But if set to `fourier`, these kernels will be provided in the Fourier domain.
    device             : torch.device
                         Device used for computation (i.e., cpu, cuda).

    Returns
    -------
    light_kernels_amplitude : torch.tensor
                              Amplitudes of the light kernels generated [w x d x p x m x n].
    light_kernels_phase     : torch.tensor
                              Phases of the light kernels generated [w x d x p x m x n].
    light_kernels_complex   : torch.tensor
                              Complex light kernels generated [w x d x p x m x n].
    light_parameters        : torch.tensor
                              Parameters of each pixel in light_kernels* [w x d x p x m x n x 5].  Last dimension contains, wavelengths, distances, pixel pitches, X and Y locations in order.
    """
    if propagation_type != 'Impulse Response Fresnel':
        resolution_factor = 1
    light_kernels_complex = torch.zeros(            
                                        len(wavelengths),
                                        len(distances),
                                        len(pixel_pitches),
                                        resolution[0] * resolution_factor,
                                        resolution[1] * resolution_factor,
                                        dtype = torch.complex64,
                                        device = device
                                       )
    light_parameters = torch.zeros(
                                   len(wavelengths),
                                   len(distances),
                                   len(pixel_pitches),
                                   resolution[0] * resolution_factor,
                                   resolution[1] * resolution_factor,
                                   5,
                                   dtype = torch.float32,
                                   device = device
                                  )
    for wavelength_id, distance_id, pixel_pitch_id in itertools.product(
                                                                        range(len(wavelengths)),
                                                                        range(len(distances)),
                                                                        range(len(pixel_pitches)),
                                                                       ):
        pixel_pitch = pixel_pitches[pixel_pitch_id]
        wavelength = wavelengths[wavelength_id]
        distance = distances[distance_id]
        kernel_fourier = get_propagation_kernel(
                                                nu = resolution[0],
                                                nv = resolution[1],
                                                dx = pixel_pitch,
                                                wavelength = wavelength,
                                                distance = distance,
                                                device = device,
                                                propagation_type = propagation_type,
                                                scale = resolution_factor,
                                                samples = samples
                                               )
        if kernel_type == 'spatial':
            kernel = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(kernel_fourier)))
        elif kernel_type == 'fourier':
            kernel = kernel_fourier
        else:
            logging.warning('Unknown kernel type requested.')
            raise ValueError('Unknown kernel type requested.')
        kernel_amplitude = calculate_amplitude(kernel)
        kernel_phase = calculate_phase(kernel) % (2 * torch.pi)
        light_kernels_complex[wavelength_id, distance_id, pixel_pitch_id] = kernel
        light_parameters[wavelength_id, distance_id, pixel_pitch_id, :, :, 0] = wavelength
        light_parameters[wavelength_id, distance_id, pixel_pitch_id, :, :, 1] = distance
        light_parameters[wavelength_id, distance_id, pixel_pitch_id, :, :, 2] = pixel_pitch
        x = torch.linspace(-1., 1., resolution[0] * resolution_factor, device = device) * pixel_pitch / 2. * resolution[0]
        y = torch.linspace(-1., 1., resolution[1] * resolution_factor, device = device) * pixel_pitch / 2. * resolution[1]
        X, Y = torch.meshgrid(x, y, indexing = 'ij')
        light_parameters[wavelength_id, distance_id, pixel_pitch_id, :, :, 3] = X
        light_parameters[wavelength_id, distance_id, pixel_pitch_id, :, :, 4] = Y
    light_kernels_amplitude = calculate_amplitude(light_kernels_complex)
    light_kernels_phase = calculate_phase(light_kernels_complex) % (2. * torch.pi)
    return light_kernels_amplitude, light_kernels_phase, light_kernels_complex, light_parameters


def fraunhofer(field, k, distance, dx, wavelength):
    """
    A definition to calculate light transport usin Fraunhofer approximation.

    Parameters
    ----------
    field            : torch.complex
                       Complex field (MxN).
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.

    Returns
    -------
    result           : torch.complex
                       Final complex field (MxN).
    """
    nv, nu = field.shape[-1], field.shape[-2]
    x = torch.linspace(-nv*dx/2, nv*dx/2, nv, dtype=torch.float32)
    y = torch.linspace(-nu*dx/2, nu*dx/2, nu, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    Z = torch.pow(X, 2) + torch.pow(Y, 2)
    c = 1. / (1j * wavelength * distance) * torch.exp(1j * k * 0.5 / distance * Z)
    c = c.to(field.device)
    result = c * torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(field))) * dx ** 2
    return result


def custom(field, kernel, zero_padding = False, aperture = 1.):
    """
    A definition to calculate convolution based Fresnel approximation for beam propagation.

    Parameters
    ----------
    field            : torch.complex
                       Complex field [m x n].
    kernel           : torch.complex
                       Custom complex kernel for beam propagation.
    zero_padding     : bool
                       Zero pad in Fourier domain.
    aperture         : torch.tensor
                       Fourier domain aperture (e.g., pinhole in a typical holographic display).
                       The default is one, but an aperture could be as large as input field [m x n].

    Returns
    -------
    result           : torch.complex
                       Final complex field (MxN).

    """
    if type(kernel) == type(None):
        H = torch.ones(field.shape).to(field.device)
    else:
        H = kernel * aperture
    U1 = torch.fft.fftshift(torch.fft.fft2(field)) * aperture
    if zero_padding == False:
        U2 = H * U1
    elif zero_padding == True:
        U2 = zero_pad(H * U1)
    result = torch.fft.ifft2(torch.fft.ifftshift(U2))
    return result


def get_impulse_response_fresnel_kernel(nu, nv, dx = 8e-6, wavelength = 515e-9, distance = 0., device = torch.device('cpu'), scale = 1, aperture_samples = [20, 20, 5, 5]):
    """
    Helper function for odak.learn.wave.impulse_response_fresnel.

    Parameters
    ----------
    nu                 : int
                         Resolution at X axis in pixels.
    nv                 : int
                         Resolution at Y axis in pixels.
    dx                 : float
                         Pixel pitch in meters.
    wavelength         : float
                         Wavelength in meters.
    distance           : float
                         Distance in meters.
    device             : torch.device
                         Device, for more see torch.device().
    scale              : int
                         Scale with respect to nu and nv (e.g., scale = 2 leads to  2 x nu and 2 x nv resolution for H).
    aperture_samples   : list
                         Number of samples to represent a rectangular pixel. First two is for XY of hologram plane pixels, and second two is for image plane pixels.

    Returns
    -------
    H                  : torch.complex64
                         Complex kernel in Fourier domain.
    """
    k = wavenumber(wavelength)
    distance = torch.as_tensor(distance, device = device)
    length_x, length_y = (torch.tensor(dx * nu, device = device), torch.tensor(dx * nv, device = device))
    x = torch.linspace(- length_x / 2., length_x / 2., nu * scale, device = device)
    y = torch.linspace(- length_y / 2., length_y / 2., nv * scale, device = device)
    X, Y = torch.meshgrid(x, y, indexing = 'ij')
    wxs = torch.linspace(- dx / 2., dx / 2., aperture_samples[0], device = device)
    wys = torch.linspace(- dx / 2., dx / 2., aperture_samples[1], device = device)
    h = torch.zeros(nu * scale, nv * scale, dtype = torch.complex64, device = device)
    pxs = torch.linspace(- dx / 2., dx / 2., aperture_samples[2], device = device)
    pys = torch.linspace(- dx / 2., dx / 2., aperture_samples[3], device = device)
    for wx in tqdm(wxs):
        for wy in wys:
            for px in pxs:
                for py in pys:
                    r = (X + px - wx) ** 2 + (Y + py - wy) ** 2
                    h += 1. / (1j * wavelength * distance) * torch.exp(1j * k / (2 * distance) * r) 
    H = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(h))) * dx ** 2 / aperture_samples[0] / aperture_samples[1] / aperture_samples[2] / aperture_samples[3]
    return H


def get_seperable_impulse_response_fresnel_kernel(
                                                  nu,
                                                  nv,
                                                  dx = 3.74e-6,
                                                  wavelength = 515e-9,
                                                  distance = 0.,
                                                  scale = 1,
                                                  aperture_samples = [50, 50, 5, 5],
                                                  device = torch.device('cpu')
                                                 ):
    """
    Returns impulse response fresnel kernel in separable form.

    Parameters
    ----------
    nu                 : int
                         Resolution at X axis in pixels.
    nv                 : int
                         Resolution at Y axis in pixels.
    dx                 : float
                         Pixel pitch in meters.
    wavelength         : float
                         Wavelength in meters.
    distance           : float
                         Distance in meters.
    device             : torch.device
                         Device, for more see torch.device().
    scale              : int
                         Scale with respect to nu and nv (e.g., scale = 2 leads to  2 x nu and 2 x nv resolution for H).
    aperture_samples   : list
                         Number of samples to represent a rectangular pixel. First two is for XY of hologram plane pixels, and second two is for image plane pixels.

    Returns
    -------
    H                  : torch.complex64
                         Complex kernel in Fourier domain.
    h                  : torch.complex64
                         Complex kernel in spatial domain.
    h_x                : torch.complex64
                         1D complex kernel in spatial domain along X axis.
    h_y                : torch.complex64
                         1D complex kernel in spatial domain along Y axis.
    """
    k = wavenumber(wavelength)
    distance = torch.as_tensor(distance, device = device)
    length_x, length_y = (
                          torch.tensor(dx * nu, device = device),
                          torch.tensor(dx * nv, device = device)
                         )
    x = torch.linspace(- length_x / 2., length_x / 2., nu * scale, device = device)
    y = torch.linspace(- length_y / 2., length_y / 2., nv * scale, device = device)
    wxs = torch.linspace(- dx / 2., dx / 2., aperture_samples[0], device = device).unsqueeze(0).unsqueeze(0)
    wys = torch.linspace(- dx / 2., dx / 2., aperture_samples[1], device = device).unsqueeze(0).unsqueeze(-1)
    pxs = torch.linspace(- dx / 2., dx / 2., aperture_samples[2], device = device).unsqueeze(0).unsqueeze(-1)
    pys = torch.linspace(- dx / 2., dx / 2., aperture_samples[3], device = device).unsqueeze(0).unsqueeze(0)
    wxs = (wxs - pxs).reshape(1, -1).unsqueeze(-1)
    wys = (wys - pys).reshape(1, -1).unsqueeze(1)

    X = x.unsqueeze(-1).unsqueeze(-1)
    Y = y[y.shape[0] // 2].unsqueeze(-1).unsqueeze(-1)
    r_x = (X + wxs) ** 2
    r_y = (Y + wys) ** 2
    r = r_x + r_y
    h_x = torch.exp(1j * k / (2 * distance) * r)
    h_x = torch.sum(h_x, axis = (1, 2))

    if nu != nv:
        X = x[x.shape[0] // 2].unsqueeze(-1).unsqueeze(-1)
        Y = y.unsqueeze(-1).unsqueeze(-1)
        r_x = (X + wxs) ** 2
        r_y = (Y + wys) ** 2
        r = r_x + r_y
        h_y = torch.exp(1j * k * r / (2 * distance))
        h_y = torch.sum(h_y, axis = (1, 2))
    else:
        h_y = h_x.detach().clone()
    h = torch.exp(1j * k * distance) / (1j * wavelength * distance) * h_x.unsqueeze(1) * h_y.unsqueeze(0)
    H = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(h))) * dx ** 2 / aperture_samples[0] / aperture_samples[1] / aperture_samples[2] / aperture_samples[3]
    return H, h, h_x, h_y


def get_point_wise_impulse_response_fresnel_kernel(
                                                   aperture_points,
                                                   aperture_field,
                                                   target_points,
                                                   resolution,
                                                   resolution_factor = 1,
                                                   wavelength = 515e-9,
                                                   distance = 0.,
                                                   randomization = False,
                                                   device = torch.device('cpu')
                                                  ):
    """
    This function is a freeform point spread function calculation routine for an aperture defined with a complex field, `aperture_field`, and locations in space, `aperture_points`.
    The point spread function is calculated over provided points, `target_points`.
    The final result is reshaped to follow the provided `resolution`.

    Parameters
    ----------
    aperture_points          : torch.tensor
                               Points representing an aperture in Euler space (XYZ) [m x 3].
    aperture_field           : torch.tensor
                               Complex field for each point provided by `aperture_points` [1 x m].
    target_points            : torch.tensor
                               Target points where the propagated field will be calculated [n x 1].
    resolution               : list
                               Final resolution that the propagated field will be reshaped [X x Y].
    resolution_factor        : int
                               Scale with respect to `resolution` (e.g., scale = 2 leads to `2 x resolution` for the final complex field.
    wavelength               : float
                               Wavelength in meters.
    randomization            : bool
                               If set `True`, this will help generate a noisy response roughly approximating a real life case, where imperfections occur.
    distance                 : float
                               Distance in meters.

    Returns
    -------
    h                        : float
                               Complex field in spatial domain.
    """
    device = aperture_field.device
    k = wavenumber(wavelength)
    if randomization:
        pp = [
              aperture_points[:, 0].max() - aperture_points[:, 0].min(),
              aperture_points[:, 1].max() - aperture_points[:, 1].min()
             ]
        target_points[:, 0] = target_points[:, 0] - torch.randn(target_points[:, 0].shape) * pp[0]
        target_points[:, 1] = target_points[:, 1] - torch.randn(target_points[:, 1].shape) * pp[1]
    deltaX = aperture_points[:, 0].unsqueeze(0) - target_points[:, 0].unsqueeze(-1)
    deltaY = aperture_points[:, 1].unsqueeze(0) - target_points[:, 1].unsqueeze(-1)
    r = deltaX ** 2 + deltaY ** 2
    h = torch.exp(1j * k / (2 * distance) * r) * aperture_field
    h = torch.sum(h, dim = 1).reshape(resolution[0] * resolution_factor, resolution[1] * resolution_factor)
    h = 1. / (1j * wavelength * distance) * h
    return h


def seperable_impulse_response_fresnel(field, k, distance, dx, wavelength, zero_padding = False, aperture = 1., scale = 1, samples = [20, 20, 5, 5]):
    """
    A definition to calculate convolution based Fresnel approximation for beam propagation for a rectangular aperture using the seperable property.

    Parameters
    ----------
    field            : torch.complex
                       Complex field (MxN).
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    zero_padding     : bool
                       Zero pad in Fourier domain.
    aperture         : torch.tensor
                       Fourier domain aperture (e.g., pinhole in a typical holographic display).
                       The default is one, but an aperture could be as large as input field [m x n].
    scale            : int
                       Resolution factor to scale generated kernel.
    samples          : list
                       When using `Impulse Response Fresnel` propagation, these sample counts along X and Y will be used to represent a rectangular aperture. First two is for hologram plane pixel and the last two is for image plane pixel.

    Returns
    -------
    result           : torch.complex
                       Final complex field (MxN).

    """
    H = get_propagation_kernel(
                               nu = field.shape[-2], 
                               nv = field.shape[-1], 
                               dx = dx, 
                               wavelength = wavelength, 
                               distance = distance, 
                               propagation_type = 'Seperable Impulse Response Fresnel',
                               device = field.device,
                               scale = scale,
                               samples = samples
                              )
    if scale > 1:
        field_amplitude = calculate_amplitude(field)
        field_phase = calculate_phase(field)
        field_scale_amplitude = torch.zeros(field.shape[-2] * scale, field.shape[-1] * scale, device = field.device)
        field_scale_phase = torch.zeros_like(field_scale_amplitude)
        field_scale_amplitude[::scale, ::scale] = field_amplitude
        field_scale_phase[::scale, ::scale] = field_phase
        field_scale = generate_complex_field(field_scale_amplitude, field_scale_phase)
    else:
        field_scale = field
    result = custom(field_scale, H, zero_padding = zero_padding, aperture = aperture)
    return result


def impulse_response_fresnel(field, k, distance, dx, wavelength, zero_padding = False, aperture = 1., scale = 1, samples = [20, 20, 5, 5]):
    """
    A definition to calculate convolution based Fresnel approximation for beam propagation.

    Parameters
    ----------
    field            : torch.complex
                       Complex field (MxN).
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    zero_padding     : bool
                       Zero pad in Fourier domain.
    aperture         : torch.tensor
                       Fourier domain aperture (e.g., pinhole in a typical holographic display).
                       The default is one, but an aperture could be as large as input field [m x n].
    scale            : int
                       Resolution factor to scale generated kernel.
    samples          : list
                       When using `Impulse Response Fresnel` propagation, these sample counts along X and Y will be used to represent a rectangular aperture. First two is for hologram plane pixel and the last two is for image plane pixel.

    Returns
    -------
    result           : torch.complex
                       Final complex field (MxN).

    """
    H = get_propagation_kernel(
                               nu = field.shape[-2], 
                               nv = field.shape[-1], 
                               dx = dx, 
                               wavelength = wavelength, 
                               distance = distance, 
                               propagation_type = 'Impulse Response Fresnel',
                               device = field.device,
                               scale = scale,
                               samples = samples
                              )
    if scale > 1:
        field_amplitude = calculate_amplitude(field)
        field_phase = calculate_phase(field)
        field_scale_amplitude = torch.zeros(field.shape[-2] * scale, field.shape[-1] * scale, device = field.device)
        field_scale_phase = torch.zeros_like(field_scale_amplitude)
        field_scale_amplitude[::scale, ::scale] = field_amplitude
        field_scale_phase[::scale, ::scale] = field_phase
        field_scale = generate_complex_field(field_scale_amplitude, field_scale_phase)
    else:
        field_scale = field
    result = custom(field_scale, H, zero_padding = zero_padding, aperture = aperture)
    return result


def get_transfer_function_fresnel_kernel(nu, nv, dx = 8e-6, wavelength = 515e-9, distance = 0., device = torch.device('cpu')):
    """
    Helper function for odak.learn.wave.transfer_function_fresnel.

    Parameters
    ----------
    nu                 : int
                         Resolution at X axis in pixels.
    nv                 : int
                         Resolution at Y axis in pixels.
    dx                 : float
                         Pixel pitch in meters.
    wavelength         : float
                         Wavelength in meters.
    distance           : float
                         Distance in meters.
    device             : torch.device
                         Device, for more see torch.device().


    Returns
    -------
    H                  : torch.complex64
                         Complex kernel in Fourier domain.
    """
    distance = torch.tensor([distance]).to(device)
    fx = torch.linspace(-1. / 2. /dx, 1. / 2. /dx, nu, dtype = torch.float32, device = device)
    fy = torch.linspace(-1. / 2. /dx, 1. / 2. /dx, nv, dtype = torch.float32, device = device)
    FY, FX = torch.meshgrid(fx, fy, indexing = 'ij')
    k = wavenumber(wavelength)
    H = torch.exp(-1j * distance * (k - torch.pi * wavelength * (FX ** 2 + FY ** 2)))
    return H


def transfer_function_fresnel(field, k, distance, dx, wavelength, zero_padding = False, aperture = 1.):
    """
    A definition to calculate convolution based Fresnel approximation for beam propagation.

    Parameters
    ----------
    field            : torch.complex
                       Complex field (MxN).
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    zero_padding     : bool
                       Zero pad in Fourier domain.
    aperture         : torch.tensor
                       Fourier domain aperture (e.g., pinhole in a typical holographic display).
                       The default is one, but an aperture could be as large as input field [m x n].


    Returns
    -------
    result           : torch.complex
                       Final complex field (MxN).

    """
    H = get_propagation_kernel(
                               nu = field.shape[-2], 
                               nv = field.shape[-1], 
                               dx = dx, 
                               wavelength = wavelength, 
                               distance = distance, 
                               propagation_type = 'Transfer Function Fresnel',
                               device = field.device
                              )
    result = custom(field, H, zero_padding = zero_padding, aperture = aperture)
    return result


def get_angular_spectrum_kernel(nu, nv, dx = 8e-6, wavelength = 515e-9, distance = 0., device = torch.device('cpu')):
    """
    Helper function for odak.learn.wave.angular_spectrum.

    Parameters
    ----------
    nu                 : int
                         Resolution at X axis in pixels.
    nv                 : int
                         Resolution at Y axis in pixels.
    dx                 : float
                         Pixel pitch in meters.
    wavelength         : float
                         Wavelength in meters.
    distance           : float
                         Distance in meters.
    device             : torch.device
                         Device, for more see torch.device().


    Returns
    -------
    H                  : float
                         Complex kernel in Fourier domain.
    """
    distance = torch.tensor([distance]).to(device)
    fx = torch.linspace(-1. / 2. / dx, 1. / 2. / dx, nu, dtype = torch.float32, device = device)
    fy = torch.linspace(-1. / 2. / dx, 1. / 2. / dx, nv, dtype = torch.float32, device = device)
    FY, FX = torch.meshgrid(fx, fy, indexing='ij')
    H = torch.exp(1j  * distance * (2 * (torch.pi * (1 / wavelength) * torch.sqrt(1. - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))))
    H = H.to(device)
    return H


def get_incoherent_angular_spectrum_kernel(nu, nv, dx = 8e-6, wavelength = 515e-9, distance = 0., device = torch.device('cpu')):
    """
    Helper function for odak.learn.wave.angular_spectrum.

    Parameters
    ----------
    nu                 : int
                         Resolution at X axis in pixels.
    nv                 : int
                         Resolution at Y axis in pixels.
    dx                 : float
                         Pixel pitch in meters.
    wavelength         : float
                         Wavelength in meters.
    distance           : float
                         Distance in meters.
    device             : torch.device
                         Device, for more see torch.device().


    Returns
    -------
    H                  : float
                         Complex kernel in Fourier domain.
    """
    distance = torch.tensor([distance]).to(device)
    fx = torch.linspace(-1. / 2. / dx, 1. / 2. / dx, nu, dtype = torch.float32, device = device)
    fy = torch.linspace(-1. / 2. / dx, 1. / 2. / dx, nv, dtype = torch.float32, device = device)
    FY, FX = torch.meshgrid(fx, fy, indexing='ij')
    H = torch.exp(1j  * distance * (2 * (torch.pi * (1 / wavelength) * torch.sqrt(1. - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))))
    H_ptime = correlation_2d(H, H)
    H = H_ptime.to(device)
    return H


def angular_spectrum(field, k, distance, dx, wavelength, zero_padding = False, aperture = 1.):
    """
    A definition to calculate convolution with Angular Spectrum method for beam propagation.

    Parameters
    ----------
    field            : torch.complex
                       Complex field [m x n].
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    zero_padding     : bool
                       Zero pad in Fourier domain.
    aperture         : torch.tensor
                       Fourier domain aperture (e.g., pinhole in a typical holographic display).
                       The default is one, but an aperture could be as large as input field [m x n].


    Returns
    -------
    result           : torch.complex
                       Final complex field (MxN).

    """
    H = get_propagation_kernel(
                               nu = field.shape[-2], 
                               nv = field.shape[-1], 
                               dx = dx, 
                               wavelength = wavelength, 
                               distance = distance, 
                               propagation_type = 'Angular Spectrum',
                               device = field.device
                              )
    result = custom(field, H, zero_padding = zero_padding, aperture = aperture)
    return result


def incoherent_angular_spectrum(field, k, distance, dx, wavelength, zero_padding = False, aperture = 1.):
    """
    A definition to calculate incoherent beam propagation with Angular Spectrum method.

    Parameters
    ----------
    field            : torch.complex
                       Complex field [m x n].
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    zero_padding     : bool
                       Zero pad in Fourier domain.
    aperture         : torch.tensor
                       Fourier domain aperture (e.g., pinhole in a typical holographic display).
                       The default is one, but an aperture could be as large as input field [m x n].


    Returns
    -------
    result           : torch.complex
                       Final complex field [m x n].
    """
    H = get_propagation_kernel(
                               nu = field.shape[-2], 
                               nv = field.shape[-1], 
                               dx = dx, 
                               wavelength = wavelength, 
                               distance = distance, 
                               propagation_type = 'Incoherent Angular Spectrum',
                               device = field.device
                              )
    result = custom(field, H, zero_padding = zero_padding, aperture = aperture)
    return result


def get_band_limited_angular_spectrum_kernel(
                                             nu,
                                             nv,
                                             dx = 8e-6,
                                             wavelength = 515e-9,
                                             distance = 0.,
                                             device = torch.device('cpu')
                                            ):
    """
    Helper function for odak.learn.wave.band_limited_angular_spectrum.

    Parameters
    ----------
    nu                 : int
                         Resolution at X axis in pixels.
    nv                 : int
                         Resolution at Y axis in pixels.
    dx                 : float
                         Pixel pitch in meters.
    wavelength         : float
                         Wavelength in meters.
    distance           : float
                         Distance in meters.
    device             : torch.device
                         Device, for more see torch.device().


    Returns
    -------
    H                  : torch.complex64
                         Complex kernel in Fourier domain.
    """
    x = dx * float(nu)
    y = dx * float(nv)
    fx = torch.linspace(
                        -1 / (2 * dx) + 0.5 / (2 * x),
                         1 / (2 * dx) - 0.5 / (2 * x),
                         nu,
                         dtype = torch.float32,
                         device = device
                        )
    fy = torch.linspace(
                        -1 / (2 * dx) + 0.5 / (2 * y),
                        1 / (2 * dx) - 0.5 / (2 * y),
                        nv,
                        dtype = torch.float32,
                        device = device
                       )
    FY, FX = torch.meshgrid(fx, fy, indexing='ij')
    HH_exp = 2 * torch.pi * torch.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))
    distance = torch.tensor([distance], device = device)
    H_exp = torch.mul(HH_exp, distance)
    fx_max = 1 / torch.sqrt((2 * distance * (1 / x))**2 + 1) / wavelength
    fy_max = 1 / torch.sqrt((2 * distance * (1 / y))**2 + 1) / wavelength
    H_filter = ((torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)).clone().detach()
    H = generate_complex_field(H_filter, H_exp)
    return H


def band_limited_angular_spectrum(
                                  field,
                                  k,
                                  distance,
                                  dx,
                                  wavelength,
                                  zero_padding = False,
                                  aperture = 1.
                                 ):
    """
    A definition to calculate bandlimited angular spectrum based beam propagation. For more 
    `Matsushima, Kyoji, and Tomoyoshi Shimobaba. "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." Optics express 17.22 (2009): 19662-19673`.

    Parameters
    ----------
    field            : torch.complex
                       A complex field.
                       The expected size is [m x n].
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    zero_padding     : bool
                       Zero pad in Fourier domain.
    aperture         : torch.tensor
                       Fourier domain aperture (e.g., pinhole in a typical holographic display).
                       The default is one, but an aperture could be as large as input field [m x n].


    Returns
    -------
    result           : torch.complex
                       Final complex field [m x n].
    """
    H = get_propagation_kernel(
                               nu = field.shape[-2], 
                               nv = field.shape[-1], 
                               dx = dx, 
                               wavelength = wavelength, 
                               distance = distance, 
                               propagation_type = 'Bandlimited Angular Spectrum',
                               device = field.device
                              )
    result = custom(field, H, zero_padding = zero_padding, aperture = aperture)
    return result


def gerchberg_saxton(field, n_iterations, distance, dx, wavelength, slm_range=6.28, propagation_type='Transfer Function Fresnel'):
    """
    Definition to compute a hologram using an iterative method called Gerchberg-Saxton phase retrieval algorithm. For more on the method, see: Gerchberg, Ralph W. "A practical algorithm for the determination of phase from image and diffraction plane pictures." Optik 35 (1972): 237-246.

    Parameters
    ----------
    field            : torch.cfloat
                       Complex field (MxN).
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    slm_range        : float
                       Typically this is equal to two pi. See odak.wave.adjust_phase_only_slm_range() for more.
    propagation_type : str
                       Type of the propagation (see odak.learn.wave.propagate_beam).

    Returns
    -------
    hologram         : torch.cfloat
                       Calculated complex hologram.
    reconstruction   : torch.cfloat
                       Calculated reconstruction using calculated hologram. 
    """
    k = wavenumber(wavelength)
    reconstruction = field
    for i in range(n_iterations):
        hologram = propagate_beam(
            reconstruction, k, -distance, dx, wavelength, propagation_type)
        reconstruction = propagate_beam(
            hologram, k, distance, dx, wavelength, propagation_type)
        reconstruction = set_amplitude(reconstruction, field)
    reconstruction = propagate_beam(
        hologram, k, distance, dx, wavelength, propagation_type)
    return hologram, reconstruction


def stochastic_gradient_descent(target, wavelength, distance, pixel_pitch, propagation_type = 'Bandlimited Angular Spectrum', n_iteration = 100, loss_function = None, learning_rate = 0.1):
    """
    Definition to generate phase and reconstruction from target image via stochastic gradient descent.

    Parameters
    ----------
    target                    : torch.Tensor
                                Target field amplitude [m x n].
                                Keep the target values between zero and one.
    wavelength                : double
                                Set if the converted array requires gradient.
    distance                  : double
                                Hologram plane distance wrt SLM plane.
    pixel_pitch               : float
                                SLM pixel pitch in meters.
    propagation_type          : str
                                Type of the propagation (see odak.learn.wave.propagate_beam()).
    n_iteration:              : int
                                Number of iteration.
    loss_function:            : function
                                If none it is set to be l2 loss.
    learning_rate             : float
                                Learning rate.

    Returns
    -------
    hologram                  : torch.Tensor
                                Phase only hologram as torch array

    reconstruction_intensity  : torch.Tensor
                                Reconstruction as torch array

    """
    phase = torch.randn_like(target, requires_grad = True)
    k = wavenumber(wavelength)
    optimizer = torch.optim.Adam([phase], lr = learning_rate)
    if type(loss_function) == type(None):
        loss_function = torch.nn.MSELoss()
    t = tqdm(range(n_iteration), leave = False, dynamic_ncols = True)
    for i in t:
        optimizer.zero_grad()
        hologram = generate_complex_field(1., phase)
        reconstruction = propagate_beam(
                                        hologram, 
                                        k, 
                                        distance, 
                                        pixel_pitch, 
                                        wavelength, 
                                        propagation_type, 
                                        zero_padding = [True, False, True]
                                       )
        reconstruction_intensity = calculate_amplitude(reconstruction) ** 2
        loss = loss_function(reconstruction_intensity, target)
        description = "Loss:{:.4f}".format(loss.item())
        loss.backward(retain_graph = True)
        optimizer.step()
        t.set_description(description)
    logging.warning(description)
    torch.no_grad()
    hologram = generate_complex_field(1., phase)
    reconstruction = propagate_beam(
                                    hologram, 
                                    k, 
                                    distance, 
                                    pixel_pitch, 
                                    wavelength, 
                                    propagation_type, 
                                    zero_padding = [True, False, True]
                                   )
    return hologram, reconstruction


def point_wise(target, wavelength, distance, dx, device, lens_size=401):
    """
    Naive point-wise hologram calculation method. For more information, refer to Maimone, Andrew, Andreas Georgiou, and Joel S. Kollin. "Holographic near-eye displays for virtual and augmented reality." ACM Transactions on Graphics (TOG) 36.4 (2017): 1-16.

    Parameters
    ----------
    target           : torch.float
                       float input target to be converted into a hologram (Target should be in range of 0 and 1).
    wavelength       : float
                       Wavelength of the electric field.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    device           : torch.device
                       Device type (cuda or cpu)`.
    lens_size        : int
                       Size of lens for masking sub holograms(in pixels).

    Returns
    -------
    hologram         : torch.cfloat
                       Calculated complex hologram.
    """
    target = zero_pad(target)
    nx, ny = target.shape
    k = wavenumber(wavelength)
    ones = torch.ones(target.shape, requires_grad=False).to(device)
    x = torch.linspace(-nx/2, nx/2, nx).to(device)
    y = torch.linspace(-ny/2, ny/2, ny).to(device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = (X**2+Y**2)**0.5
    mask = (torch.abs(Z) <= lens_size)
    mask[mask > 1] = 1
    fz = quadratic_phase_function(nx, ny, k, focal=-distance, dx=dx).to(device)
    A = torch.nan_to_num(target**0.5, nan=0.0)
    fz = mask*fz
    FA = torch.fft.fft2(torch.fft.fftshift(A))
    FFZ = torch.fft.fft2(torch.fft.fftshift(fz))
    H = torch.mul(FA, FFZ)
    hologram = torch.fft.ifftshift(torch.fft.ifft2(H))
    hologram = crop_center(hologram)
    return hologram


def shift_w_double_phase(phase, depth_shift, pixel_pitch, wavelength, propagation_type='Transfer Function Fresnel', kernel_length=4, sigma=0.5, amplitude=None):
    """
    Shift a phase-only hologram by propagating the complex hologram and double phase principle. Coded following in [here](https://github.com/liangs111/tensor_holography/blob/6fdb26561a4e554136c579fa57788bb5fc3cac62/optics.py#L131-L207) and Shi, L., Li, B., Kim, C., Kellnhofer, P., & Matusik, W. (2021). Towards real-time photorealistic 3D holography with deep neural networks. Nature, 591(7849), 234-239.

    Parameters
    ----------
    phase            : torch.tensor
                       Phase value of a phase-only hologram.
    depth_shift      : float
                       Distance in meters.
    pixel_pitch      : float
                       Pixel pitch size in meters.
    wavelength       : float
                       Wavelength of light.
    propagation_type : str
                       Beam propagation type. For more see odak.learn.wave.propagate_beam().
    kernel_length    : int
                       Kernel length for the Gaussian blur kernel.
    sigma            : float
                       Standard deviation for the Gaussian blur kernel.
    amplitude        : torch.tensor
                       Amplitude value of a complex hologram.
    """
    if type(amplitude) == type(None):
        amplitude = torch.ones_like(phase)
    hologram = generate_complex_field(amplitude, phase)
    k = wavenumber(wavelength)
    hologram_padded = zero_pad(hologram)
    shifted_field_padded = propagate_beam(
                                          hologram_padded,
                                          k,
                                          depth_shift,
                                          pixel_pitch,
                                          wavelength,
                                          propagation_type
                                         )
    shifted_field = crop_center(shifted_field_padded)
    phase_shift = torch.exp(torch.tensor([-2 * torch.pi * depth_shift / wavelength]).to(phase.device))
    shift = torch.cos(phase_shift) + 1j * torch.sin(phase_shift)
    shifted_complex_hologram = shifted_field * shift

    if kernel_length > 0 and sigma >0:
        blur_kernel = generate_2d_gaussian(
                                           [kernel_length, kernel_length],
                                           [sigma, sigma]
                                          ).to(phase.device)
        blur_kernel = blur_kernel.unsqueeze(0)
        blur_kernel = blur_kernel.unsqueeze(0)
        field_imag = torch.imag(shifted_complex_hologram)
        field_real = torch.real(shifted_complex_hologram)
        field_imag = field_imag.unsqueeze(0)
        field_imag = field_imag.unsqueeze(0)
        field_real = field_real.unsqueeze(0)
        field_real = field_real.unsqueeze(0)
        field_imag = torch.nn.functional.conv2d(field_imag, blur_kernel, padding='same')
        field_real = torch.nn.functional.conv2d(field_real, blur_kernel, padding='same')
        shifted_complex_hologram = torch.complex(field_real, field_imag)
        shifted_complex_hologram = shifted_complex_hologram.squeeze(0)
        shifted_complex_hologram = shifted_complex_hologram.squeeze(0)

    shifted_amplitude = calculate_amplitude(shifted_complex_hologram)
    shifted_amplitude = shifted_amplitude / torch.amax(shifted_amplitude, [0,1])

    shifted_phase = calculate_phase(shifted_complex_hologram)
    phase_zero_mean = shifted_phase - torch.mean(shifted_phase)

    phase_offset = torch.arccos(shifted_amplitude)
    phase_low = phase_zero_mean - phase_offset
    phase_high = phase_zero_mean + phase_offset

    phase_only = torch.zeros_like(phase)
    phase_only[0::2, 0::2] = phase_low[0::2, 0::2]
    phase_only[0::2, 1::2] = phase_high[0::2, 1::2]
    phase_only[1::2, 0::2] = phase_high[1::2, 0::2]
    phase_only[1::2, 1::2] = phase_low[1::2, 1::2]
    return phase_only
