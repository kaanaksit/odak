import numpy as np
import torch
import torch.fft
import logging
from .util import set_amplitude, produce_phase_only_slm_pattern, generate_complex_field, calculate_amplitude, calculate_phase
from .lens import quadratic_phase_function
from .util import wavenumber
from ..tools import zero_pad, crop_center, generate_2d_gaussian
from tqdm import tqdm


def propagate_beam(
                   field, 
                   k, 
                   distance, 
                   dx, 
                   wavelength, 
                   propagation_type='Bandlimited Angular Spectrum', 
                   kernel = None, 
                   zero_padding = [False, False, False],
                   aperture = 1.
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
                       The options are Transfer Function Fresnel, Angular Spectrum, Bandlimited Angular Spectrum, Fraunhofer.
    kernel           : torch.complex
                       Custom complex kernel.
    zero_padding     : list
                       Zero padding the input field if the first item in the list set True.
                       Zero padding in the Fourier domain if the second item in the list set to True.
                       Cropping the result with half resolution if the third item in the list is set to true. 
                       Note that in Fraunhofer propagation, setting the second item True or False will have no effect.

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
    elif propagation_type == 'Transfer Function Fresnel':
        result = transfer_function_fresnel(field, k, distance, dx, wavelength, zero_padding[1], aperture = aperture)
    elif propagation_type == 'custom':
        result = custom(field, kernel, zero_padding[1], aperture = aperture)
    elif propagation_type == 'Fraunhofer':
        result = fraunhofer(field, k, distance, dx, wavelength)
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
                           scale = 1
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
   

    Returns
    -------
    kernel             : torch.tensor
                         Complex kernel for the given propagation type.
    """
    if propagation_type == 'Bandlimited Angular Spectrum':
        kernel = get_band_limited_angular_spectrum_kernel(nu, nv, dx, wavelength, distance, device)
    elif propagation_type == 'Angular Spectrum':
        kernel = get_angular_spectrum_kernel(nu, nv, dx, wavelength, distance, device)
    elif propagation_type == 'Transfer Function Fresnel':
        kernel = get_transfer_function_fresnel_kernel(nu, nv, dx, wavelength, distance, device)
    else:
        logging.warning('Propagation type not recognized')
        assert True == False
    return kernel



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
    c = 1./(1j*wavelength*distance)*torch.exp(1j*k*0.5/distance*Z)
    c = c.to(field.device)
    result = c * \
             torch.fft.ifftshift(torch.fft.fft2(
             torch.fft.fftshift(field)))*pow(dx, 2)
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
        H = torch.zeros(field.shape).to(field.device)
    else:
        H = kernel * aperture
    U1 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field))) * aperture
    if zero_padding == False:
        U2 = H * U1
    elif zero_padding == True:
        U2 = zero_pad(H * U1)
    result = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(U2)))
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
    H                  : float
                         Complex kernel in Fourier domain.
    """
    distance = torch.tensor([distance]).to(device)
    fx = torch.linspace(-1. / 2. /dx, 1. / 2. /dx, nu, dtype = torch.float32, device = device)
    fy = torch.linspace(-1. / 2. /dx, 1. / 2. /dx, nv, dtype = torch.float32, device = device)
    FY, FX = torch.meshgrid(fx, fy, indexing = 'ij')
    k = wavenumber(wavelength)
    H = torch.exp(1j* k * distance * (1 - (FX * wavelength) ** 2 - (FY * wavelength) ** 2) ** 0.5).to(device)
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
    H = get_transfer_function_fresnel_kernel(
                                             field.shape[-2], 
                                             field.shape[-1], 
                                             dx = dx, 
                                             wavelength = wavelength, 
                                             distance = distance, 
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
    fx = torch.linspace(-1. /2. / dx, 1. / 2. / dx, nu, dtype = torch.float32, device = device)
    fy = torch.linspace(-1. /2. / dx, 1. / 2. / dx, nv, dtype = torch.float32, device = device)
    FY, FX = torch.meshgrid(fx, fy, indexing='ij')
    H = torch.exp(1j  * distance * (2 * (np.pi * (1 / wavelength) * torch.sqrt(1. - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))))
    H = H.to(device)
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
    H = get_angular_spectrum_kernel(
                                    field.shape[-2], 
                                    field.shape[-1], 
                                    dx = dx, 
                                    wavelength = wavelength, 
                                    distance = distance, 
                                    device = field.device
                                   )
    result = custom(field, H, zero_padding = zero_padding, aperture = aperture)
    return result


def get_band_limited_angular_spectrum_kernel(nu, nv, dx = 8e-6, wavelength = 515e-9, distance = 0., device = torch.device('cpu')):
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
    H                  : float
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
    HH_exp = 2 * np.pi * torch.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))
    distance = torch.tensor([distance], device = device)
    H_exp = torch.mul(HH_exp, distance)
    fx_max = 1 / torch.sqrt((2 * distance * (1 / x))**2 + 1) / wavelength
    fy_max = 1 / torch.sqrt((2 * distance * (1 / y))**2 + 1) / wavelength
    H_filter = ((torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)).clone().detach()
    H = generate_complex_field(H_filter, H_exp)
    return H


def band_limited_angular_spectrum(field, k, distance, dx, wavelength, zero_padding = False, aperture = 1.):
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
    H = get_band_limited_angular_spectrum_kernel(
                                                 field.shape[-2], 
                                                 field.shape[-1], 
                                                 dx = dx, 
                                                 wavelength = wavelength, 
                                                 distance = distance, 
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
        hologram, _ = produce_phase_only_slm_pattern(hologram, slm_range)
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
    print(description)
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
    phase_shift = torch.exp(torch.tensor([-2 * np.pi * depth_shift / wavelength]).to(phase.device))
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
