from odak import np
import torch, torch.fft
from .__init__ import set_amplitude, produce_phase_only_slm_pattern, generate_complex_field, calculate_amplitude
from odak.wave import wavenumber
from odak.learn.tools import zero_pad, crop_center
from tqdm import tqdm

def propagate_beam(field,k,distance,dx,wavelength,propagation_type='IR Fresnel'):
    """
    Definitions for Fresnel impulse respone (IR), Fresnel Transfer Function (TF), Fraunhofer diffraction in accordence with "Computational Fourier Optics" by David Vuelz.

    Parameters
    ==========
    field            : torch.complex128
                       Complex field (MxN).
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    propagation_type : str
                       Type of the propagation (IR Fresnel, TR Fresnel, Fraunhofer).

    Returns
    =======
    result           : torch.complex128
                       Final complex field (MxN).
    """
    if propagation_type == 'IR Fresnel':
        result = impulse_response_fresnel(field,k,distance,dx,wavelength)
    elif propagation_type == 'Bandlimited Angular Spectrum':
        result = band_limited_angular_spectrum(field,k,distance,dx,wavelength)
    elif propagation_type == 'TR Fresnel':
        result = transfer_function_fresnel(field,k,distance,dx,wavelength)
    elif propagation_type == 'Fraunhofer':
        nv, nu = field.shape[-1], field.shape[-2]
        x      = torch.linspace(-nv*dx/2, nv*dx/2, nv, dtype=torch.float64)
        y      = torch.linspace(-nu*dx/2, nu*dx/2, nu, dtype=torch.float64)
        Y, X   = torch.meshgrid(y, x)
        Z      = torch.pow(X,2) + torch.pow(Y,2)
        c      = 1./(1j*wavelength*distance)*torch.exp(1j*k*0.5/distance*Z)
        c      = c.to(field.device)
        result = c*torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(field)))*pow(dx,2)
    return result


def transfer_function_fresnel(field,k,distance,dx,wavelength):
    """
    A definition to calculate convolution based Fresnel approximation for beam propagation.

    Parameters
    ----------
    field            : troch.complex
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
    ---------
    result           : torch.complex
                       Final complex field (MxN).

    """
    distance  = torch.tensor([distance]).to(field.device)
    nv, nu    = field.shape[-1], field.shape[-2]
    fx        = torch.linspace(-1./2./dx,1./2./dx,nu,dtype=torch.float64).to(field.device)
    fy        = torch.linspace(-1./2./dx,1./2./dx,nv,dtype=torch.float64).to(field.device)
    FY, FX    = torch.meshgrid(fx, fy)
    H         = torch.exp(1j*k*distance*(1-(FX*wavelength)**2-(FY*wavelength)**2)**0.5)
    H         = H.to(field.device)
    U1        = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field)))
    U2        = H*U1
    result    = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(U2)))
    return result


def band_limited_angular_spectrum(field,k,distance,dx,wavelength):
    """
    A definition to calculate bandlimited angular spectrum based beam propagation. For more Matsushima, Kyoji, and Tomoyoshi Shimobaba. "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." Optics express 17.22 (2009): 19662-19673.

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
    =======
    result           : torch.complex
                       Final complex field (MxN).
    """
    assert True==False,"Refer to Issue 19 for more. This definition is unreliable."
    nv, nu    = field.shape[-1], field.shape[-2]
    x         = torch.linspace(-nv*dx/2, nv*dx/2, nv, dtype=torch.float64)
    y         = torch.linspace(-nu*dx/2, nu*dx/2, nu, dtype=torch.float64)
    Y, X      = torch.meshgrid(y, x)
    Z         = torch.pow(X,2) + torch.pow(Y,2)
    distance  = torch.FloatTensor([distance])
    h         = 1./(1j*wavelength*distance)*torch.exp(1j*k*(distance+Z/2/distance))
    h         = torch.fft.fft2(torch.fft.fftshift(h)) * pow(dx, 2)
    h         = h.to(field.device)
    flimx     = torch.ceil(1/(((2*distance*(1./(nv)))**2+1)**0.5*wavelength))
    flimy     = torch.ceil(1/(((2*distance*(1./(nu)))**2+1)**0.5*wavelength))
    mask      = torch.zeros((nu,nv), dtype=torch.cfloat).to(field.device)
    mask[...] = torch.logical_and(torch.lt(torch.abs(X), flimx), torch.lt(torch.abs(Y), flimy))
    mask      = set_amplitude(h, mask)
    U1        = torch.fft.fft2(torch.fft.fftshift(field))
    U2        = mask * U1
    result    = torch.fft.ifftshift(torch.fft.ifft2(U2))
    return result

def impulse_response_fresnel(field,k,distance,dx,wavelength):
    """
    A definition to calculate impulse response based Fresnel approximation for beam propagation.

    Parameters
    ----------
    field            : np.complex
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
    =======
    result           : np.complex
                       Final complex field (MxN).
    """
    assert True==False,"Refer to Issue 19 for more. This definition is unreliable."
    nv, nu   = field.shape[-1], field.shape[-2]
    x        = torch.linspace(-nu/2*dx,nu/2*dx,nu)
    y        = torch.linspace(-nv/2*dx,nv/2*dx,nv)
    X,Y      = torch.meshgrid(x,y)
    Z        = X**2+Y**2
    distance = torch.tensor([distance]).to(field.device)
    h        = torch.exp(1j*k*distance)/(1j*wavelength*distance)*torch.exp(1j*k/2/distance*Z)
    h        = torch.fft.fft2(torch.fft.fftshift(h))*dx**2
    h        = h.to(field.device)
    U1       = torch.fft.fft2(torch.fft.fftshift(field))
    U2       = h*U1
    result   = torch.fft.ifftshift(torch.fft.ifft2(U2))
    return result

def gerchberg_saxton(field,n_iterations,distance,dx,wavelength,slm_range=6.28,propagation_type='IR Fresnel'):
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
                       Type of the propagation (IR Fresnel, TR Fresnel, Fraunhofer).

    Result
    ---------
    hologram         : torch.cfloat
                       Calculated complex hologram.
    reconstruction   : torch.cfloat
                       Calculated reconstruction using calculated hologram. 
    """
    k              = wavenumber(wavelength)
    reconstruction = field
    for i in range(n_iterations):
        hologram       = propagate_beam(reconstruction,k,-distance,dx,wavelength,propagation_type)
        hologram,_     = produce_phase_only_slm_pattern(hologram,slm_range)
        reconstruction = propagate_beam(hologram,k,distance,dx,wavelength,propagation_type)
        reconstruction = set_amplitude(hologram,field)
    reconstruction = propagate_beam(hologram,k,distance,dx,wavelength,propagation_type)
    return hologram,reconstruction

def stochastic_gradient_descent(field,wavelength,distance,dx,resolution,propogation_type,n_iteration=100,loss_function=None,cuda=False,learning_rate=1.0):
    """
    Definition to generate phase and reconstruction from target image via stochastic gradient descent.

    Parameters
    ----------
    field                   : ndarray
                              Input field as Numpy array.
    wavelength              : double
                              Set if the converted array requires gradient.
    distance                : double
                              Hologaram plane distance wrt SLM plane
    dx                      : float
                              SLM pixel pitch
    resolution              : array
                              SLM resolution
    propogation type        : str
                              Type of the propagation (IR Fresnel, Angular Spectrum, Bandlimited Angular Spectrum, TR Fresnel, Fraunhofer)
    n_iteration:            : int
                              Max iteratation 
    loss_function:          : function
                              If none it is set to be l2 loss
    cuda                    : boolean
                              GPU enabled
    learning_rate           : float
                              Learning rate.

    Returns
    ----------
    hologram                : torch.Tensor
                              Phase only hologram as torch array

    reconstruction_intensity: torch.Tensor
                              Reconstruction as torch array

    """
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    device    = torch.device("cuda" if cuda else "cpu") 
    field     = field.float().to(device)
    phase     = torch.zeros((resolution),requires_grad=True,device=device)
    amplitude = torch.ones((resolution),requires_grad=False).to(device)
    k         = wavenumber(wavelength)
    optimizer = torch.optim.Adam([{'params': phase}],lr=learning_rate)
    if type(loss_function) == type(None):
        loss_function = torch.nn.MSELoss().to(device)
    t = tqdm(range(n_iteration),leave=False)
    for i in t:
        optimizer.zero_grad()
        hologram                 = generate_complex_field(amplitude,phase)
        hologram                 = zero_pad(hologram)
        reconstruction           = propagate_beam(hologram,k,distance,dx,wavelength,propogation_type)
        reconstruction           = crop_center(reconstruction)
        reconstruction_amplitude = calculate_amplitude(reconstruction).float()
        loss                     = loss_function(reconstruction_amplitude**2,field)
        loss.backward(retain_graph=True)
        optimizer.step()
        description              = "loss:{}".format(loss.item())
        t.set_description(description)
    hologram       = generate_complex_field(amplitude,phase)
    hologram       = zero_pad(hologram)
    reconstruction = propagate_beam(hologram,k,distance,dx,wavelength,propogation_type)
    reconstruction = crop_center(reconstruction)
    hologram       = crop_center(hologram)
    return hologram, reconstruction
