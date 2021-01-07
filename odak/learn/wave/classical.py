from odak import np
import torch, torch.fft
from .toolkit import fftshift, ifftshift
from .__init__ import set_amplitude, produce_phase_only_slm_pattern
from odak.wave import wavenumber

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
    nv, nu = field.shape[-1], field.shape[-2]
    x      = torch.linspace(-nv*dx/2, nv*dx/2, nv, dtype=torch.float64)
    y      = torch.linspace(-nu*dx/2, nu*dx/2, nu, dtype=torch.float64)
    Y, X   = torch.meshgrid(y, x)
    Z      = torch.pow(X,2) + torch.pow(Y,2)
    if propagation_type == 'IR Fresnel':
        result    = impulse_response_fresnel(field,k,distance,dx,wavelength)
    elif propagation_type == 'Bandlimited Angular Spectrum':
        h         = 1./(1j*wavelength*distance)*torch.exp(1j*k*(distance+Z/2/distance))  
        h         = torch.fft.fftn(fftshift(h)) * pow(dx, 2)
        h         = h.to(field.device)
        flimx     = np.ceil(1/(((2*distance*(1./(nv)))**2+1)**0.5*wavelength))
        flimy     = np.ceil(1/(((2*distance*(1./(nu)))**2+1)**0.5*wavelength))
        mask      = torch.zeros((nu,nv), dtype=torch.cfloat)
        mask[...] = torch.logical_and(torch.lt(torch.abs(X), flimx), torch.lt(torch.abs(Y), flimy))
        mask      = mask.to(field.device)
        mask      = set_amplitude(h, mask)
        U1        = torch.fft.fftn(fftshift(field))
        U2        = mask * U1 
        result    = ifftshift(torch.fft.ifftn(U2))
    elif propagation_type == 'TR Fresnel':
        h      = torch.exp(1j*k*distance)*torch.exp(-1j*np.pi*wavelength*distance*Z)
        h      = fftshift(h)
        h      = h.to(field.device)
        U1     = torch.fft.fftn(fftshift(field))
        U2     = h*U1
        result = ifftshift(torch.fft.ifftn(U2))
    elif propagation_type == 'Fraunhofer':
        c      = 1./(1j*wavelength*distance)*torch.exp(1j*k*0.5/distance*Z)
        c      = c.to(field.device)
        result = c*ifftshift(torch.fft.fftn(fftshift(field)))*pow(dx,2)
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
    nv, nu   = field.shape[-1], field.shape[-2]
    x        = torch.linspace(-nv*dx/2, nv*dx/2, nv, dtype=torch.float64)
    y        = torch.linspace(-nu*dx/2, nu*dx/2, nu, dtype=torch.float64)
    Y, X     = torch.meshgrid(y, x)
    Z        = torch.pow(X,2) + torch.pow(Y,2)
    distance = torch.FloatTensor([distance])
    h        = torch.exp(1j*k*distance)/(1j*wavelength*distance)*torch.exp(1j*k*0.5/distance*Z)
    h        = torch.fft.fftn(fftshift(h))*pow(dx,2)
    h        = h.to(field.device)
    U1       = torch.fft.fftn(fftshift(field))
    U2       = h*U1
    result   = ifftshift(torch.fft.ifftn(U2))
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
        hologram       = produce_phase_only_slm_pattern(hologram,slm_range)
        reconstruction = propagate_beam(hologram,k,distance,dx,wavelength,propagation_type)
        reconstruction = set_amplitude(hologram,field)
    reconstruction = propagate_beam(hologram,k,distance,dx,wavelength,propagation_type)
    return hologram,reconstruction
