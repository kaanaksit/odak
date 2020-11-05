from odak import np
import torch, torch.fft

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
    nv, nu = field.shape
    x      = torch.linspace(-nv*dx,nv*dx,nv)
    y      = torch.linspace(-nu*dx,nu*dx,nu)
    X,Y    = torch.meshgrid(x,y)
    k      = torch.tensor(k, dtype=torch.complex128)
    Z      = X**2+Y**2
    if propagation_type == 'IR Fresnel':
       h      = 1./(1j*wavelength*distance)*torch.exp(1j*k*0.5/distance*Z)
       h      = torch.fft.fft2(torch.fft.fftshift(h))*pow(dx,2)
       U1     = torch.fft.fft2(torch.fft.fftshift(field))
       U2     = h*U1
       result = torch.fft.ifftshift(torch.fft.ifft2(U2))
    elif propagation_type == 'TR Fresnel':
       h      = torch.exp(1j*k*distance)*torch.exp(-1j*np.pi*wavelength*distance*Z)
       h      = torch.fft.fftshift(h)
       U1     = torch.fft.fft2(torch.fft.fftshift(field))
       U2     = h*U1
       result = torch.fft.ifftshift(torch.fft.ifft2(U2))
    elif propagation_type == 'Fraunhofer':
       c      = 1./(1j*wavelength*distance)*torch.exp(1j*k*0.5/distance*Z)
       result = c*torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(field)))*pow(dx,2)
    result = result.squeeze(0)
    return result
