from odak import np

def propagate_beam(field,k,distance,dx,wavelength,propagation_type='IR Fresnel'):
    """
    Definitions for Fresnel impulse respone (IR), Fresnel Transfer Function (TF), Fraunhofer diffraction in accordence with "Computational Fourier Optics" by David Vuelz.

    Parameters
    ==========
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
    propagation_type : str
                       Type of the propagation (IR Fresnel, TR Fresnel, Fraunhofer).

    Returns
    =======
    result           : np.complex
                       Final complex field (MxN).
    """
    nu,nv  = field.shape
    x      = np.linspace(-nv*dx,nv*dx,nv)
    y      = np.linspace(-nu*dx,nu*dx,nu)
    X,Y    = np.meshgrid(x,y)
    Z      = X**2+Y**2
    if propagation_type == 'IR Fresnel':
       h      = 1./(1j*wavelength*distance)*np.exp(1j*k*0.5/distance*Z)
       h      = np.fft.fft2(np.fft.fftshift(h))*pow(dx,2)
       U1     = np.fft.fft2(np.fft.fftshift(field))
       U2     = h*U1
       result = np.fft.ifftshift(np.fft.ifft2(U2))
    elif propagation_type == 'TR Fresnel':
       h      = np.exp(1j*k*distance)*np.exp(-1j*np.pi*wavelength*distance*Z)
       h      = np.fft.fftshift(h)
       U1     = np.fft.fft2(np.fft.fftshift(field))
       U2     = h*U1
       result = np.fft.ifftshift(np.fft.ifft2(U2))
    elif propagation_type == 'Fraunhofer':
       c      = 1./(1j*wavelength*distance)*np.exp(1j*k*0.5/distance*Z)
       result = c*np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))*pow(dx,2)
    return result
