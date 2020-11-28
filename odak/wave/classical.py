from odak import np
from .__init__ import wavenumber,produce_phase_only_slm_pattern, calculate_amplitude,set_amplitude

def propagate_beam(field,k,distance,dx,wavelength,propagation_type='IR Fresnel'):
    """
    Definitions for Fresnel Impulse Respone (IR), Angular Spectrum (AS), Bandlimited Angular Spectrum (BAS), Fresnel Transfer Function (TF), Fraunhofer diffraction in accordence with "Computational Fourier Optics" by David Vuelz. For more on Bandlimited Fresnel impulse response also known as Bandlimited Angular Spectrum method see "Band-limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields".

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
    propagation_type : str
                       Type of the propagation (IR Fresnel, Angular Spectrum, Bandlimited Angular Spectrum, TR Fresnel, Fraunhofer).

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
    if propagation_type == 'Angular Spetrum':
       h      = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
       h      = np.fft.fft2(np.fft.fftshift(h))*pow(dx,2)
       U1     = np.fft.fft2(np.fft.fftshift(field))
       U2     = h*U1
       result = np.fft.ifftshift(np.fft.ifft2(U2))
    elif propagation_type == 'IR Fresnel':
       h      = np.exp(1j*k*distance)/(1j*wavelength*distance)*np.exp(1j*k/2/distance*Z)
       h      = np.fft.fft2(np.fft.fftshift(h))*pow(dx,2)
       U1     = np.fft.fft2(np.fft.fftshift(field))
       U2     = h*U1
       result = np.fft.ifftshift(np.fft.ifft2(U2))
    elif propagation_type == 'Bandlimited Angular Spectrum':
       h      = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
       h      = np.fft.fft2(np.fft.fftshift(h))*pow(dx,2)
       flimx  = int(1/(((2*distance*(1./(nu)))**2+1)**0.5*wavelength))
       flimy  = int(1/(((2*distance*(1./(nv)))**2+1)**0.5*wavelength))
       mask   = np.zeros((nu,nv),dtype=np.complex64)
       mask   = (np.abs(X)<flimx) & (np.abs(Y)<flimy)
       mask   = set_amplitude(h,mask)
       U1     = np.fft.fft2(np.fft.fftshift(field))
       U2     = mask*U1
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

def gerchberg_saxton(field,n_iterations,distance,dx,wavelength,slm_range=6.28,propagation_type='IR Fresnel'):
    """
    Definition to compute a hologram using an iterative method called Gerchberg-Saxton phase retrieval algorithm. For more on the method, see: Gerchberg, Ralph W. "A practical algorithm for the determination of phase from image and diffraction plane pictures." Optik 35 (1972): 237-246.

    Parameters
    ----------
    field            : np.complex
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
    hologram         : np.complex
                       Calculated complex hologram.
    reconstruction   : np.complex
                       Calculated reconstruction using calculated hologram. 
    """
    k              = wavenumber(wavelength)
    reconstruction = np.copy(field)
    for i in range(n_iterations):
        hologram       = propagate_beam(reconstruction,k,-distance,dx,wavelength,propagation_type)
        hologram       = produce_phase_only_slm_pattern(hologram,slm_range)
        reconstruction = propagate_beam(hologram,k,distance,dx,wavelength,propagation_type)
        reconstruction = set_amplitude(hologram,field)
    reconstruction = propagate_beam(hologram,k,distance,dx,wavelength,propagation_type)
    return hologram,reconstruction
