from odak import np
from odak.tools import nufft2,nuifft2
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
    if propagation_type == 'Rayleigh-Sommerfeld':
        result = rayleigh_sommerfeld(field,k,distance,dx,wavelength)
    elif propagation_type == 'Angular Spectrum':
        result = angular_spectrum(field,k,distance,dx,wavelength)
    elif propagation_type == 'IR Fresnel':
        result = impulse_response_fresnel(field,k,distance,dx,wavelength)
    elif propagation_type == 'Bandlimited Angular Spectrum':
        result = band_limited_angular_spectrum(field,k,distance,dx,wavelength)
    elif propagation_type == 'Bandextended Angular Spectrum':
        result = band_extended_angular_spectrum(field,k,distance,dx,wavelength)
    elif propagation_type == 'Adaptive Sampling Angular Spectrum':
        result = adaptive_sampling_angular_spectrum(field,k,distance,dx,wavelength)
    elif propagation_type == 'TR Fresnel':
        result = transfer_function_fresnel(field,k,distance,dx,wavelength)
    elif propagation_type == 'Fraunhofer':
        result = fraunhofer(field,k,distance,dx,wavelength)
    elif propagation_type == 'Fraunhofer Inverse':
        result = fraunhofer_inverse(field,k,distance,dx,wavelength)
    else:
        raise Exception("Unknown propagation type selected.")
    return result

def adaptive_sampling_angular_spectrum(field,k,distance,dx,wavelength):
    """
    A definition to calculate adaptive sampling angular spectrum based beam propagation. For more Zhang, Wenhui, Hao Zhang, and Guofan Jin. "Adaptive-sampling angular spectrum method with full utilization of space-bandwidth product." Optics Letters 45.16 (2020): 4416-4419.

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
    raise Exception("Adaptive sampling angular spectrum method is not yet stable. See issue https://github.com/kunguz/odak/issues/13")
    iflag = -1
    eps   = 10**(-12)
    nu,nv = field.shape
    l     = nu*dx
    x     = np.linspace(-l/2,l/2,nu)
    y     = np.linspace(-l/2,l/2,nv)
    X,Y   = np.meshgrid(x,y)
    fx    = np.linspace(-1./2./dx,1./2./dx,nu)
    fy    = np.linspace(-1./2./dx,1./2./dx,nv)
    FX,FY = np.meshgrid(fx,fy)
    forig = 1./2./dx
    fc2   = 1./2*(nu/wavelength/np.abs(distance))**0.5
    ss    = np.abs(fc2)/forig
    zc    = nu*dx**2/wavelength
    K     = nu/2/np.amax(np.abs(fx))
    if np.abs(distance) <= zc*2:
        nnu2  = nu
        nnv2  = nv
        fxn   = np.linspace(-1./2./dx,1./2./dx,nnu2)
        fyn   = np.linspace(-1./2./dx,1./2./dx,nnv2)
    else:
        nnu2  = nu
        nnv2  = nv
        fxn   = np.linspace(-fc2,fc2,nnu2)
        fyn   = np.linspace(-fc2,fc2,nnv2)
    FXN,FYN   = np.meshgrid(fxn,fxn)
    Hn        = np.exp(1j*k*distance*(1-(FXN*wavelength)**2-(FYN*wavelength)**2)**0.5)
    FX        = FX/np.amax(FX)*np.pi
    FY        = FY/np.amax(FY)*np.pi
    t_2       = nufft2(field,FX*ss,FY*ss,size=[nnu2,nnv2],sign=iflag,eps=eps)
    FX        = FXN/np.amax(FXN)*np.pi
    FY        = FYN/np.amax(FYN)*np.pi
    result    = nufft2(Hn*t_2,FX*ss,FY*ss,size=[nu,nv],sign=-iflag,eps=eps)
    return result


def fraunhofer(field,k,distance,dx,wavelength):
    """
    A definition to calculate Fraunhofer based beam propagation.

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
    nu,nv  = field.shape
    l      = nu*dx
    l2     = wavelength*distance/dx
    dx2    = wavelength*distance/l
    fx     = np.linspace(-l2/2.,l2/2.,nu)
    fy     = np.linspace(-l2/2.,l2/2.,nv)
    FX,FY  = np.meshgrid(fx,fy)
    FZ     = FX**2+FY**2
    c      = np.exp(1j*k*distance)/(1j*wavelength*distance)*np.exp(1j*k/(2*distance)*FZ)
    result = c*np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))*dx**2
    return result

def fraunhofer_inverse(field,k,distance,dx,wavelength):
    """
    A definition to calculate Inverse Fraunhofer based beam propagation.

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
    distance = np.abs(distance)
    nu,nv    = field.shape
    l        = nu*dx
    l2       = wavelength*distance/dx
    dx2      = wavelength*distance/l
    fx       = np.linspace(-l2/2.,l2/2.,nu)
    fy       = np.linspace(-l2/2.,l2/2.,nv)
    FX,FY    = np.meshgrid(fx,fy)
    FZ       = FX**2+FY**2
    c        = np.exp(1j*k*distance)/(1j*wavelength*distance)*np.exp(1j*k/(2*distance)*FZ)
    result   = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field/dx**2/c )))
    return result

def band_limited_angular_spectrum(field,k,distance,dx,wavelength):
    """
    A definition to calculate bandlimited angular spectrum based beam propagation. For more Matsushima, Kyoji, and Tomoyoshi Shimobaba. "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." Optics express 17.22 (2009): 19662-19673.

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
    nu,nv  = field.shape
    l      = nu*dx
    l2     = wavelength*distance/dx
    dx2    = wavelength*distance/l
    fx     = np.linspace(-l2/2.,l2/2.,nu)
    fy     = np.linspace(-l2/2.,l2/2.,nv)
    FX,FY  = np.meshgrid(fx,fy)
    FZ     = FX**2+FY**2
    c      = 1./(1j*wavelength*distance)*np.exp(1j*k*(2./distance)*FZ)
    c      = np.exp(1j*k*distance)/(1j*wavelength*distance)*np.exp(1j*k/(2*distance)*FZ)
    result = c*np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))*dx**2
    return result

def band_limited_angular_spectrum(field,k,distance,dx,wavelength):
    """
    A definition to calculate bandlimited angular spectrum based beam propagation. For more Matsushima, Kyoji, and Tomoyoshi Shimobaba. "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." Optics express 17.22 (2009): 19662-19673.

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
    nu,nv  = field.shape
    x      = np.linspace(-nu/2*dx,nu/2*dx,nu)
    y      = np.linspace(-nv/2*dx,nv/2*dx,nv)
    X,Y    = np.meshgrid(x,y)
    Z      = X**2+Y**2
    h      = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
    h      = np.fft.fft2(np.fft.fftshift(h))*dx**2
    flimx  = np.ceil(1/(((2*distance*(1./(nu)))**2+1)**0.5*wavelength))
    flimy  = np.ceil(1/(((2*distance*(1./(nv)))**2+1)**0.5*wavelength))
    mask   = np.zeros((nu,nv),dtype=np.complex64)
    mask   = (np.abs(X)<flimx) & (np.abs(Y)<flimy)
    mask   = set_amplitude(h,mask)
    U1     = np.fft.fft2(np.fft.fftshift(field))
    U2     = mask*U1
    result = np.fft.ifftshift(np.fft.ifft2(U2))
    return result

def angular_spectrum(field,k,distance,dx,wavelength):
    """
    A definition to calculate angular spectrum based beam propagation.

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
    nu,nv  = field.shape
    x      = np.linspace(-nu/2*dx,nu/2*dx,nu)
    y      = np.linspace(-nv/2*dx,nv/2*dx,nv)
    X,Y    = np.meshgrid(x,y)
    Z      = X**2+Y**2
    h      = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
    h      = np.fft.fft2(np.fft.fftshift(h))*dx**2
    U1     = np.fft.fft2(np.fft.fftshift(field))
    U2     = h*U1
    result = np.fft.ifftshift(np.fft.ifft2(U2))
    return result

def angular_spectrum(field,k,distance,dx,wavelength):
    """
    A definition to calculate angular spectrum based beam propagation.

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
    nu,nv  = field.shape
    x      = np.linspace(-nu/2*dx,nu/2*dx,nu)
    y      = np.linspace(-nv/2*dx,nv/2*dx,nv)
    X,Y    = np.meshgrid(x,y)
    Z      = X**2+Y**2
    h      = 1./(1j*wavelength*distance)*np.exp(1j*k*(distance+Z/2/distance))
    h      = np.fft.fft2(np.fft.fftshift(h))*dx**2
    U1     = np.fft.fft2(np.fft.fftshift(field))
    U2     = h*U1
    result = np.fft.ifftshift(np.fft.ifft2(U2))
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
    nu,nv  = field.shape
    x      = np.linspace(-nu/2*dx,nu/2*dx,nu)
    y      = np.linspace(-nv/2*dx,nv/2*dx,nv)
    X,Y    = np.meshgrid(x,y)
    Z      = X**2+Y**2
    h      = np.exp(1j*k*distance)/(1j*wavelength*distance)*np.exp(1j*k/2/distance*Z)
    h      = np.fft.fft2(np.fft.fftshift(h))*dx**2
    U1     = np.fft.fft2(np.fft.fftshift(field))
    U2     = h*U1
    result = np.fft.ifftshift(np.fft.ifft2(U2))
    return result

def transfer_function_fresnel(field,k,distance,dx,wavelength):
    """
    A definition to calculate convolution based Fresnel approximation for beam propagation.

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
    nu,nv  = field.shape
    fx     = np.linspace(-1./2./dx,1./2./dx,nu)
    fy     = np.linspace(-1./2./dx,1./2./dx,nv)
    FX,FY  = np.meshgrid(fx,fy)
    H      = np.exp(1j*k*distance*(1-(FX*wavelength)**2-(FY*wavelength)**2)**0.5)
    U1     = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
    U2     = H*U1
    result = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(U2)))
    return result

def band_extended_angular_spectrum(field,k,distance,dx,wavelength):
    """
    A definition to calculate bandextended angular spectrum based beam propagation. For more Zhang, Wenhui, Hao Zhang, and Guofan Jin. "Band-extended angular spectrum method for accurate diffraction calculation in a wide propagation range." Optics Letters 45.6 (2020): 1543-1546.

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
    iflag = -1
    eps   = 10**(-12)
    nu,nv = field.shape
    l     = nu*dx
    x     = np.linspace(-l/2,l/2,nu)
    y     = np.linspace(-l/2,l/2,nv)
    X,Y   = np.meshgrid(x,y)
    Z     = X**2+Y**2
    fx    = np.linspace(-1./2./dx,1./2./dx,nu)
    fy    = np.linspace(-1./2./dx,1./2./dx,nv)
    FX,FY = np.meshgrid(fx,fy)
    K     = nu/2/np.amax(fx)
    fcn   = 1./2*(nu/wavelength/np.abs(distance))**0.5
    ss    = np.abs(fcn)/np.amax(np.abs(fx))
    zc    = nu*dx**2/wavelength
    if np.abs(distance) < zc:
        fxn = fx
        fyn = fy
    else:
        fxn = fx*ss
        fyn = fy*ss
    FXN,FYN   = np.meshgrid(fxn,fxn)
    Hn        = np.exp(1j*k*distance*(1-(FXN*wavelength)**2-(FYN*wavelength)**2)**0.5)
    X         = X/np.amax(X)*np.pi
    Y         = Y/np.amax(Y)*np.pi
    t_asmNUFT = nufft2(field,X*ss,Y*ss,sign=iflag,eps=eps)
    result    = nuifft2(Hn*t_asmNUFT,X*ss,Y*ss,sign=-iflag,eps=eps)
    return result

def rayleigh_sommerfeld(field,k,distance,dx,wavelength):
    """
    Definition to compute beam propagation using Rayleigh-Sommerfeld's diffraction formula (Huygens-Fresnel Principle). For more see Section 3.5.2 in Goodman, Joseph W. Introduction to Fourier optics. Roberts and Company Publishers, 2005.

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
    nu,nv     = field.shape
    x         = np.linspace(-nv*dx/2,nv*dx/2,nv)
    y         = np.linspace(-nu*dx/2,nu*dx/2,nu)
    X,Y       = np.meshgrid(x,y)
    Z         = X**2+Y**2
    result    = np.zeros(field.shape,dtype=np.complex64)
    direction = int(distance/np.abs(distance))
    for i in range(nu):
        for j in range(nv):
            r01      = np.sqrt(distance**2+(X-X[i,j])**2+(Y-Y[i,j])**2)*direction
            cosnr01  = np.cos(distance/r01)
            result  += field[i,j]*np.exp(1j*k*r01)/r01*cosnr01
    result *= 1./(1j*wavelength)
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
