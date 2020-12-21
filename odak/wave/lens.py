from odak import np

def quadratic_phase_function(nx,ny,k,focal=0.4,dx=0.001):
    """ 
    A definition to generate 2D quadratic phase function, which is typically use to represent lenses.

    Parameters
    ----------
    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    focal      : float
                 Focal length of the quadratic phase function.
    dx         : float
                 Pixel pitch.

    Returns
    ---------
    function   : ndarray
                 Generated quadratic phase function.
    """
    size = [ny,nx]
    x    = np.linspace(-size[0]*dx/2,size[0]*dx/2,size[0])
    y    = np.linspace(-size[1]*dx/2,size[1]*dx/2,size[1])
    X,Y  = np.meshgrid(x,y)
    Z    = X**2+Y**2
    qwf  = np.exp(1j*k*0.5*np.sin(Z/focal))
    return qwf

def prism_phase_function(nx,ny,k,angle,dx=0.001,axis='x'):
    """
    A definition to generate 2D phase function that represents a prism. See Goodman's Introduction to Fourier Optics book for more.

    Parameters
    ----------
    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    angle      : float
                 Tilt angle of the prism in degrees.
    dx         : float
                 Pixel pitch.
    axis       : str
                 Axis of the prism.

    Returns
    ----------
    prism      : ndarray
                 Generated phase function for a prism.
    """
    angle = np.radians(angle)
    size  = [ny,nx]
    x     = np.linspace(-size[0]*dx/2,size[0]*dx/2,size[0])
    y     = np.linspace(-size[1]*dx/2,size[1]*dx/2,size[1])
    X,Y   = np.meshgrid(x,y)
    if axis == 'y':
        prism = np.exp(-1j*k*np.sin(angle)*Y)
    elif axis == 'x':
        prism = np.exp(-1j*k*np.sin(angle)*X)
    return prism

def freeform(nx,ny,k,distances,dx=0.001):
    """
    A definition to generate a freeform field pattern from a depth map.

    Parameters
    ----------
    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    distances  : ndarray
                 Depth map.
    dx         : float
                 Pixel pitch.

    Returns
    ---------
    field      : ndarray
                 Generated pattern.
    """
    size  = [ny,nx]
    x     = np.linspace(-size[0]*dx/2,size[0]*dx/2,size[0])
    y     = np.linspace(-size[1]*dx/2,size[1]*dx/2,size[1])
    X,Y   = np.meshgrid(x,y)
    Z     = X**2+Y**2
    field = np.exp(1j*k*0.5*np.sin(Z/distances))
    return field

def plane_tilt(nx,ny,k,focals,dx=0.001,axis='x'):
    """
    A definition to tilt a complex field.

    Parameters
    ----------
    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    focals     : list
                 Focus ranges, two for X and two for Y, in total four numbers. Make sure to pass nonzero numbers.
    dx         : float
                 Pixel pitch.
    axis       : str
                 Which state to tilt, x, y or xy.

    Returns
    ----------
    field      : ndarray
                 Field to tilt a plane.
    """
    size       = [ny,nx]
    x          = np.linspace(-size[0]*dx/2,size[0]*dx/2,size[0])
    y          = np.linspace(-size[1]*dx/2,size[1]*dx/2,size[1])
    X,Y        = np.meshgrid(x,y)
    Z          = X**2+Y**2
    if np.all((focals==0)):
        raise Exception("Focals must be non zero.")
    focal_x    = np.geomspace(focals[0],focals[1],size[0])
    focal_y    = np.geomspace(focals[2],focals[3],size[1])
    FX,FY      = np.meshgrid(focal_x,focal_y)
    field      = np.ones((nx,ny),dtype=np.complex64)
    if axis == 'x' or axis == 'xy':
        field *= np.exp(1j*k*0.5*np.sin(Z/FX))
    if axis == 'y' or axis == 'xy':
        field *= np.exp(1j*k*0.5*np.sin(Z/FY))
    return field
