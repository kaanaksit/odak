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
    x   = np.linspace(-nx*dx/2,nx*dx/2,nx)
    y   = np.linspace(-ny*dx/2,ny*dx/2,ny)
    X,Y = np.meshgrid(x,y)
    Z   = X**2+Y**2
    qwf = np.exp(1j*k*0.5*Z/focal)
    return qwf
