import numpy as np


def double_convergence(nx, ny, k, r, dx):
    """
    A definition to generate initial phase for a Gerchberg-Saxton method. For more details consult Sun, Peng, et al. "Holographic near-eye display system based on double-convergence light Gerchberg-Saxton algorithm." Optics express 26.8 (2018): 10140-10151.

    Parameters
    ----------
    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    r          : float
                 The distance between location of a light source and an image plane.
    dx         : float
                 Pixel pitch.

    Returns
    -------
    function   : ndarray
                 Generated phase pattern for a Gerchberg-Saxton method.
    """
    size = [ny, nx]
    x = np.linspace(-size[0]*dx/2, size[0]*dx/2, size[0])
    y = np.linspace(-size[1]*dx/2, size[1]*dx/2, size[1])
    X, Y = np.meshgrid(x, y)
    Z = X**2+Y**2
    w = np.exp(1j*k*Z/r)
    return w


def quadratic_phase_function(nx, ny, k, focal=0.4, dx=0.001, offset=[0, 0]):
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
    offset     : list
                 Deviation from the center along X and Y axes.

    Returns
    -------
    function   : ndarray
                 Generated quadratic phase function.
    """
    size = [nx, ny]
    x = np.linspace(-size[0]*dx/2, size[0]*dx/2, size[0])-offset[1]*dx
    y = np.linspace(-size[1]*dx/2, size[1]*dx/2, size[1])-offset[0]*dx
    X, Y = np.meshgrid(x, y)
    Z = X**2+Y**2
    qwf = np.exp(1j*k*0.5*np.sin(Z/focal))
    return qwf


def prism_phase_function(nx, ny, k, angle, dx=0.001, axis='x'):
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
    -------
    prism      : ndarray
                 Generated phase function for a prism.
    """
    angle = np.radians(angle)
    size = [ny, nx]
    x = np.linspace(-size[0]*dx/2, size[0]*dx/2, size[0])
    y = np.linspace(-size[1]*dx/2, size[1]*dx/2, size[1])
    X, Y = np.meshgrid(x, y)
    if axis == 'y':
        prism = np.exp(-1j*k*np.sin(angle)*Y)
    elif axis == 'x':
        prism = np.exp(-1j*k*np.sin(angle)*X)
    return prism


def linear_grating(nx, ny, every=2, add=3.14, axis='x'):
    """
    A definition to generate a linear grating.

    Parameters
    ----------
    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    every      : int
                 Add the add value at every given number.
    add        : float
                 Angle to be added.
    axis       : string
                 Axis eiter X,Y or both.

    Returns
    -------
    field      : ndarray
                 Linear grating term.
    """
    grating = np.zeros((nx, ny), dtype=np.complex64)
    if axis == 'x':
        grating[::every, :] = np.exp(1j*add)
    if axis == 'y':
        grating[:, ::every] = np.exp(1j*add)
    if axis == 'xy':
        checker = np.indices((nx, ny)).sum(axis=0) % every
        checker += 1
        checker = checker % 2
        grating = np.exp(1j*checker*add)
    return grating
