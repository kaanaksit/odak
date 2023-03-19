import torch
import numpy as np
from .util import wavenumber
from ..tools import zero_pad, crop_center


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
    function   : torch.tensor
                 Generated quadratic phase function.
    """
    size = [nx, ny]
    x = torch.linspace(-size[0] * dx / 2, size[0] * dx / 2, size[0]) - offset[1] * dx
    y = torch.linspace(-size[1] * dx / 2, size[1] * dx / 2, size[1]) - offset[0] * dx
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = X**2 + Y**2
    qwf = torch.exp(-0.5j * k / focal * Z)
    return qwf


def prism_grating(nx, ny, k, angle, dx = 0.001, axis = 'x', phase_offset = 0.):
    """
    A definition to generate 2D phase function that represents a prism. See Goodman's Introduction to Fourier Optics book or Engström, David, et al. "Improved beam steering accuracy of a single beam with a 1D phase-only spatial light modulator." Optics express 16.22 (2008): 18275-18287. for more.

    Parameters
    ----------
    nx           : int
                   Size of the output along X.
    ny           : int
                   Size of the output along Y.
    k            : odak.wave.wavenumber
                   See odak.wave.wavenumber for more.
    angle        : float
                   Tilt angle of the prism in degrees.
    dx           : float
                   Pixel pitch.
    axis         : str
                   Axis of the prism.
    phase_offset : float
                   Phase offset in angles. Default is zero.

    Returns
    ----------
    prism        : torch.tensor
                   Generated phase function for a prism.
    """
    angle = torch.deg2rad(torch.tensor([angle]))
    phase_offset = torch.deg2rad(torch.tensor([phase_offset]))
    x = torch.arange(0, nx) * dx
    y = torch.arange(0, ny) * dx
    X, Y = torch.meshgrid(x, y, indexing='ij')
    if axis == 'y':
        phase = k * torch.sin(angle) * Y + phase_offset
        prism = torch.exp(-1j * phase)
    elif axis == 'x':
        phase = k * torch.sin(angle) * X + phase_offset
        prism = torch.exp(-1j * phase)
    return prism


def blazed_grating(nx, ny, levels = 2, axis = 'x'):
    """
    A defininition to generate a blazed grating (also known as ramp grating). For more consult de Blas, Mario García, et al. "High resolution 2D beam steerer made from cascaded 1D liquid crystal phase gratings." Scientific Reports 12.1 (2022): 5145 and Igasaki, Yasunori, et al. "High efficiency electrically-addressable phase-only spatial light modulator." optical review 6 (1999): 339-344.


    Parameters
    ----------
    nx           : int
                   Size of the output along X.
    ny           : int
                   Size of the output along Y.
    levels       : int
                   Number of pixels.
    axis         : str
                   Axis of glazed grating. It could be `x` or `y`.

    """
    if levels < 2:
        levels = 2
    x = (torch.abs(torch.arange(-nx, 0)) % levels) / levels * (2 * np.pi)
    y = (torch.abs(torch.arange(-ny, 0)) % levels) / levels * (2 * np.pi)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    if axis == 'x':
        blazed_grating = torch.exp(1j * X)
    elif axis == 'y':
        blazed_grating = torch.exp(1j * Y)
    return blazed_grating


def linear_grating(nx, ny, every = 2, add = None, axis = 'x'):
    """
    A definition to generate a linear grating. This could also be interpreted as two levels blazed grating. For more on blazed gratings see odak.learn.wave.blazed_grating() function.

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
    ----------
    field      : torch.tensor
                 Linear grating term.
    """
    if isinstance(add, type(None)):
        add = np.pi
    grating = torch.zeros((nx, ny), dtype=torch.complex64)
    if axis == 'x':
        grating[::every, :] = torch.exp(torch.tensor(1j*add))
    if axis == 'y':
        grating[:, ::every] = torch.exp(torch.tensor(1j*add))
    if axis == 'xy':
        checker = np.indices((nx, ny)).sum(axis=0) % every
        checker = torch.from_numpy(checker)
        checker += 1
        checker = checker % 2
        grating = torch.exp(1j*checker*add)
    return grating
