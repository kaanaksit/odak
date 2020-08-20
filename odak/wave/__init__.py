"""
``odak.raytracing``
===================
Provides necessary definitions for geometric optics. See "General Ray tracing procedure" from G.H. Spencerand M.V.R.K Murty for the theoratical explanation.

"""
# To get sub-modules.
from .parameters import *

def rayleigh_resolution(diameter,focal=None,wavelength=0.0005):
    """
    Definition to calculate rayleigh resolution limit of a lens with a certain focal length and an aperture. Lens is assumed to be focusing a plane wave at a focal distance.

    Parameter
    ---------
    diameter    : float
                  Diameter of a lens.
    focal       : float
                  Focal length of a lens, when focal length is provided, spatial resolution is provided at the focal plane. When focal length isn't provided angular resolution is provided.
    wavelength  : float
                  Wavelength of light.

    Returns
    --------
    resolution  : float
                  Resolvable angular or spatial spot size, see focal in parameters to know what to expect.

    """
    resolution = 1.22*wavelength/diameter
    if type(focal) != type(None):
        resolution *= focal
    return resolution
