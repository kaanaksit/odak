"""
``odak.wave``
===================
Provides necessary definitions for merging geometric optics with wave theory and classical approaches in the wave theory as well. See "Introduction to Fourier Optcs" from Joseph Goodman for the theoratical explanation.

"""
# To get sub-modules.
from .vector import *
from .classical import *

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

def calculate_phase(field,deg=False):
    """ 
    Definition to calculate phase of a single or multiple given electric field(s).

    Parameters
    ----------
    field        : ndarray.complex or complex
                   Electric fields or an electric field.
    deg          : bool
                   If set True, the angles will be returned in degrees.

    Returns
    ----------
    phase        : float
                   Phase or phases of electric field(s) in radians.
    """
    phase = np.angle(field)
    if deg == True:
        phase *= 180./np.pi
    return phase

def calculate_amplitude(field):
    """ 
    Definition to calculate amplitude of a single or multiple given electric field(s).

    Parameters
    ----------
    field        : ndarray.complex or complex
                   Electric fields or an electric field.

    Returns
    ----------
    amplitude    : float
                   Amplitude or amplitudes of electric field(s).
    """
    amplitude = np.abs(field)
    return amplitude

def calculate_intensity(field):
    """
    Definition to calculate intensity of a single or multiple given electric field(s).

    Parameters
    ----------
    field        : ndarray.complex or complex
                   Electric fields or an electric field.

    Returns
    ----------
    intensity    : float
                   Intensity or intensities of electric field(s).
    """
    intensity = np.abs(field)**2
    return intensity

def wavenumber(wavelength):
    """
    Definition for calculating the wavenumber of a plane wave.

    Parameters
    ----------
    wavelength   : float
                   Wavelength of a wave in mm.

    Returns
    ----------
    k            : float
                   Wave number for a given wavelength.
    """
    k = 2*np.pi/wavelength
    return k

def rotationspeed(wavelength,c=3*10**11):
    """
    Definition for calculating rotation speed of a wave (w in A*e^(j(wt+phi))).

    Parameters
    ----------
    wavelength   : float
                   Wavelength of a wave in mm.
    c            : float
                   Speed of wave in mm/seconds. Default is the speed of light in the void!

    Returns
    ----------
    w            : float

    """
    f = c*wavelength
    w = 2*np.pi*f
    return w
