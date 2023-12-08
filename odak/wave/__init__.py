"""
``odak.wave``

Provides necessary definitions for merging geometric optics with wave theory and classical approaches in the wave theory as well.
See "Introduction to Fourier Optcs" from Joseph Goodman for the theoratical explanation.

"""
# To get sub-modules.
from .vector import *
from .utils import *
from .classical import *
from .lens import *
from odak.tools import save_image


def rayleigh_resolution(diameter, focal=None, wavelength=0.0005):
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


def calculate_intensity(field):
    """
    Definition to calculate intensity of a single or multiple given electric field(s).

    Parameters
    ----------
    field        : ndarray.complex or complex
                   Electric fields or an electric field.

    Returns
    -------
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
    -------
    k            : float
                   Wave number for a given wavelength.
    """
    k = 2*np.pi/wavelength
    return k


def rotationspeed(wavelength, c=3*10**11):
    """
    Definition for calculating rotation speed of a wave (w in A*e^(j(wt+phi))).

    Parameters
    ----------
    wavelength   : float
                   Wavelength of a wave in mm.
    c            : float
                   Speed of wave in mm/seconds. Default is the speed of light in the void!

    Returns
    -------
    w            : float
                   Rotation speed.

    """
    f = c*wavelength
    w = 2*np.pi*f
    return w


def add_random_phase(field):
    """
    Definition for adding a random phase to a given complex field.

    Parameters
    ----------
    field        : np.complex64
                   Complex field.

    Returns
    -------
    new_field    : np.complex64
                   Complex field.
    """
    random_phase = np.pi*np.random.random(field.shape)
    new_field = add_phase(field, random_phase)
    return new_field


def add_phase(field, new_phase):
    """
    Definition for adding a phase to a given complex field.

    Parameters
    ----------
    field        : np.complex64
                   Complex field.
    new_phase    : np.complex64
                   Complex phase.

    Returns
    -------
    new_field    : np.complex64
                   Complex field.
    """
    phase = calculate_phase(field)
    amplitude = calculate_amplitude(field)
    new_field = amplitude*np.cos(phase+new_phase) + \
        1j*amplitude*np.sin(phase+new_phase)
    return new_field


def set_amplitude(field, amplitude):
    """
    Definition to keep phase as is and change the amplitude of a given field.

    Parameters
    ----------
    field        : np.complex64
                   Complex field.
    amplitude    : np.array or np.complex64
                   Amplitudes.

    Returns
    -------
    new_field    : np.complex64
                   Complex field.
    """
    amplitude = calculate_amplitude(amplitude)
    phase = calculate_phase(field)
    new_field = amplitude*np.cos(phase)+1j*amplitude*np.sin(phase)
    return new_field


def generate_complex_field(amplitude, phase):
    """
    Definition to generate a complex field with a given amplitude and phase.

    Parameters
    ----------
    amplitude         : ndarray
                        Amplitude of the field.
    phase             : ndarray
                        Phase of the field.

    Returns
    -------
    field             : ndarray
                        Complex field.
    """
    field = amplitude*np.cos(phase)+1j*amplitude*np.sin(phase)
    return field


def adjust_phase_only_slm_range(native_range, working_wavelength, native_wavelength):
    """
    Definition for calculating the phase range of the Spatial Light Modulator (SLM) for a given wavelength. Here you prove maximum angle as the lower bound is typically zero. If the lower bound isn't zero in angles, you can use this very same definition for calculating lower angular bound as well.

    Parameters
    ----------
    native_range       : float
                         Native range of the phase only SLM in radians (i.e. two pi).
    working_wavelength : float
                         Wavelength of the illumination source or some working wavelength.
    native_wavelength  : float
                         Wavelength which the SLM is designed for.

    Returns
    -------
    new_range          : float
                         Calculated phase range in radians.
    """
    new_range = native_range/working_wavelength*native_wavelength
    return new_range


def produce_phase_only_slm_pattern(hologram, slm_range, filename=None, bits=8, default_range=6.28, illumination=None):
    """
    Definition for producing a pattern for a phase only Spatial Light Modulator (SLM) using a given field.

    Parameters
    ----------
    hologram           : np.complex64
                         Input holographic field.
    slm_range          : float
                         Range of the phase only SLM in radians for a working wavelength (i.e. two pi). See odak.wave.adjust_phase_only_slm_range() for more.
    filename           : str
                         Optional variable, if provided the patterns will be save to given location.
    bits               : int
                         Quantization bits.
    default_range      : float 
                         Default range of phase only SLM.
    illumination       : np.ndarray
                         Spatial illumination distribution.

    Returns
    -------
    pattern            : np.complex64
                         Adjusted phase only pattern.
    hologram_digital   : np.int
                         Digital representation of the hologram.
    """
    #hologram_phase   = calculate_phase(hologram) % default_range
    hologram_phase = calculate_phase(hologram)
    hologram_phase = hologram_phase % slm_range
    hologram_phase /= slm_range
    hologram_phase *= 2**bits
    hologram_phase = hologram_phase.astype(np.int32)
    hologram_digital = np.copy(hologram_phase)
    if type(filename) != type(None):
        save_image(
            filename,
            hologram_phase,
            cmin=0,
            cmax=2**bits
        )
    hologram_phase = hologram_phase.astype(np.float64)
    hologram_phase *= slm_range/2**bits
    if type(illumination) == type(None):
        A = 1.
    else:
        A = illumination
    return A*np.cos(hologram_phase)+A*1j*np.sin(hologram_phase), hologram_digital
