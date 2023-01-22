import numpy as np
import torch


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


def calculate_phase(field, deg=False):
    """ 
    Definition to calculate phase of a single or multiple given electric field(s).

    Parameters
    ----------
    field        : torch.cfloat
                   Electric fields or an electric field.
    deg          : bool
                   If set True, the angles will be returned in degrees.

    Returns
    -------
    phase        : torch.float
                   Phase or phases of electric field(s) in radians.
    """
    phase = field.imag.atan2(field.real)
    if deg == True:
        phase *= 180./3.14
    return phase


def calculate_amplitude(field):
    """ 
    Definition to calculate amplitude of a single or multiple given electric field(s).

    Parameters
    ----------
    field        : torch.cfloat
                   Electric fields or an electric field.

    Returns
    -------
    amplitude    : torch.float
                   Amplitude or amplitudes of electric field(s).
    """
    amplitude = torch.abs(field)
    return amplitude


def set_amplitude(field, amplitude):
    """
    Definition to keep phase as is and change the amplitude of a given field.

    Parameters
    ----------
    field        : torch.cfloat
                   Complex field.
    amplitude    : torch.cfloat or torch.float
                   Amplitudes.

    Returns
    -------
    new_field    : torch.cfloat
                   Complex field.
    """
    amplitude = calculate_amplitude(amplitude)
    phase = calculate_phase(field)
    new_field = amplitude * torch.cos(phase) + 1j * amplitude * torch.sin(phase)
    return new_field


def generate_complex_field(amplitude, phase):
    """
    Definition to generate a complex field with a given amplitude and phase.

    Parameters
    ----------
    amplitude         : ndarray or float
                        Amplitude of the field.
    phase             : ndarray or float
                        Phase of the field.

    Returns
    -------
    field             : ndarray
                        Complex field.
    """
    if isinstance(type(phase), type(1.)):
        phase = torch.tensor([phase], requires_grad = True)
    if isinstance(type(amplitude), type(1.)):
        amplitude = torch.tensor([amplitude], requires_grad =True)
    field = amplitude * torch.cos(phase) + 1j * amplitude * torch.sin(phase)
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


def produce_phase_only_slm_pattern(hologram, slm_range, bits=8, default_range=6.28, illumination=None):
    """
    Definition for producing a pattern for a phase only Spatial Light Modulator (SLM) using a given field.

    Parameters
    ----------
    hologram           : torch.cfloat
                         Input holographic field.
    slm_range          : float
                         Range of the phase only SLM in radians for a working wavelength (i.e. two pi). See odak.wave.adjust_phase_only_slm_range() for more.
    bits               : int
                         Quantization bits.
    default_range      : float
                         Default range of phase only SLM.
    illumination       : torch.tensor
                         Spatial illumination distribution.

    Returns
    -------
    pattern            : torch.cfloat
                         Adjusted phase only pattern.
    hologram_digital   : np.int
                         Digital representation of the hologram.
    """
#    hologram_phase    = calculate_phase(hologram) % default_range
    hologram_phase = calculate_phase(hologram)
    hologram_phase = hologram_phase % slm_range
    hologram_phase /= slm_range
    hologram_phase *= 2**bits
    hologram_digital = hologram_phase.detach().clone()
    hologram_phase = torch.ceil(hologram_phase)
    hologram_phase *= slm_range/2**bits
    if type(illumination) == type(None):
        A = torch.tensor([1.]).to(hologram_phase.device)
    else:
        A = illumination
    return A*torch.cos(hologram_phase)+A*1j*torch.sin(hologram_phase), hologram_digital
