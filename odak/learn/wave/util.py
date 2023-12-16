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
    k = 2 * np.pi / wavelength
    return k


def calculate_phase(field, deg = False):
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
    if deg:
        phase *= 180. / np.pi
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
    amplitude         : torch.tensor
                        Amplitude of the field.
                        The expected size is [m x n] or [1 x m x n].
    phase             : torch.tensor
                        Phase of the field.
                        The expected size is [m x n] or [1 x m x n].

    Returns
    -------
    field             : ndarray
                        Complex field.
                        Depending on the input, the expected size is [m x n] or [1 x m x n].
    """
    field = amplitude * torch.cos(phase) + 1j * amplitude * torch.sin(phase)
    return field
