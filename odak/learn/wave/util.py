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
    k = 2 * torch.pi / wavelength
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
    if deg:
        phase *= 180.0 / torch.pi
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


def normalize_phase(
    phase: torch.Tensor,
    period: float = 2.0 * torch.pi,
    multiplier: float = 1.0,
) -> torch.Tensor:
    """
    Normalizes the input phase tensor by applying a modulo operation and dividing it by the period, then multiplying it by the multiplier. The resulting normalized_phase tensor will have zero mean and unit variance.

    Args:
        phase (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing the raw phase values.
        period (float, optional): The period of the waveform to be normalized. Default is 2*pi.
        multiplier (float, optional): A scaling factor used to adjust the range of the normalized phase values. Default is 1.

    Returns:
        torch.Tensor: Normalized phase tensor of shape (batch_size, sequence_length) with zero mean and unit variance.
    """
    normalized_phase = phase % period / period * multiplier
    normalized_phase = normalized_phase - normalized_phase.mean() + multiplier / 2.0
    return normalized_phase
