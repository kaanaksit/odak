import torch


def wavenumber(wavelength):
    """
    Definition for calculating the wavenumber of a plane wave.

    Parameters
    ----------
    wavelength   : float
                   Wavelength of a wave in mm.

    Returns
    ----- --
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
    ----- --
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
    ----- --
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
    ----- --
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
    ----- --
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


def decompose_double_phase(tensor):
    """
    Decompose tensor into two components with shape [h, w//2].

    component_high: [0,0] and [1,1] positions interleaved vertically
    component_low: [0,1] and [1,0] positions interleaved vertically

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape [m, n] or [b, m, n] where m and n are even.

    Returns
    ----- --
    component_high : torch.Tensor
        Shape is [m, n//2] or [b, m, n//2].
    component_low : torch.Tensor
        Shape is [m, n//2] or [b, m, n//2].
    """
    has_batch = False
    original_dim = tensor.dim()

    if original_dim == 3:
        has_batch = True

    # Work with 2D (h, w) tensor
    h, w = tensor.shape[-2], tensor.shape[-1]
    h = h - (h % 2)
    w = w - (w % 2)

    # Extract 2x2 patch positions
    high_00 = tensor[..., 0:h:2, 0:w:2]    # even rows, even cols
    high_11 = tensor[..., 1:h:2, 1:w:2]    # odd rows, odd cols
    low_01 = tensor[..., 0:h:2, 1:w:2]     # even rows, odd cols
    low_10 = tensor[..., 1:h:2, 0:w:2]     # odd rows, even cols

    # Interleave along height to get [h, w//2]
    component_high = torch.zeros(
        (h, w // 2), dtype=tensor.dtype, device=tensor.device
    )
    component_low = torch.zeros(
        (h, w // 2), dtype=tensor.dtype, device=tensor.device
    )

    component_high[0::2, :] = high_00
    component_high[1::2, :] = high_11
    component_low[0::2, :] = low_01
    component_low[1::2, :] = low_10

    return component_high, component_low


def compose_double_phase(component_high, component_low):
    """
    Reconstruct tensor from high and low components with shape [h, w//2].

    Even rows from component_high go to [0,0] positions, odd rows to [1,1] positions.
    Even rows from component_low go to [0,1] positions, odd rows to [1,0] positions.

    Parameters
    ----------
    component_high : torch.Tensor
        Shape is [h, w//2] or [b, h, w//2].
    component_low : torch.Tensor
        Shape is [h, w//2] or [b, h, w//2].

    Returns
    ----- --
    reconstructed : torch.Tensor
        Reconstructed tensor of shape [h, w] or [b, h, w].
    """
    has_batch = component_high.dim() == 3

    h = component_high.shape[-2]
    w_prime = component_high.shape[-1]
    h_out, w_out = h, w_prime * 2

    # Extract rows
    h_even = component_high[..., 0::2, :]    # even rows -> [0,0]
    h_odd = component_high[..., 1::2, :]     # odd rows -> [1,1]
    l_even = component_low[..., 0::2, :]     # even rows -> [0,1]
    l_odd = component_low[..., 1::2, :]      # odd rows -> [1,0]

    if has_batch:
        batch_size = component_high.shape[0]
        full = torch.zeros(
            (batch_size, h_out, w_out), dtype=component_high.dtype,
            device=component_high.device
        )

        full[:, 0::2, 0::2] = h_even
        full[:, 0::2, 1::2] = l_even
        full[:, 1::2, 0::2] = l_odd
        full[:, 1::2, 1::2] = h_odd

        return full
    else:
        reconstructed = torch.zeros(
            (h_out, w_out), dtype=component_high.dtype, device=component_high.device
        )
        reconstructed[0::2, 0::2] = h_even
        reconstructed[0::2, 1::2] = l_even
        reconstructed[1::2, 0::2] = l_odd
        reconstructed[1::2, 1::2] = h_odd
        return reconstructed
