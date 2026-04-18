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
    Decompose a tensor into high and low components by extracting values from 2x2 patches.

    For each 2x2 patch, [0,0] and [1,1] values go to component_high,
    while [0,1] and [1,0] values go to component_low.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape [m, n] or [b, m, n] where m and n are even.

    Returns
    ----- --
    component_high : torch.Tensor
        Tensor containing values from [0,0] and [1,1] positions of each 2x2 patch.
        Shape is [m//2, n//2, 2] or [b, m//2, n//2, 2].
        Last dimension index 0 holds [0,0], index 1 holds [1,1].
    component_low : torch.Tensor
        Tensor containing values from [0,1] and [1,0] positions of each 2x2 patch.
        Shape is [m//2, n//2, 2] or [b, m//2, n//2, 2].
        Last dimension index 0 holds [0,1], index 1 holds [1,0].
    """
    has_batch = False
    original_dim = tensor.dim()

    if original_dim == 3:
        has_batch = True

    # Work with 2D (m, n) tensor
    h, w = tensor.shape[-2], tensor.shape[-1]
    h = h - (h % 2)
    w = w - (w % 2)

    # [0,0] positions: even rows, even cols
    high_00 = tensor[..., 0:h:2, 0:w:2]
    # [1,1] positions: odd rows, odd cols
    high_11 = tensor[..., 1:h:2, 1:w:2]
    # [0,1] positions: even rows, odd cols
    low_01 = tensor[..., 0:h:2, 1:w:2]
    # [1,0] positions: odd rows, even cols
    low_10 = tensor[..., 1:h:2, 0:w:2]

    component_high = torch.stack([high_00, high_11], dim=-1)   # [m//2, n//2, 2]
    component_low = torch.stack([low_01, low_10], dim=-1)      # [m//2, n//2, 2]

    if has_batch:
        component_high = component_high.unsqueeze(0)  # [1, m//2, n//2, 2]
        component_low = component_low.unsqueeze(0)    # [1, m//2, n//2, 2]

    return component_high, component_low


def compose_double_phase(component_high, component_low):
    """
    Reconstruct a tensor from high and low components generated by decompose_double_phase.

    For each 2x2 patch, component_high provides [0,0] and [1,1] values,
    while component_low provides [0,1] and [1,0] values.

    Parameters
    ----------
    component_high : torch.Tensor
        High-component tensor with shape [m', n', 2] or [b, m', n', 2].
        Last dimension index 0 holds [0,0] positions, index 1 holds [1,1] positions.
    component_low : torch.Tensor
        Low-component tensor with shape [m', n', 2] or [b, m', n', 2].
        Last dimension index 0 holds [0,1] positions, index 1 holds [1,0] positions.

    Returns
    ----- --
    reconstructed : torch.Tensor
        Reconstructed tensor of shape [2*m', 2*n'] or [b, 2*m', 2*n'].
    """
    # Infer output dimensions from component shape
    has_batch = component_high.dim() == 4

    # Extract half dimensions
    m_prime = component_high.shape[-3]
    n_prime = component_high.shape[-2]
    m, n = m_prime * 2, n_prime * 2

    # Extract individual components from last dimension
    h_00 = component_high[..., 0]   # [m', n'] or [b, m', n']
    h_11 = component_high[..., 1]   # [m', n'] or [b, m', n']
    l_01 = component_low[..., 0]    # [m', n'] or [b, m', n']
    l_10 = component_low[..., 1]    # [m', n'] or [b, m', n']

    if has_batch:
        # Batch operation
        batch_size = component_high.shape[0]
        full = torch.zeros(
            (batch_size, m, n), dtype=component_high.dtype,
            device=component_high.device
        )

        full[:, 0::2, 0::2] = h_00
        full[:, 0::2, 1::2] = l_01
        full[:, 1::2, 0::2] = l_10
        full[:, 1::2, 1::2] = h_11

        return full
    else:
        # No batch
        reconstructed = torch.zeros(
            (m, n), dtype=component_high.dtype, device=component_high.device
        )
        reconstructed[0::2, 0::2] = h_00  # [0,0]
        reconstructed[0::2, 1::2] = l_01  # [0,1]
        reconstructed[1::2, 0::2] = l_10  # [1,0]
        reconstructed[1::2, 1::2] = h_11  # [1,1]
        return reconstructed
