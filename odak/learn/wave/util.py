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
    amplitude = field.abs()
    return amplitude


def set_amplitude(field, amplitude):
    """
    Definition to set amplitude of a given electric field(s) while preserving phase.

    Parameters
    ----------
    field        : torch.cfloat
                   Electric fields or an electric field.
    amplitude    : torch.float
                   New amplitude or amplitudes.

    Returns
    ----- --
    new_field    : torch.cfloat
                   New electric field(s) with modified amplitude.
    """
    phase = calculate_phase(field)
    new_field = amplitude * torch.exp(1j * phase)
    return new_field


def generate_complex_field(amplitude, phase):
    """
    Definition to generate complex field from amplitude and phase.

    Parameters
    ----------
    amplitude    : torch.float
                   Amplitude or amplitudes.
    phase        : torch.float
                   Phase or phases in radians.

    Returns
    ----- --
    field        : torch.cfloat
                   Complex field or fields.
    """
    field = amplitude * torch.exp(1j * phase)
    return field


def normalize_phase(
    phase,
    range_min=0.0,
    range_max=2 * torch.pi,
):
    """
    Definition to normalize phase values to a specified range.

    Parameters
    ----------
    phase        : torch.float
                   Phase tensor to normalize.
    range_min    : float
                   Minimum value of the output range.
    range_max    : float
                   Maximum value of the output range.

    Returns
    ----- --
    normalized   : torch.float
                   Normalized phase tensor.
    """
    phase_min = phase.min()
    phase_max = phase.max()
    phase_range = phase_max - phase_min
    if phase_range == 0:
        return torch.full_like(phase, (range_min + range_max) / 2)
    normalized = (phase - phase_min) / phase_range * (range_max - range_min) + range_min
    return normalized


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
    # Work with even dimensions to avoid mismatch
    h = tensor.shape[-2] // 2 * 2
    w = tensor.shape[-1] // 2 * 2
    t = tensor[..., :h, :w]

    # Extract 2x2 patch positions
    high_00 = t[..., 0::2, 0::2]    # even rows, even cols
    high_11 = t[..., 1::2, 1::2]    # odd rows, odd cols
    low_01 = t[..., 0::2, 1::2]     # even rows, odd cols
    low_10 = t[..., 1::2, 0::2]     # odd rows, even cols

    # Interleave along height to get [h, w//2] using stack and reshape for differentiability/batching
    component_high = torch.stack([high_00, high_11], dim=-2).reshape(*t.shape[:-2], h, w // 2)
    component_low = torch.stack([low_01, low_10], dim=-2).reshape(*t.shape[:-2], h, w // 2)

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
    # Extract rows (each has shape ..., h/2, w/2)
    h_even = component_high[..., 0::2, :]
    h_odd = component_high[..., 1::2, :]
    l_even = component_low[..., 0::2, :]
    l_odd = component_low[..., 1::2, :]

    # Interleave columns to create rows of width w
    row_even = torch.stack([h_even, l_even], dim=-1).reshape(*h_even.shape[:-1], -1)
    row_odd = torch.stack([l_odd, h_odd], dim=-1).reshape(*h_odd.shape[:-1], -1)

    # Interleave rows to create grid of height h
    shape = list(component_high.shape)
    h, w_half = shape[-2], shape[-1]
    reconstructed = torch.stack([row_even, row_odd], dim=-2).reshape(*shape[:-2], h, w_half * 2)

    return reconstructed
