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



def generate_decompose_index_map(h, w, device=None):
    """
    Generate index maps for decompose_map and compose_map functions.

    Creates separate index maps for high and low components following the 
    2x2 interleaved pattern used in double-phase decomposition.

    Parameters
    ----------
    h : int
        Height of the input tensor (must be even).
    w : int
        Width of the input tensor (must be even).
    device : torch.device or str or None
        Device to create the maps on. Defaults to cpu.

    Returns
    ----- --
    decompose_map : tuple of torch.Tensor
        A tuple (indices_high, indices_low) containing flat indices:
        - indices_high: flat indices of input elements for high component
        - indices_low: flat indices of input elements for low component
        Both have shape [h * w // 2] representing output in row-major order.
    """
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    h = h - (h % 2)
    w = w - (w % 2)
    w_comp = w // 2
    
    # Create base indices for high component
    # Even rows: [0, 2, 4, ...] repeated
    # Odd rows:  [1, 3, 5, ...] repeated
    
    # First create the column pattern [0, 2, 4, ..., 1, 3, 5, ...]
    even_cols = torch.arange(0, w, 2, device=device, dtype=torch.long)  # [0, 2, 4, ..., w-2]
    odd_cols = torch.arange(1, w, 2, device=device, dtype=torch.long)   # [1, 3, 5, ..., w-1]
    
    # Interleave: row 0 gets even_cols, row 1 gets odd_cols, row 2 gets even_cols, etc.
    # For h rows, we need to repeat the pattern h/2 times
    col_pattern_high = torch.stack([even_cols, odd_cols], dim=0).reshape(-1)  # [0, 2, 4, ..., 1, 3, 5, ...]
    col_pattern_high = col_pattern_high.repeat(h // 2 + h // 2)  # Repeat h times actually
    
    # Actually let me think again
    # Output shape is [h, w_comp] = [h, w/2]
    # Row 0: [col 0, col 2, col 4, ...]
    # Row 1: [col 1, col 3, col 5, ...]
    # Row 2: [col 0, col 2, col 4, ...]
    # ...
    # Flattened: [0,2,4,..., 1,3,5,..., 0,2,4,..., 1,3,5,..., ...]
    
    # Generate flat indices directly
    # For output position (r, c), flat output index = r * w_comp + c
    # Input flat index = r * w + col
    
    rows = torch.arange(h, device=device, dtype=torch.long).unsqueeze(1).expand(-1, w_comp)  # [h, w_comp]
    
    # Column indices alternate between even and odd
    col_indices = torch.zeros(h, w_comp, device=device, dtype=torch.long)
    col_indices[0::2, :] = even_cols  # Even rows get even columns
    col_indices[1::2, :] = odd_cols   # Odd rows get odd columns
    
    indices_high = (rows * w + col_indices).reshape(-1)  # Flatten
    
    # For low: swap even/odd
    col_indices_low = torch.zeros(h, w_comp, device=device, dtype=torch.long)
    col_indices_low[0::2, :] = odd_cols
    col_indices_low[1::2, :] = even_cols
    indices_low = (rows * w + col_indices_low).reshape(-1)
    
    return indices_high, indices_low


def decompose_map(tensor):
    """
    Decompose tensor into two components and return decomposition map for reconstruction.

    Decouples the extraction of components from their placement, returning both
    the components and the composition map needed to reconstruct the original.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape [h, w] or [b, h, w] where h and w are even.

    Returns
    ----- --
    component_high : torch.Tensor
        High component of shape [h, w//2] or [b, h, w//2].
    component_low : torch.Tensor
        Low component of shape [h, w//2] or [b, h, w//2].
    compose_map : tuple of torch.Tensor
        Destination indices (dest_high, dest_low) needed by compose_map function.
    """
    h, w = tensor.shape[-2], tensor.shape[-1]
    device = tensor.device
    
    # Generate maps
    indices_high, indices_low = generate_decompose_index_map(h, w, device)
    dest_high, dest_low = generate_compose_index_map(h, w, device)
    
    # Decompose
    out_shape = (h, w // 2)
    
    if tensor.dim() == 2:
        flat = tensor.reshape(-1)
        comp_high = flat[indices_high].reshape(out_shape)
        comp_low = flat[indices_low].reshape(out_shape)
    else:
        batch_size = tensor.shape[0]
        flat = tensor.reshape(batch_size, -1)
        comp_high = flat.gather(1, indices_high.unsqueeze(0).expand(batch_size, -1)).reshape(batch_size, *out_shape)
        comp_low = flat.gather(1, indices_low.unsqueeze(0).expand(batch_size, -1)).reshape(batch_size, *out_shape)
    
    return comp_high, comp_low, (dest_high, dest_low)


def decompose_map_standalone(tensor, decompose_map_indices):
    """
    Decompose tensor into two components using precomputed index maps.

    Legacy function for backward compatibility when you already have the 
    decomposition map from generate_decompose_index_map.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape [h, w] or [b, h, w] where h and w are even.
    decompose_map_indices : tuple of torch.Tensor
        indices_high and indices_low from generate_decompose_index_map.

    Returns
    ----- --
    component_high : torch.Tensor
        High component of shape [h, w//2] or [b, h, w//2].
    component_low : torch.Tensor
        Low component of shape [h, w//2] or [b, h, w//2].
    """
    indices_high, indices_low = decompose_map_indices
    
    h, w = tensor.shape[-2], tensor.shape[-1]
    w_comp = w // 2
    out_shape = (h, w_comp)
    
    if tensor.dim() == 2:
        flat = tensor.reshape(-1)
        comp_high = flat[indices_high].reshape(out_shape)
        comp_low = flat[indices_low].reshape(out_shape)
    else:
        batch_size = tensor.shape[0]
        flat = tensor.reshape(batch_size, -1)
        comp_high = flat.gather(1, indices_high.unsqueeze(0).expand(batch_size, -1)).reshape(batch_size, *out_shape)
        comp_low = flat.gather(1, indices_low.unsqueeze(0).expand(batch_size, -1)).reshape(batch_size, *out_shape)
    
    return comp_high, comp_low


def compose_map(component_high, component_low, compose_map):
    """
    Reconstruct tensor from two components using precomputed index maps.

    Parameters
    ----------
    component_high : torch.Tensor
        High component of shape [h, w//2] or [b, h, w//2].
    component_low : torch.Tensor
        Low component of shape [h, w//2] or [b, h, w//2].
    compose_map : tuple of torch.Tensor
        Destination indices from generate_compose_index_map.

    Returns
    ----- --
    reconstructed : torch.Tensor
        Reconstructed tensor of shape [h, w] or [b, h, w].
    """
    dest_high, dest_low = compose_map
    
    h_comp, w_comp = component_high.shape[-2:]
    h, w = h_comp, w_comp * 2
    
    if component_high.dim() == 2:
        flat = torch.zeros(h * w, dtype=component_high.dtype, device=component_high.device)
        flat[dest_high] = component_high.reshape(-1)
        flat[dest_low] = component_low.reshape(-1)
        reconstructed = flat.reshape(h, w)
    else:
        batch_size = component_high.shape[0]
        flat = torch.zeros(batch_size, h * w, dtype=component_high.dtype, device=component_high.device)
        flat.scatter_(1, dest_high.unsqueeze(0).expand(batch_size, -1), component_high.reshape(batch_size, -1))
        flat.scatter_(1, dest_low.unsqueeze(0).expand(batch_size, -1), component_low.reshape(batch_size, -1))
        reconstructed = flat.reshape(batch_size, h, w)
    
    return reconstructed


def generate_compose_index_map(h, w, device=None):
    """
    Generate compose index map for compose_map function.

    Parameters
    ----------
    h : int
        Height of component tensors.
    w : int
        Full width of output tensor (components have width w//2).
    device : torch.device or str or None
        Device to create the map on. Defaults to cpu.

    Returns
    ----- --
    compose_map : tuple of torch.Tensor
        A tuple (dest_high, dest_low) containing destination flat indices:
        - dest_high: where each element of component_high goes in flattened output
        - dest_low: where each element of component_low goes in flattened output
    """
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    h = h - (h % 2)
    w_out = w
    w_comp = w_out // 2
    
    # Create row indices [h, w_comp]
    rows = torch.arange(h, device=device, dtype=torch.long).unsqueeze(1).expand(-1, w_comp)
    
    # Column indices for high and low
    even_cols = torch.arange(0, w_out, 2, device=device, dtype=torch.long)  # [0, 2, 4, ...]
    odd_cols = torch.arange(1, w_out, 2, device=device, dtype=torch.long)   # [1, 3, 5, ...]
    
    # For high: even rows -> even cols, odd rows -> odd cols
    col_indices_high = torch.zeros(h, w_comp, device=device, dtype=torch.long)
    col_indices_high[0::2, :] = even_cols
    col_indices_high[1::2, :] = odd_cols
    
    # For low: even rows -> odd cols, odd rows -> even cols
    col_indices_low = torch.zeros(h, w_comp, device=device, dtype=torch.long)
    col_indices_low[0::2, :] = odd_cols
    col_indices_low[1::2, :] = even_cols
    
    # Convert to flat indices
    dest_high = (rows * w_out + col_indices_high).reshape(-1)
    dest_low = (rows * w_out + col_indices_low).reshape(-1)
    
    return dest_high, dest_low
    return dest_high, dest_low

