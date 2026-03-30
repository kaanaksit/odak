"""
Content-Aware Contrast Ratio Measure (CWMC) and Image Contrast Metrics

This module provides:
- content_aware_contrast_ratio: Full CWMC implementation using ISODATA clustering
- weber_contrast: Weber contrast ratio for specified image regions
- michelson_contrast: Michelson contrast ratio for specified image regions
"""

import torch
from typing import Dict, Tuple, Optional, Union
import torch.nn.functional as F


def _validate_contrast_inputs(
    image: torch.Tensor,
    roi_high: list,
    roi_low: list
) -> torch.Tensor:
    """
    Validate inputs for contrast ratio functions.
    
    Parameters
    ----------
    image      : torch.Tensor
                 Input image tensor
    roi_high   : list
                 High ROI coordinates [m_start, m_end, n_start, n_end]
    roi_low    : list
                 Low ROI coordinates [m_start, m_end, n_start, n_end]
    
    Returns
    -------
    image_normalized : torch.Tensor
                      Normalized and reshaped image tensor [B, C, H, W]
    
    Raises
    ------
    TypeError
        If image is not a torch.Tensor
    ValueError
        If ROI coordinates are invalid or out of bounds
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be torch.Tensor, got {type(image)}")
    
    # Validate ROI coordinates
    if len(roi_high) != 4 or len(roi_low) != 4:
        raise ValueError("roi_high and roi_low must have 4 elements [m_start, m_end, n_start, n_end]")
    
    if not all(isinstance(x, (int, torch.Tensor)) for x in roi_high + roi_low):
        raise ValueError("ROI coordinates must be integers or 0-dim tensors")
    
    roi_high = [int(x) for x in roi_high]
    roi_low = [int(x) for x in roi_low]
    
    # Validate ROI bounds
    if len(image.shape) == 2:
        H, W = image.shape
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        C, H, W = image.shape
        image = image.unsqueeze(0)
    elif len(image.shape) == 4:
        B, C, H, W = image.shape
    else:
        raise ValueError(f"image must be 2D, 3D, or 4D tensor, got {len(image.shape)}D")
    
    # Validate ROI coordinates are within bounds
    for name, roi in [("roi_high", roi_high), ("roi_low", roi_low)]:
        if roi[0] < 0 or roi[1] > H or roi[2] < 0 or roi[3] > W:
            raise ValueError(
                f"{name} coordinates [{roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}] "
                f"out of bounds for image size {H}x{W}"
            )
        if roi[0] >= roi[1] or roi[2] >= roi[3]:
            raise ValueError(
                f"{name} has invalid coordinates: start must be < end "
                f"(got [{roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}])"
            )
    
    return image


def weber_contrast(
    image: torch.Tensor,
    roi_high: Union[list, Tuple[int, int, int, int]],
    roi_low: Union[list, Tuple[int, int, int, int]]
) -> torch.Tensor:
    """
    Calculate Weber contrast ratio for specified image regions.
    
    Weber contrast is defined as: C_W = (I_max - I_min) / I_min
    where I_max and I_min are the mean intensities of the high and low regions,
    respectively. This metric is particularly useful for images with uniform
    background and localized features.
    
    Parameters
    ----------
    image         : torch.Tensor
                    Input image with shape [H, W], [C, H, W], or [B, C, H, W].
                    Values should be in a meaningful intensity range.
    roi_high      : list or tuple
                    Corner locations of the high intensity region in the format
                    [m_start, m_end, n_start, n_end] where m is the row (spatial)
                    dimension and n is the column (spatial) dimension.
    roi_low       : list or tuple
                    Corner locations of the low intensity region in the same format
                    as roi_high.
    
    Returns
    -------
    contrast      : torch.Tensor
                    Weber contrast value(s). Shape is [1] for grayscale or 
                    [C] for multi-channel images, squeezed to scalar if single channel.
    
    Raises
    ------
    TypeError
        If image is not a torch.Tensor
    ValueError
        If ROI coordinates are invalid, out of bounds, or if I_min is zero/negative
    
    Example
    -------
    >>> import torch
    >>> from odak.learn.perception import weber_contrast
    >>> 
    >>> # Create test image
    >>> image = torch.rand(1, 3, 64, 64)
    >>> 
    >>> # Define regions (row_start, row_end, col_start, col_end)
    >>> roi_high = [0, 32, 0, 32]  # Top-left quadrant
    >>> roi_low = [32, 64, 32, 64]  # Bottom-right quadrant
    >>> 
    >>> # Compute Weber contrast
    >>> contrast = weber_contrast(image, roi_high, roi_low)
    >>> print(f"Weber contrast: {contrast.item():.4f}")
    """
    image = _validate_contrast_inputs(image, roi_high, roi_low)
    
    # Extract regions
    region_low = image[:, :, roi_low[0]:roi_low[1], roi_low[2]:roi_low[3]]
    region_high = image[:, :, roi_high[0]:roi_high[1], roi_high[2]:roi_high[3]]
    
    # Compute mean intensities
    high = torch.mean(region_high, dim=(2, 3))
    low = torch.mean(region_low, dim=(2, 3))
    
    # Avoid division by zero
    if torch.any(low <= 0):
        raise ValueError(
            f"Low region mean intensity must be positive (got {low[low <= 0]})"
        )
    
    # Weber contrast: C_W = (I_high - I_low) / I_low
    contrast = (high - low) / low
    
    return contrast.squeeze(0)


def michelson_contrast(
    image: torch.Tensor,
    roi_high: Union[list, Tuple[int, int, int, int]],
    roi_low: Union[list, Tuple[int, int, int, int]]
) -> torch.Tensor:
    """
    Calculate Michelson contrast ratio for specified image regions.
    
    Michelson contrast is defined as: C_M = (I_max - I_min) / (I_max + I_min)
    where I_max and I_min are the mean intensities of the high and low regions.
    This metric produces values in the range [0, 1] and is commonly used for
    periodic patterns and sinusoidal gratings.
    
    Parameters
    ----------
    image         : torch.Tensor
                    Input image with shape [H, W], [C, H, W], or [B, C, H, W].
                    Values should be in a meaningful intensity range.
    roi_high      : list or tuple
                    Corner locations of the high intensity region in the format
                    [m_start, m_end, n_start, n_end] where m is the row (spatial)
                    dimension and n is the column (spatial) dimension.
    roi_low       : list or tuple
                    Corner locations of the low intensity region in the same format
                    as roi_high.
    
    Returns
    -------
    contrast      : torch.Tensor
                    Michelson contrast value(s). Values are in range [0, 1].
                    Shape is [1] for grayscale or [C] for multi-channel images,
                    squeezed to scalar if single channel.
    
    Raises
    ------
    TypeError
        If image is not a torch.Tensor
    ValueError
        If ROI coordinates are invalid, out of bounds, or if sum of means is zero
    
    Example
    -------
    >>> import torch
    >>> from odak.learn.perception import michelson_contrast
    >>> 
    >>> # Create test image
    >>> image = torch.rand(1, 3, 64, 64)
    >>> 
    >>> # Define regions (row_start, row_end, col_start, col_end)
    >>> roi_high = [0, 32, 0, 32]  # Top-left quadrant
    >>> roi_low = [32, 64, 32, 64]  # Bottom-right quadrant
    >>> 
    >>> # Compute Michelson contrast
    >>> contrast = michelson_contrast(image, roi_high, roi_low)
    >>> print(f"Michelson contrast: {contrast.item():.4f}")
    """
    image = _validate_contrast_inputs(image, roi_high, roi_low)
    
    # Extract regions
    region_low = image[:, :, roi_low[0]:roi_low[1], roi_low[2]:roi_low[3]]
    region_high = image[:, :, roi_high[0]:roi_high[1], roi_high[2]:roi_high[3]]
    
    # Compute mean intensities
    high = torch.mean(region_high, dim=(2, 3))
    low = torch.mean(region_low, dim=(2, 3))
    
    # Avoid division by zero
    denom = high + low
    if torch.any(denom <= 0):
        raise ValueError(
            f"Sum of high and low means must be positive (got {denom[denom <= 0]})"
        )
    
    # Michelson contrast: C_M = (I_high - I_low) / (I_high + I_low)
    contrast = (high - low) / denom
    
    return contrast.squeeze(0)


def _extract_patches_torch(
    image: torch.Tensor,
    window_size: int = 15,
    step: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract sliding window patches from a tensor using unfold.
    
    Parameters
    ----------
    image      : torch.Tensor
                 Input tensor of shape [H, W] or [B, H, W]
    window_size: int
                 Odd patch size (default: 15)
    step       : int
                 Sliding window step size (default: 3)
    
    Returns
    -------
    patches    : torch.Tensor
                 Extracted patches of shape [num_patches, window_size*window_size]
    row_idx    : torch.Tensor
                 Row indices of patch centers
    col_idx    : torch.Tensor
                 Column indices of patch centers
    """
    if image.ndim == 2:
        image = image.unsqueeze(0)  # Add batch dimension
    
    B, H, W = image.shape
    half = window_size // 2
    
    # Extract patches using unfold
    patches = F.unfold(
        image.unsqueeze(1),  # [B, 1, H, W]
        kernel_size=window_size,
        stride=step,
        padding=0
    )  # [B, window_size*window_size, num_patches]
    
    # Reshape to [B*num_patches, window_size*window_size]
    B, _, num_patches = patches.shape
    patches = patches.permute(0, 2, 1).reshape(-1, window_size * window_size)
    
    # Calculate patch center coordinates
    Hp = (H - window_size) // step + 1
    Wp = (W - window_size) // step + 1
    
    row_idx = torch.arange(Hp) * step + half
    col_idx = torch.arange(Wp) * step + half
    
    return patches, row_idx, col_idx


def _isodata_threshold_torch(
    patches: torch.Tensor,
    eps: float = 1e-6,
    max_iter: int = 100
) -> torch.Tensor:
    """
    ISODATA clustering to find optimal threshold per patch.
    
    Initializes with patch mean and iteratively refines threshold
    by partitioning pixels into foreground and background.
    
    Parameters
    ----------
    patches  : torch.Tensor
               Flattened patches of shape [num_patches, window_size*window_size]
    eps      : float
               Convergence tolerance (default: 1e-6)
    max_iter : int
               Maximum iterations (default: 100)
    
    Returns
    -------
    threshold: torch.Tensor
               Optimal thresholds of shape [num_patches]
    """
    # Initialize threshold with patch mean
    T = patches.mean(dim=1, keepdim=False)
    
    for _ in range(max_iter):
        # Partition into foreground (below) and background (above)
        below = patches < T.unsqueeze(1)
        above = ~below
        
        # Count pixels in each partition
        n_f = below.sum(dim=1, dtype=torch.float32)
        n_b = above.sum(dim=1, dtype=torch.float32)
        
        # Compute mean of each partition (avoid division by zero)
        f = torch.where(
            n_f > 0,
            (patches * below.float()).sum(dim=1) / n_f,
            T
        )
        
        b = torch.where(
            n_b > 0,
            (patches * above.float()).sum(dim=1) / n_b,
            T
        )
        
        # Update threshold (new threshold is mean of f and b)
        T_new = (f + b) * 0.5
        
        # Check convergence
        diff = torch.abs(T_new - T)
        if diff.max() <= eps:
            T = T_new
            break
        
        T = T_new
    
    return T


def content_aware_contrast_ratio(
    image: torch.Tensor,
    window_size: int = 15,
    step: int = 3,
    pooling_percentile: float = 75.0,
    eps: float = 1e-6,
    max_iter: int = 100,
    lightness_channel: int = 0
) -> Dict:
    """
    Content-Aware Contrast Ratio Measure (CWMC) using Weber contrast.
    
    This implementation faithfully reproduces the Ortiz-Jaramillo et al. (2018)
    methodology using PyTorch for GPU acceleration and differentiability.
    
    Algorithm Overview:
    1. Extract sliding window patches with specified window size and step
    2. For each patch, apply ISODATA clustering to find optimal threshold
    3. Partition pixels into foreground (below threshold) and background (above)
    4. Compute Weber contrast: c = (b - f) / b = 1 - f/b
    5. Apply percentile pooling to keep top contrast values
    6. Return global harmonic mean of filtered contrasts
    
    Parameters
    ----------
    image               : torch.Tensor
                          Input image tensor of shape [H, W] or [B, H, W] or [B, C, H, W]
                          - For [B, C, H, W]: uses channel specified by lightness_channel
                          - Assumes intensity/lightness values in [0, 1] range
                          - For LAB images (L* in [0, 100]), preprocess by dividing by 100
    window_size         : int
                          Odd patch size for sliding window (default: 15)
    step                : int
                          Sliding window step size (default: 3)
                          Use 1 for dense per-pixel contrast map
    pooling_percentile  : float
                          Percentile threshold (default: 75.0)
                          Keeps top (100 - percentile)% of contrast values
                          Example: 75.0 keeps top 25% of values
    eps                 : float
                          ISODATA convergence tolerance (default: 1e-6)
    max_iter            : int
                          Maximum ISODATA iterations (default: 100)
    lightness_channel   : int
                          Channel index for lightness if image has multiple channels (default: 0)
    
    Returns
    -------
    result : dict
             Dictionary containing:
             - 'contrast_map': Full image contrast map [H, W] with values at window centers
             - 'threshold_map': ISODATA thresholds at window centers [H, W]
             - 'foreground_mean_map': Mean foreground intensity per patch [H, W]
             - 'background_mean_map': Mean background intensity per patch [H, W]
             - 'pooled_harmonic_mean': Global CWMC score (harmonic mean of filtered contrasts)
             - 'max_contrast_ratio': Maximum local Weber contrast
             - 'window_size': Used patch size
             - 'step': Used sliding step
             - 'n_patches': Number of patches evaluated
    
    Raises
    ------
    ValueError
        If window_size is not odd and >= 3, or if step < 1,
        or if pooling_percentile not in (0, 100]
    
    Example
    -------
    >>> import torch
    >>> from odak.learn.perception import content_aware_contrast_ratio
    >>> 
    >>> # Create sample image
    >>> image = torch.rand(1, 1, 256, 256)  # [B, C, H, W]
    >>> 
    >>> # Compute CWMC
    >>> result = content_aware_contrast_ratio(image, window_size=15, step=3)
    >>> print(f"CWMC score: {result['pooled_harmonic_mean']:.4f}")
    >>> print(f"Max contrast: {result['max_contrast_ratio']:.4f}")
    """
    # Validation
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if not (0.0 < pooling_percentile <= 100.0):
        raise ValueError(
            f"pooling_percentile must be in (0, 100], got {pooling_percentile}"
        )
    
    # Handle input dimensions
    if image.ndim == 2:
        # [H, W]
        L = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        squeeze_batch = True
    elif image.ndim == 3:
        # Could be [B, H, W] or [C, H, W]
        if image.shape[0] > 100:  # Likely not a batch dimension
            L = image.unsqueeze(0)  # [1, C, H, W]
            squeeze_batch = True
        else:
            L = image.unsqueeze(1)  # [B, 1, H, W]
            squeeze_batch = False
    elif image.ndim == 4:
        # [B, C, H, W]
        L = image
        squeeze_batch = False
    else:
        raise ValueError(f"image must have shape [H, W], [B, H, W], or [B, C, H, W], got {image.shape}")
    
    # Extract lightness channel if multi-channel
    if L.ndim == 4 and L.shape[1] > 1:
        L = L[:, lightness_channel:lightness_channel+1, :, :]
    
    B, C, H, W = L.shape
    L = L.squeeze(1)  # [B, H, W]
    
    # Ensure values are in valid range
    L = torch.clamp(L, 0.0, 1.0)
    
    # Extract patches using sliding window
    patches, row_idx, col_idx = _extract_patches_torch(L, window_size, step)
    
    # Apply ISODATA thresholding
    thresholds = _isodata_threshold_torch(patches, eps, max_iter)
    
    # Final partition based on converged thresholds
    below = patches < thresholds.unsqueeze(1)
    above = ~below
    
    n_f = below.sum(dim=1, dtype=torch.float32)
    n_b = above.sum(dim=1, dtype=torch.float32)
    
    f = torch.where(
        n_f > 0,
        (patches * below.float()).sum(dim=1) / n_f,
        thresholds
    )
    
    b = torch.where(
        n_b > 0,
        (patches * above.float()).sum(dim=1) / n_b,
        thresholds
    )
    
    # Compute Weber contrast: c = 1 - f/b = (b - f) / b
    safe_b = torch.clamp(b, min=1e-12)
    cr_vals = torch.where(b > 1e-12, 1.0 - (f / safe_b), torch.zeros_like(f))
    cr_vals = torch.clamp(cr_vals, min=0.0)
    
    # Compute number of patches
    Hp = (H - window_size) // step + 1
    Wp = (W - window_size) // step + 1
    num_patches = Hp * Wp
    
    # Reshape results to grid
    if B == 1:
        contrast_grid = cr_vals.reshape(Hp, Wp)
        threshold_grid = thresholds.reshape(Hp, Wp)
        f_grid = f.reshape(Hp, Wp)
        b_grid = b.reshape(Hp, Wp)
        
        # Create full-size maps
        contrast_map = torch.zeros(H, W, dtype=L.dtype, device=L.device)
        threshold_map = torch.zeros(H, W, dtype=L.dtype, device=L.device)
        foreground_mean_map = torch.zeros(H, W, dtype=L.dtype, device=L.device)
        background_mean_map = torch.zeros(H, W, dtype=L.dtype, device=L.device)
        
        # Place values at window centers
        for i, r in enumerate(row_idx):
            for j, c in enumerate(col_idx):
                contrast_map[r, c] = contrast_grid[i, j]
                threshold_map[r, c] = threshold_grid[i, j]
                foreground_mean_map[r, c] = f_grid[i, j]
                background_mean_map[r, c] = b_grid[i, j]
        
        pooled_harmonic_mean = torch.tensor(0.0, dtype=L.dtype, device=L.device)
        max_contrast_ratio = torch.max(cr_vals)
        
        # Percentile pooling + harmonic mean
        local_vals = cr_vals[torch.isfinite(cr_vals)]
        if local_vals.numel() > 0:
            threshold_val = torch.quantile(local_vals, pooling_percentile / 100.0)
            pooled_vals = local_vals[local_vals >= threshold_val]
            
            if pooled_vals.numel() > 0 and torch.all(pooled_vals > 0):
                harmonic_mean = pooled_vals.numel() / torch.sum(1.0 / pooled_vals)
                pooled_harmonic_mean = harmonic_mean
        
        result = {
            'contrast_map': contrast_map,
            'threshold_map': threshold_map,
            'foreground_mean_map': foreground_mean_map,
            'background_mean_map': background_mean_map,
            'pooled_harmonic_mean': pooled_harmonic_mean.item(),
            'max_contrast_ratio': max_contrast_ratio.item(),
            'window_size': window_size,
            'step': step,
            'n_patches': num_patches,
        }
    else:
        # Batch processing
        result = []
        for b_idx in range(B):
            batch_cr = cr_vals[b_idx * num_patches: (b_idx + 1) * num_patches]
            batch_thresh = thresholds[b_idx * num_patches: (b_idx + 1) * num_patches]
            batch_f = f[b_idx * num_patches: (b_idx + 1) * num_patches]
            batch_b = b[b_idx * num_patches: (b_idx + 1) * num_patches]
            
            contrast_grid = batch_cr.reshape(Hp, Wp)
            threshold_grid = batch_thresh.reshape(Hp, Wp)
            f_grid = batch_f.reshape(Hp, Wp)
            b_grid = batch_b.reshape(Hp, Wp)
            
            contrast_map = torch.zeros(H, W, dtype=L.dtype, device=L.device)
            threshold_map = torch.zeros(H, W, dtype=L.dtype, device=L.device)
            foreground_mean_map = torch.zeros(H, W, dtype=L.dtype, device=L.device)
            background_mean_map = torch.zeros(H, W, dtype=L.dtype, device=L.device)
            
            for i, r in enumerate(row_idx):
                for j, c in enumerate(col_idx):
                    contrast_map[r, c] = contrast_grid[i, j]
                    threshold_map[r, c] = threshold_grid[i, j]
                    foreground_mean_map[r, c] = f_grid[i, j]
                    background_mean_map[r, c] = b_grid[i, j]
            
            pooled_harmonic_mean = torch.tensor(0.0, dtype=L.dtype, device=L.device)
            max_contrast_ratio = torch.max(batch_cr)
            
            # Percentile pooling
            local_vals = batch_cr[torch.isfinite(batch_cr)]
            if local_vals.numel() > 0:
                threshold_val = torch.quantile(local_vals, pooling_percentile / 100.0)
                pooled_vals = local_vals[local_vals >= threshold_val]
                
                if pooled_vals.numel() > 0 and torch.all(pooled_vals > 0):
                    harmonic_mean = pooled_vals.numel() / torch.sum(1.0 / pooled_vals)
                    pooled_harmonic_mean = harmonic_mean
            
            result.append({
                'contrast_map': contrast_map,
                'threshold_map': threshold_map,
                'foreground_mean_map': foreground_mean_map,
                'background_mean_map': background_mean_map,
                'pooled_harmonic_mean': pooled_harmonic_mean.item(),
                'max_contrast_ratio': max_contrast_ratio.item(),
                'window_size': window_size,
                'step': step,
                'n_patches': num_patches,
            })
        
        return result
    
    return result
