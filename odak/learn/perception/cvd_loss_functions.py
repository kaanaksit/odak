"""
CVD-Oriented Loss Functions for Personalized Image Generation

Implements the CVD-friendly loss functions from Section 4.2 of:
"Personalized Image Generation for Color Vision Deficiency Population" (ICCV 2023)

Authors: Shuyi Jiang, Daochang Liu, Dingquan Li, Chang Xu

This module provides:
- Local Contrast Loss (L_LC): Preserves local contrast after CVD simulation
- Color Information Loss (L_CI): Maintains color information after simulation
- Combined CVD Loss (L_CVD): Wrapper combining both losses
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from ...learn.tools.matrix import blur_gaussian


def extract_local_patches(
    image: torch.Tensor,
    patch_size: int = 4,
    stride: int = 2
) -> Tuple[torch.Tensor, list]:
    """
    Extract non-overlapping local patches from an image using unfolding.
    
    Parameters
    ----------
    image         : torch.Tensor
                    Input tensor of shape [B, C, H, W]
    patch_size    : int
                    Size of local patches (default: 4)
    stride        : int
                    Stride between patches (default: 2)
    
    Returns
    -------
    patches       : torch.Tensor
                    Tensor of shape [B, C*patch_size*patch_size, N] where N is number of patches
    coordinates   : list
                    List of [(x1, y1), (x2, y2), ...] patch positions
    
    Raises
    ------
    TypeError
        If image is not a torch.Tensor
    ValueError
        If image is not 4D or has incompatible dimensions
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be torch.Tensor, got {type(image)}")
    
    if image.dim() != 4:
        raise ValueError(f"image must be 4D [B, C, H, W], got {image.dim()}D")
    
    # Use unfold to extract patches: [B, C, H/new, W/new, patch_size, patch_size]
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    
    # Reshape to [B, C*patch_size*patch_size, N]
    B, C, Nh, Nw, ps_h, ps_w = patches.shape
    patches = patches.reshape(B, C * ps_h * ps_w, Nh * Nw)
    
    # Generate coordinates
    coordinates = []
    for i in range(Nh):
        for j in range(Nw):
            coordinates.append((j * stride, i * stride))
    
    return patches, coordinates


def compute_local_contrast_similarity(
    patch_x: torch.Tensor,
    patch_y: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute contrast similarity between two corresponding patches.
    
    Based on SSIM contrast term: c(x,y) = (2*σ_x*σ_y + ε2) / (σ_x² + σ_y² + ε2)
    
    Parameters
    ----------
    patch_x    : torch.Tensor
                 First patch of shape [C*patch_size*patch_size, N]
    patch_y    : torch.Tensor
                 Second patch of shape [C*patch_size*patch_size, N]
    epsilon    : float
                 Small constant to avoid division-insiability
    
    Returns
    -------
    similarity : torch.Tensor
                 Similarity score [1, N] for each patch
    
    Raises
    ------
    TypeError
        If inputs are not torch.Tensor
    ValueError
        If patch shapes are incompatible
    """
    if not isinstance(patch_x, torch.Tensor):
        raise TypeError(f"patch_x must be torch.Tensor, got {type(patch_x)}")
    if not isinstance(patch_y, torch.Tensor):
        raise TypeError(f"patch_y must be torch.Tensor, got {type(patch_y)}")
    
    if patch_x.shape != patch_y.shape:
        raise ValueError(
            f"patch shapes must match, "
            f"got {patch_x.shape} vs {patch_y.shape}"
        )
    
    # Compute mean and std for each patch across channels
    # patch shape: [C*ps*ps, N], we want std across the first dimension
    std_x = patch_x.std(dim=0, unbiased=False) + epsilon
    std_y = patch_y.std(dim=0, unbiased=False) + epsilon
    
    # SSIM contrast term
    numerator = 2 * std_x * std_y + epsilon
    denominator = std_x.pow(2) + std_y.pow(2) + epsilon
    
    similarity = numerator / denominator
    
    return similarity


def local_contrast_loss(
    image: torch.Tensor,
    simulated_image: torch.Tensor,
    patch_size: int = 4,
    stride: int = 2,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Local Contrast Loss (L_LC) - Equation 5 from the paper.
    
    Measures the decay of local contrast between an image and its CVD simulation.
    Lower loss indicates better preservation of local contrast boundaries.
    
    L_LC(I, δs) = (1/|N|) * Σ(1 - c(x, y))
    where c(x,y) is the contrast similarity between corresponding patches
    
    Parameters
    ----------
    image             : torch.Tensor
                        Original generated image [B, C, H, W]
    simulated_image   : torch.Tensor
                        CVD-simulated image [B, C, H, W]
    patch_size        : int
                        Size of local patches for contrast evaluation (default: 4)
    stride            : int
                        Stride between patches (default: 2)
    epsilon           : float
                        Small constant for numerical stability (default: 1e-8)
    
    Returns
    -------
    loss              : torch.Tensor
                        Scalar loss value (mean contrast decay across all patches)
    
    Raises
    ------
    TypeError
        If inputs are not torch.Tensor
    ValueError
        If image shapes are incompatible or dimensions too small
    
    Example
    -------
    >>> image = torch.rand(2, 3, 64, 64)
    >>> simulated = image * 0.9  # Simulated CVD version
    >>> loss = local_contrast_loss(image, simulated)
    >>> print(f"Local contrast loss: {loss.item():.4f}")
    """
    # Validate inputs
    _validate_image_pair(image, simulated_image)
    
    # Extract local patches from both images
    patches_original, _ = extract_local_patches(image, patch_size, stride)
    patches_simulated, _ = extract_local_patches(simulated_image, patch_size, stride)
    
    # Get shape: [B, C*patch_size*patch_size, N]
    B, _, N = patches_original.shape
    
    # Compute contrast similarity for each patch
    similarities = []
    for b in range(B):
        sim = compute_local_contrast_similarity(
            patches_original[b],
            patches_simulated[b],
            epsilon
        )
        similarities.append(sim)
    
    # Stack similarities: [B, N]
    similarities = torch.stack(similarities)
    
    # Compute loss: 1 - similarity (Eq. 5)
    contrast_decay = 1 - similarities
    
    # Mean across all patches and batch
    loss = contrast_decay.mean()
    
    return loss


def color_information_loss(
    image: torch.Tensor,
    simulated_image: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Color Information Loss (L_CI) - Equation 6 from the paper.
    
    Measures the L1 distance between primary colors of original and simulated images.
    Gaussian blur is applied to avoid excessive detail and focus on main colors.
    
    L_CI(I, δs) = ||Φ(I) - Φ(Sim(I, δs))||_1
    
    Parameters
    ----------
    image           : torch.Tensor
                      Original generated image [B, C, H, W]
    simulated_image : torch.Tensor
                      CVD-simulated image [B, C, H, W]
    kernel_size     : int
                      Size of Gaussian blur kernel (default: 5)
    sigma           : float
                      Standard deviation for Gaussian blur (default: 1.0)
    
    Returns
    -------
    loss            : torch.Tensor
                      Scalar L1 loss value
    
    Raises
    ------
    TypeError
        If inputs are not torch.Tensor
    ValueError
        If image shapes are incompatible or dimensions too small
    
    Example
    -------
    >>> image = torch.rand(2, 3, 64, 64)
    >>> simulated = image * 0.9
    >>> loss = color_information_loss(image, simulated)
    >>> print(f"Color information loss: {loss.item():.4f}")
    """
    # Validate inputs
    _validate_image_pair(image, simulated_image)
    
    # Apply Gaussian blur to both images using odak.learn.tools.matrix.blur_gaussian
    kernel_length = [kernel_size, kernel_size]
    nsigma = [sigma, sigma]
    
    # Apply blur to each channel separately
    def apply_blur(img):
        B, C, H, W = img.shape
        blurred = torch.zeros_like(img)
        for b in range(B):
            for c in range(C):
                channel = img[b:b+1, c:c+1, :, :]
                blurred_channel = blur_gaussian(
                    channel,
                    kernel_length=kernel_length,
                    nsigma=nsigma,
                    padding='same'
                )
                if blurred_channel.dim() == 2:
                    blurred_channel = blurred_channel.unsqueeze(0).unsqueeze(0)
                blurred[b, c, :, :] = blurred_channel[0, 0, :, :]
        return blurred
    
    blurred_original = apply_blur(image)
    blurred_simulated = apply_blur(simulated_image)
    
    # Compute L1 norm (Eq. 6)
    loss = F.l1_loss(blurred_original, blurred_simulated, reduction='mean')
    
    return loss


def cvd_loss(
    image: torch.Tensor,
    simulated_image: torch.Tensor,
    alpha: float = 15.0,
    beta: float = 1.0,
    lc_patch_size: int = 4,
    lc_stride: int = 2,
    ci_kernel_size: int = 5,
    ci_sigma: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined CVD-Oriented Loss (L_CVD) - Equation 3 from the paper.
    
    L_CVD = α * L_LC(I, δs) + β * L_CI(I, δs)
    
    This loss function is designed to preserve image information after CVD simulation
    to prevent perception bias for color vision deficiency populations.
    
    Parameters
    ----------
    image             : torch.Tensor
                        Original generated image [B, C, H, W]
    simulated_image   : torch.Tensor
                        CVD-simulated image [B, C, H, W]
    alpha             : float
                        Weight for local contrast loss (default: 15.0 as per paper)
    beta              : float
                        Weight for color information loss (default: 1.0 as per paper)
    lc_patch_size     : int
                        Patch size for local contrast loss (default: 4)
    lc_stride         : int
                        Stride for local contrast loss (default: 2)
    ci_kernel_size    : int
                        Kernel size for color information loss (default: 5)
    ci_sigma          : float
                        Sigma for color information loss blur (default: 1.0)
    
    Returns
    -------
    total_loss        : torch.Tensor
                        Combined weighted loss
    lc_loss           : torch.Tensor
                        Local contrast loss component
    ci_loss           : torch.Tensor
                        Color information loss component
    
    Raises
    ------
    TypeError
        If inputs are not torch.Tensor
    ValueError
        If image shapes are incompatible or dimensions too small
    
    Example
    -------
    >>> image = torch.rand(2, 3, 64, 64, requires_grad=True)
    >>> simulated = image * 0.9
    >>> total_loss, lc_loss, ci_loss = cvd_loss(image, simulated)
    >>> total_loss.backward()
    >>> print(f"Total CVD loss: {total_loss.item():.4f}")
    """
    # Compute individual losses
    lc_loss = local_contrast_loss(
        image, simulated_image,
        patch_size=lc_patch_size,
        stride=lc_stride
    )
    
    ci_loss = color_information_loss(
        image, simulated_image,
        kernel_size=ci_kernel_size,
        sigma=ci_sigma
    )
    
    # Combined loss
    total_loss = alpha * lc_loss + beta * ci_loss
    
    return total_loss, lc_loss, ci_loss


def _validate_image_pair(
    image: torch.Tensor,
    simulated_image: torch.Tensor
) -> None:
    """
    Validate that both images are valid tensors with compatible shapes.
    
    Parameters
    ----------
    image             : torch.Tensor
                        First image tensor
    simulated_image   : torch.Tensor
                        Second image tensor
    
    Raises
    ------
    TypeError
        If inputs are not torch.Tensor
    ValueError
        If shapes are incompatible or dimensions are wrong
    """
    # Type check
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be torch.Tensor, got {type(image)}")
    if not isinstance(simulated_image, torch.Tensor):
        raise TypeError(f"simulated_image must be torch.Tensor, got {type(simulated_image)}")
    
    # Shape validation
    if image.dim() != 4:
        raise ValueError(f"image must be 4D [B, C, H, W], got {image.dim()}D")
    if simulated_image.dim() != 4:
        raise ValueError(f"simulated_image must be 4D [B, C, H, W], got {simulated_image.dim()}D")
    
    # Compatible shapes
    if image.shape != simulated_image.shape:
        raise ValueError(
            f"image and simulated_image must have same shape, "
            f"got {image.shape} vs {simulated_image.shape}"
        )
    
    # Positive dimensions (minimum 4x4 for patch extraction)
    if image.shape[2] < 4 or image.shape[3] < 4:
        raise ValueError(
            f"Image dimensions must be at least 4x4, got {image.shape[2]}x{image.shape[3]}"
        )
