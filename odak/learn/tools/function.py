import torch
from .transformation import rotate_points
import torch.nn.functional as F


def evaluate_3d_gaussians(
    points,
    centers=torch.zeros(1, 3),
    scales=torch.ones(1, 3),
    angles=torch.zeros(1, 3),
    opacity=torch.ones(1, 1),
) -> torch.Tensor:
    """
    Evaluate 3D Gaussian functions at given points, with optional rotation.

    Parameters
    ----------
    points      : torch.Tensor, shape [n, 3]
                  The 3D points at which to evaluate the Gaussians.
    centers     : torch.Tensor, shape [n, 3]
                  The centers of the Gaussians.
    scales      : torch.Tensor, shape [n, 3]
                  The standard deviations (spread) of the Gaussians along each axis.
    angles      : torch.Tensor, shape [n, 3]
                  The rotation angles (in radians) for each Gaussian, applied to the points.
    opacity     : torch.Tensor, shape [n, 1]
                  Opacity of the Gaussians.

    Returns
    -------
    intensities : torch.Tensor, shape [n, 1]
                  The evaluated Gaussian intensities at each point.
    """
    points_rotated, _, _, _ = rotate_points(point=points, angles=angles, origin=centers)
    points_rotated = points_rotated - centers.unsqueeze(0)
    scales = scales.unsqueeze(0)
    exponent = torch.sum(-0.5 * (points_rotated / scales) ** 2, dim=-1)
    divider = (scales[:, :, 0] * scales[:, :, 1] * scales[:, :, 2]) * (
        2.0 * torch.pi
    ) ** (3.0 / 2.0)
    exponential = torch.exp(exponent)
    intensities = exponential / divider
    intensities = opacity.T * intensities
    return intensities


def zernike_polynomial(
    n,
    m,
    rho,
    theta,
):
    """
    Compute the 2D Zernike polynomial Z_n^m(rho, theta).

    Parameters
    ----------
    n          : int
                 Radial degree of the polynomial (n >= 0).
    m          : int
                 Azimuthal frequency of the polynomial. Must satisfy |m| <= n and (n - |m|) % 2 == 0.
    rho        : torch.Tensor
                 Radial distance from the origin (0 to 1). Shape (H, W).
    theta      : torch.Tensor
                 Azimuthal angle in radians. Shape (H, W).


    Returns
    -------
    zernike    : torch.Tensor
                 The computed 2D Zernike polynomial.
                 Values are zero where rho > 1.
    """
    m_abs = abs(m)
    if m_abs > n or (n - m_abs) % 2 != 0:
        return torch.zeros(rho.shape, dtype=torch.complex64, device=rho.device)

    radial = torch.zeros_like(rho)

    for k in range((n - m_abs) // 2 + 1):
        num = (-1) ** k * torch.exp(torch.lgamma(torch.tensor(n - k + 1.0)))
        den = (
            torch.exp(torch.lgamma(torch.tensor(k + 1.0)))
            * torch.exp(torch.lgamma(torch.tensor((n + m_abs) // 2 - k + 1.0)))
            * torch.exp(torch.lgamma(torch.tensor((n - m_abs) // 2 - k + 1.0)))
        )
        radial += (num / den) * torch.pow(rho, n - 2 * k)

    if m >= 0:
        zernike = radial * torch.cos(m * theta)
    else:
        zernike = radial * torch.sin(m_abs * theta)
    zernike[rho > 1] = 0

    return zernike


def generate_bspline_basis_2d(num_control_points, resolution, device=torch.device("cpu")):
    """
    Generate a 2D B-spline basis function matrix for decomposition using proper uniform cubic B-splines.
    
    Parameters
    ----------
    num_control_points : list
        Number of control points in [x, y] directions, e.g., [8, 8].
    resolution : list
        Output resolution [height, width] for evaluating the basis.
    device : torch.device
        Device to create the tensors on.
    
    Returns
    ----- --
    basis_matrix : torch.Tensor
        Basis matrix of shape (num_control_points[0]*num_control_points[1], resolution[0]*resolution[1]).
    control_grid : torch.Tensor
        Grid of control point positions normalized to [-1, 1].
    """
    H, W = resolution
    Cp_x, Cp_y = num_control_points
    device = torch.device(device) if not isinstance(device, torch.device) else device
    
    x_coords = torch.linspace(0, 1, W, device=device)
    y_coords = torch.linspace(0, 1, H, device=device)
    
    def cubic_bspline(t):
        t = t.float()
        result = torch.zeros_like(t)
        mask = torch.abs(t) <= 1
        t_abs = torch.abs(t[mask])
        result[mask] = (2.0/3.0) - t_abs**2 + 0.5*t_abs**3
        mask2 = (torch.abs(t) > 1) & (torch.abs(t) <= 2)
        t_abs2 = torch.abs(t[mask2])
        result[mask2] = (2.0 - t_abs2)**3 / 6.0
        return result
    
    def compute_basis_1d(num_cp, coords):
        n_points = len(coords)
        basis = torch.zeros(num_cp, n_points, device=device)
        if num_cp == 1:
            return torch.ones(1, n_points, device=device)
        spacing = 1.0 / (num_cp - 1)
        for cp in range(num_cp):
            cp_pos = cp * spacing
            t = (coords - cp_pos) / spacing
            basis[cp, :] = cubic_bspline(t)
        return basis
    
    basis_x = compute_basis_1d(Cp_x, x_coords)
    basis_y = compute_basis_1d(Cp_y, y_coords)
    
    basis_matrix = torch.zeros(Cp_x * Cp_y, H * W, device=device)
    for i in range(Cp_y):
        for j in range(Cp_x):
            idx = i * Cp_x + j
            basis_matrix[idx, :] = torch.outer(basis_y[i, :], basis_x[j, :]).flatten()
    
    control_x = torch.linspace(-1, 1, Cp_x, device=device)
    control_y = torch.linspace(-1, 1, Cp_y, device=device)
    control_grid = torch.stack(torch.meshgrid(control_x, control_y, indexing='ij'), dim=-1)
    
    return basis_matrix, control_grid


def decompose_bspline_2d(image, num_control_points, regularization=1e-6):
    """
    Decompose a 2D image into cubic B-spline control point coefficients.
    """
    if image.dim() == 3:
        image = image.squeeze(0)
    
    H, W = image.shape
    Cp_x, Cp_y = num_control_points
    device = image.device
    image_flat = image.view(-1)
    
    basis_matrix, control_grid = generate_bspline_basis_2d(num_control_points, [H, W], device)
    
    BtB = basis_matrix @ basis_matrix.t()
    Bty = basis_matrix @ image_flat
    reg_matrix = regularization * torch.eye(BtB.shape[0], device=device)
    
    try:
        control_coefficients = torch.linalg.solve(BtB + reg_matrix, Bty)
    except:
        control_coefficients = torch.linalg.lstsq(basis_matrix, image_flat).solution
    
    return control_coefficients, basis_matrix, control_grid


def compose_bspline_2d(control_coefficients, basis_matrix, resolution):
    """
    Reconstruct a 2D image from B-spline control point coefficients.
    
    Parameters
    ----------
    control_coefficients : torch.Tensor
        Control point coefficients of shape (num_control_points[0]*num_control_points[1],).
    basis_matrix : torch.Tensor
        Basis matrix from decompose_bspline_2d.
    resolution : list
        Original resolution [height, width].
    
    Returns
    ----- --
    reconstructed : torch.Tensor
        Reconstructed 2D image of shape (height, width).
    """
    H, W = resolution
    reconstructed = basis_matrix.t() @ control_coefficients
    return reconstructed.view(H, W)


def decompose_wavelet_like(image, n_scales=4):
    """
    Decompose a 2D image into multi-scale B-spline smooths and detail residuals.
    
    This wavelet-like decomposition achieves perfect reconstruction by storing
    the high-frequency residuals at each scale. Ideal for faithful representation
    of images with fine detail.
    
    Parameters
    ----------
    image : torch.Tensor
        Input 2D image of shape (H, W) or (1, H, W).
    n_scales : int
        Number of decomposition scales (default: 4). Each scale captures
        progressively finer details.
    
    Returns
    ----- --
    coefficients : list of torch.Tensor
        B-spline coefficients at each scale.
    residuals : list of torch.Tensor
        Detail residuals at each scale (full frequency information).
    base : torch.Tensor
        Coarse approximation after all decomposition levels.
    
    Examples
    --------
    >>> coeffs, residuals, base = decompose_wavelet_like(phase_image, n_scales=5)
    >>> reconstructed = reconstruct_wavelet_like(coeffs, residuals, base)
    """
    H, W = image.shape if image.dim() == 2 else image.shape[1:]
    image = image.view(H, W)
    
    coefficients = []
    residuals = []
    current = image.clone()
    
    for scale in range(n_scales):
        # Determine control point count for this scale
        base_size = min(H, W) // (2 ** scale)
        cp_size = max(4, base_size // 8)
        cp_size = min(cp_size, min(H, W) // 2)
        
        # Fit B-spline to current residual
        coeffs, basis, _ = decompose_bspline_2d(current, [cp_size, cp_size], regularization=1e-8)
        smooth = compose_bspline_2d(coeffs, basis, [H, W])
        
        # Store detail residual
        residual = current - smooth
        residuals.append(residual)
        coefficients.append(coeffs)
        
        current = smooth
    
    return coefficients, residuals, current


def reconstruct_wavelet_like(coefficients, residuals, base):
    """
    Reconstruct image from wavelet-like decomposition.
    
    Parameters
    ----------
    coefficients : list of torch.Tensor
        B-spline coefficients from decompose_wavelet_like.
    residuals : list of torch.Tensor
        Detail residuals from decompose_wavelet_like.
    base : torch.Tensor
        Coarse base from decompose_wavelet_like.
    
    Returns
    ----- --
    reconstructed : torch.Tensor
        Fully reconstructed 2D image.
    """
    result = base.clone()
    for residual in reversed(residuals):
        result = result + residual
    return result

