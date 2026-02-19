import torch
from .transformation import rotate_points


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
