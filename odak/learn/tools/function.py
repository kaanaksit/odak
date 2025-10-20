import torch
from .transformation import rotate_points


def evaluate_3d_gaussians(
                          points: torch.Tensor,
                          centers: torch.Tensor,
                          sigmas: torch.Tensor,
                          angles: torch.Tensor,
                         ) -> torch.Tensor:
    """
    Evaluate 3D Gaussian functions at given points, with optional rotation.

    Parameters
    ----------
    points : torch.Tensor, shape [n, 3]
        The 3D points at which to evaluate the Gaussians.
    centers : torch.Tensor, shape [n, 3]
        The centers of the Gaussians.
    sigmas : torch.Tensor, shape [n, 3]
        The standard deviations (spread) of the Gaussians along each axis.
    angles : torch.Tensor, shape [n, 3]
        The rotation angles (in radians) for each Gaussian, applied to the points.

    Returns
    -------
    torch.Tensor, shape [n]
        The evaluated Gaussian intensities at each point.

    Notes
    -----
    - The function first rotates the points according to the given angles and centers.
    - The Gaussian is evaluated as:
      .. math::
         I = \\frac{1}{\\sigma_x \\sigma_y \\sigma_z (2\\pi)^{3/2}}
             \\exp\\left(-\\frac{1}{2}\\sum_{i=1}^3 \\left(\\frac{x_i'}{\\sigma_i}\\right)^2\\right)
      where $\\(x'\\)$ are the rotated points.
    - If `sigmas` has more than one dimension, it is reshaped appropriately.
    """
    points_rotated, _, _, _ = rotate_points(
                                            point = points,
                                            angles = angles,
                                            offset = centers
                                           )
    if sigmas.shape[0] > 1:
        sigmas = sigmas.unsqueeze(1)
    exponent = torch.sum(-0.5 * (points_rotated / sigmas) ** 2, dim = -1)
    if len(sigmas.shape) == 3:
        sigmas = sigmas.squeeze(1)
    divider = (sigmas[:, 0] * sigmas[:, 1] * sigmas[:, 2]) * (2. * torch.pi) ** (3. / 2.)
    divider = divider.unsqueeze(-1)
    exponential = torch.exp(exponent)
    intensities = exponential / divider
    return intensities


