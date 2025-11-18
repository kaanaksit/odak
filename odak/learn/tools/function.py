import torch
from .transformation import rotate_points


def evaluate_3d_gaussians(
                          points,
                          centers = torch.zeros(1, 3),
                          sigmas = torch.ones(1, 3),
                          angles = torch.zeros(1, 3),
                          opacity = torch.ones(1, 1),
                         ) -> torch.Tensor:
    """
    Evaluate 3D Gaussian functions at given points, with optional rotation.

    Parameters
    ----------
    points      : torch.Tensor, shape [n, 3]
                  The 3D points at which to evaluate the Gaussians.
    centers     : torch.Tensor, shape [n, 3]
                  The centers of the Gaussians.
    sigmas      : torch.Tensor, shape [n, 3]
                  The standard deviations (spread) of the Gaussians along each axis.
    angles      : torch.Tensor, shape [n, 3]
                  The rotation angles (in radians) for each Gaussian, applied to the points.
    opacity     : torch.Tensor, shape [n, 1]
                  Opacity of the Gaussians.

    Returns
    -------
    intensities : torch.Tensor, shape [n, 3]
                  The evaluated Gaussian intensities at each point.
    """
    points_rotated, _, _, _ = rotate_points(
                                            point = points,
                                            angles = angles,
                                            origin = centers
                                           )
    points_rotated = points_rotated - centers.unsqueeze(0)
    sigmas = sigmas.unsqueeze(0)
    exponent = torch.sum(-0.5 * (points_rotated / sigmas) ** 2, dim = -1)
    divider = (sigmas[:, :, 0] * sigmas[:, :, 1] * sigmas[:, :, 2]) * (2. * torch.pi) ** (3. / 2.)
    exponential = torch.exp(exponent)
    intensities = (exponential / divider)
    intensities = opacity.T * intensities
    return intensities
