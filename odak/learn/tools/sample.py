import torch
from .transformation import rotate_points


def grid_sample(
    no=[10, 10], size=[100.0, 100.0], center=[0.0, 0.0, 0.0], angles=[0.0, 0.0, 0.0]
):
    """
    Generate samples over a surface.

    Parameters
    ----------
    no : list
        Number of samples along each dimension.
    size : list
        Physical size of the surface along each dimension.
    center : list
        Center location of the surface.
    angles : list
        Tilt angles of the surface around X, Y, and Z axes.

    Returns
    -------
    samples : torch.tensor
        Generated samples.
    rotx : torch.tensor
        Rotation matrix around X axis.
    roty : torch.tensor
        Rotation matrix around Y axis.
    rotz : torch.tensor
        Rotation matrix around Z axis.
    """
    center = torch.tensor(center, dtype=torch.float32)
    angles = torch.tensor(angles, dtype=torch.float32)
    size = torch.tensor(size, dtype=torch.float32)
    samples = torch.zeros((no[0], no[1], 3), dtype=torch.float32)
    x = torch.linspace(-size[0] / 2.0, size[0] / 2.0, no[0])
    y = torch.linspace(-size[1] / 2.0, size[1] / 2.0, no[1])
    X, Y = torch.meshgrid(x, y, indexing="ij")
    samples[:, :, 0] = X
    samples[:, :, 1] = Y
    samples = samples.reshape((-1, 3))
    samples, rotx, roty, rotz = rotate_points(samples, angles=angles, offset=center)
    return samples, rotx, roty, rotz
