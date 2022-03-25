import torch
from .transformation import rotate_points, rotate_point


def grid_sample(no=[10, 10], size=[100., 100.], center=[0., 0., 0.], angles=[0., 0., 0.]):
    """
    Definition to generate samples over a surface.
    Parameters
    ----------
    no          : list
                  Number of samples.
    size        : list
                  Physical size of the surface.
    center      : list
                  Center location of the surface.
    angles      : list
                  Tilt of the surface.
    Returns
    ----------
    samples     : torch.tensor
                  Samples generated.
    rotx        : torch.tensor
                  Rotation matrix at X axis.
    roty        : torch.tensor
                  Rotation matrix at Y axis.
    rotz        : torch.tensor
                  Rotation matrix at Z axis.
    """
    samples = torch.zeros((no[0], no[1], 3))
    step = [
        size[0]/(no[0]-1),
        size[1]/(no[1]-1)
    ]
    x = torch.linspace(-size[0], size[0], no[0])
    y = torch.linspace(-size[1], size[1], no[1])
    X, Y = torch.meshgrid(x, y, indexing='ij')
    samples[:, :, 0] = X.detach().clone()
    samples[:, :, 1] = Y.detach().clone()
    samples = samples.reshape(
        (samples.shape[0]*samples.shape[1], samples.shape[2]))
    samples, rotx, roty, rotz = rotate_points(samples, angles=angles, offset=center)
    return samples, rotx, roty, rotz
