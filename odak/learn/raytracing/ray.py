import torch
import numpy as np
from ..tools import rotate_points


def create_ray(x0y0z0, abg):
    """
    Definition to create a ray.

    Parameters
    ----------
    x0y0z0       : list
                   List that contains X,Y and Z start locations of a ray.
    abg          : list
                   List that contaings angles in degrees with respect to the X,Y and Z axes.

    Returns
    ----------
    ray          : torch.tensor
                   Array that contains starting points and cosines of a created ray.
    """
    # Due to Python 2 -> Python 3.
    x0, y0, z0 = x0y0z0
    alpha, beta, gamma = abg
    # Create a vector with the given points and angles in each direction
    point = torch.tensor([x0, y0, z0], dtype=torch.float64)
    alpha = torch.cos(torch.radians(torch.tensor([alpha])))
    beta = torch.cos(torch.radians(torch.tensor([beta])))
    gamma = torch.cos(torch.radians(torch.tensor([gamma])))
    # Cosines vector.
    cosines = torch.tensor([alpha, beta, gamma], dtype=torch.float32)
    ray = torch.tensor([point, cosines], dtype=torch.float32)
    return ray


def create_ray_from_two_points(x0y0z0, x1y1z1):
    """
    Definition to create a ray from two given points. Note that both inputs must match in shape.

    Parameters
    ----------
    x0y0z0       : torch.tensor
                   List that contains X,Y and Z start locations of a ray (3). It can also be a list of points as well (mx3). This is the starting point.
    x1y1z1       : torch.tensor
                   List that contains X,Y and Z ending locations of a ray (3). It can also be a list of points as well (mx3). This is the end point.

    Returns
    ----------
    ray          : torch.tensor
                   Array that contains starting points and cosines of a created ray.
    """
    if len(x0y0z0.shape) == 1:
        x0y0z0 = x0y0z0.view((1, 3))
    if len(x1y1z1.shape) == 1:
        x1y1z1 = x1y1z1.view((1, 3))
    xdiff = x1y1z1[:, 0] - x0y0z0[:, 0]
    ydiff = x1y1z1[:, 1] - x0y0z0[:, 1]
    zdiff = x1y1z1[:, 2] - x0y0z0[:, 2]
    s = (xdiff**2 + ydiff**2 + zdiff**2)**0.5
    s[s == 0] = float('nan')
    cosines = torch.zeros((xdiff.shape[0], 3)).to(x0y0z0.device)
    cosines[:, 0] = xdiff / s
    cosines[:, 1] = ydiff / s
    cosines[:, 2] = zdiff / s
    ray = torch.zeros((xdiff.shape[0], 2, 3)).to(x0y0z0.device)
    ray[:, 0] = x0y0z0
    ray[:, 1] = cosines
    if ray.shape[0] == 1:
        ray = ray.view((2, 3))
    return ray


def propagate_a_ray(ray, distance):
    """
    Definition to propagate a ray at a certain given distance.

    Parameters
    ----------
    ray        : torch.tensor
                 A ray.
    distance   : float
                 Distance.

    Returns
    ----------
    new_ray    : torch.tensor
                 Propagated ray.
    """
    if len(ray.shape) == 2:
        ray = ray.reshape((1, 2, 3))
    new_ray = ray.clone()
    new_ray[:, 0, 0] = distance*new_ray[:, 1, 0] + new_ray[:, 0, 0]
    new_ray[:, 0, 1] = distance*new_ray[:, 1, 1] + new_ray[:, 0, 1]
    new_ray[:, 0, 2] = distance*new_ray[:, 1, 2] + new_ray[:, 0, 2]
    if new_ray.shape[0] == 1:
        new_ray = new_ray.reshape((2, 3))
    return new_ray
