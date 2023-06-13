import torch
import numpy as np
from ..tools import rotate_points


def create_ray(xyz, abg):
    """
    Definition to create a ray.

    Parameters
    ----------
    xyz          : torch.tensor
                   List that contains X,Y and Z start locations of a ray.
                   Size could be [1 x 3], [3], [m x 3].
    abg          : torch.tensor
                   List that contains angles in degrees with respect to the X,Y and Z axes.
                   Size could be [1 x 3], [3], [m x 3].

    Returns
    ----------
    ray          : torch.tensor
                   Array that contains starting points and cosines of a created ray.
                   Size will be either [1 x 3] or [m x 3].
    """
    points = xyz
    angles = abg
    if len(xyz) == 1:
        points = xyz.unsqueeze(0)
    if len(abg) == 1:
        angles = abg.unsqueeze(0)
    ray = torch.zeros(points.shape[0], 2, 3, device = points.device)
    ray[:, 0] = points
    ray[:, 1] = torch.cos(torch.deg2rad(abg))
    return ray


def create_ray_from_two_points(x0y0z0, x1y1z1):
    """
    Definition to create a ray from two given points. Note that both inputs must match in shape.

    Parameters
    ----------
    x0y0z0       : torch.tensor
                   List that contains X,Y and Z start locations of a ray.
                   Size could be [1 x 3], [3], [m x 3].
    x1y1z1       : torch.tensor
                   List that contains X,Y and Z ending locations of a ray or batch of rays.
                   Size could be [1 x 3], [3], [m x 3].

    Returns
    ----------
    ray          : torch.tensor
                   Array that contains starting points and cosines of a created ray(s).
    """
    if len(x0y0z0.shape) == 1:
        x0y0z0 = x0y0z0.unsqueeze(0)
    if len(x1y1z1.shape) == 1:
        x1y1z1 = x1y1z1.unsqueeze(0)
    xdiff = x1y1z1[:, 0] - x0y0z0[:, 0]
    ydiff = x1y1z1[:, 1] - x0y0z0[:, 1]
    zdiff = x1y1z1[:, 2] - x0y0z0[:, 2]
    s = (xdiff ** 2 + ydiff ** 2 + zdiff ** 2) ** 0.5
    s[s == 0] = float('nan')
    cosines = torch.zeros_like(x0y0z0 * x1y1z1)
    cosines[:, 0] = xdiff / s
    cosines[:, 1] = ydiff / s
    cosines[:, 2] = zdiff / s
    ray = torch.zeros(xdiff.shape[0], 2, 3, device = x0y0z0.device)
    ray[:, 0] = x0y0z0
    ray[:, 1] = cosines
    return ray


def propagate_ray(ray, distance):
    """
    Definition to propagate a ray at a certain given distance.

    Parameters
    ----------
    ray        : torch.tensor
                 A ray with a size of [2 x 3], [1 x 2 x 3] or a batch of rays with [m x 2 x 3].
    distance   : torch.tensor
                 Distance with a size of [1], [1, m] or distances with a size of [m], [1, m].

    Returns
    ----------
    new_ray    : torch.tensor
                 Propagated ray with a size of [1 x 2 x 3] or batch of rays with [m x 2 x 3].
    """
    if len(ray.shape) == 2:
        ray = ray.unsqueeze(0)
    if len(distance.shape) == 2:
        distance = distance.squeeze(-1)
    new_ray = torch.zeros_like(ray)
    new_ray[:, 0, 0] = distance * ray[:, 1, 0] + ray[:, 0, 0]
    new_ray[:, 0, 1] = distance * ray[:, 1, 1] + ray[:, 0, 1]
    new_ray[:, 0, 2] = distance * ray[:, 1, 2] + ray[:, 0, 2]
    return new_ray
