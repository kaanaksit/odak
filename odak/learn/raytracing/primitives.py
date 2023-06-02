import torch
from ..tools.vector import same_side
from ..tools.transformation import rotate_point


def define_plane(point, angles = [0., 0., 0.]):
    """ 
    Definition to generate a rotation matrix along X axis.

    Parameters
    ----------
    point        : torch.tensor
                   A point that is at the center of a plane.
    angles       : list
                   Rotation angles in degrees.

    Returns
    ----------
    plane        : torch.tensor
                   Points defining plane.
    """
    plane = torch.tensor([
        [10., 10., 0.],
        [0., 10., 0.],
        [0.,  0., 0.]
    ]).to(point.device)
    for i in range(0, plane.shape[0]):
        plane[i], _, _, _ = rotate_point(plane[i], angles = angles)
        plane[i] = plane[i] + point
    return plane


def center_of_triangle(triangle):
    """
    Definition to calculate center of a triangle.

    Parameters
    ----------
    triangle      : torch.tensor
                    An array that contains three points defining a triangle (Mx3). It can also parallel process many triangles (NxMx3).


    Returns
    -------
    centers       : torch.tensor
                    Triangle centers.
    """
    if len(triangle.shape) == 2:
        triangle = triangle.view((1, 3, 3))
    center = torch.mean(triangle, axis=1)
    return center


def is_it_on_triangle(point_to_check, triangle):
    """
    Definition to check if a given point is inside a triangle. If the given point is inside a defined triangle, this definition returns True.

    Parameters
    ----------
    point_to_check  : torch.tensor
                      Point(s) to check.
                      Expected size is [3], [1 x 3] or [m x 3].
    triangle        : torch.tensor
                      Triangle described with three points.
                      Expected size is [3 x 3], [1 x 3 x 3] or [m x 3 x3].

    Returns
    -------
    result          : torch.tensor
                      Is it on a triangle? Returns NaN if condition not satisfied.
                      Expected size is [1] or [m] depending on the input.
    """
    if len(point_to_check) == 1:
        point_to_check = point_to_check.unsqueeze(0)
    if len(triangle) == 2:
        triangle = triangle.unsqueeze(0)
    w0 = triangle[:, 0] - triangle[:, 1]
    w1 = triangle[:, 0] - triangle[:, 2]
    if len(w0.shape) == 1:
        w0 = w0.unsqueeze(0)
        w1 = w1.unsqueeze(0)
    area = torch.sqrt(torch.sum((w0 * w1) ** 2, dim = 1)) / 2
    p0 = point_to_check - triangle[:, 0]
    p1 = point_to_check - triangle[:, 1]
    p2 = point_to_check - triangle[:, 2]
    alpha = torch.sqrt(torch.sum(torch.cross(p1, p2, dim = 1) ** 2, dim = 1)) / 2 / area
    beta = torch.sqrt(torch.sum(torch.cross(p2, p0, dim = 1) ** 2, dim = 1)) / 2 / area
    gamma = 1. - alpha - beta
    total_sum = alpha + beta + gamma
    result = (alpha >= 0.) * (alpha <= 1.)
    result *= (beta >= 0.) * (beta <= 1.)
    result *= (gamma >= 0.) * (gamma <= 1.)
    result *= (total_sum == 1)
    return result
