import torch
from ..tools.vector import same_side
from ..tools.transformation import rotate_points


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
                         ], device = point.device)
    for i in range(0, plane.shape[0]):
        plane[i], _, _, _ = rotate_points(plane[i], angles = angles)
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
    For more details, visit: [https://blackpawn.com/texts/pointinpoly/](https://blackpawn.com/texts/pointinpoly/).

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
    if len(point_to_check.shape) == 1:
        point_to_check = point_to_check.unsqueeze(0)
    if len(triangle.shape) == 2:
        triangle = triangle.unsqueeze(0)
    v0 = triangle[:, 2] - triangle[:, 0]
    v1 = triangle[:, 1] - triangle[:, 0]
    v2 = point_to_check - triangle[:, 0]
    if len(v0.shape) == 1:
        v0 = v0.unsqueeze(0)
    if len(v1.shape) == 1:
        v1 = v1.unsqueeze(0)
    if len(v2.shape) == 1:
        v2 = v2.unsqueeze(0)
    dot00 = torch.mm(v0, v0.T)
    dot01 = torch.mm(v0, v1.T)
    dot02 = torch.mm(v0, v2.T) 
    dot11 = torch.mm(v1, v1.T)
    dot12 = torch.mm(v1, v2.T)
    invDenom = 1. / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    result = (u >= 0.) & (v >= 0.) & ((u + v) < 1)
    return result
