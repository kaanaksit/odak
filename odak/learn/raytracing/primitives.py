import torch
from odak.learn.tools.vector import same_side
from odak.learn.tools.transformation import rotate_point


def define_plane(point, angles=[0., 0., 0.]):
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
    """
    if len(triangle.shape) == 2:
        triangle = triangle.view((1, 3, 3))
    center = torch.mean(triangle, axis=1)
    return center


def is_it_on_triangle(pointtocheck, point0, point1, point2):
    """
    Definition to check if a given point is inside a triangle. If the given point is inside a defined triangle, this definition returns True.

    Parameters
    ----------
    pointtocheck  : list
                    Point to check.
    point0        : list
                    First point of a triangle.
    point1        : list
                    Second point of a triangle.
    point2        : list
                    Third point of a triangle.
    """
    # point0, point1 and point2 are the corners of the triangle.
    pointtocheck = pointtocheck.reshape(3)
    side0 = same_side(pointtocheck, point0, point1, point2)
    side1 = same_side(pointtocheck, point1, point0, point2)
    side2 = same_side(pointtocheck, point2, point0, point1)
    if side0 == True and side1 == True and side2 == True:
        return True
    return False
