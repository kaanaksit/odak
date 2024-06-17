import torch
from ..tools.vector import same_side
from ..tools.transformation import rotate_points


def define_plane(point, angles = torch.tensor([0., 0., 0.])):
    """ 
    Definition to generate a rotation matrix along X axis.

    Parameters
    ----------
    point        : torch.tensor
                   A point that is at the center of a plane.
    angles       : torch.tensor
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
        plane[i], _, _, _ = rotate_points(plane[i], angles = angles.to(point.device))
        plane[i] = plane[i] + point
    return plane


def define_plane_mesh(
                      number_of_meshes = [10, 10], 
                      size = [1., 1.], 
                      angles = torch.tensor([0., 0., 0.]), 
                      offset = torch.tensor([[0., 0., 0.]])
                     ):
    """
    Definition to generate a plane with meshes.


    Parameters
    -----------
    number_of_meshes  : torch.tensor
                        Number of squares over plane.
                        There are two triangles at each square.
    size              : list
                        Size of the plane.
    angles            : torch.tensor
                        Rotation angles in degrees.
    offset            : torch.tensor
                        Offset along XYZ axes.
                        Expected dimension is [1 x 3] or offset for each triangle [m x 3].
                        m here refers to `2 * number_of_meshes[0]` times  `number_of_meshes[1]`. 
    
    Returns
    -------
    triangles         : torch.tensor
                        Triangles [m x 3 x 3], where m is `2 * number_of_meshes[0]` times  `number_of_meshes[1]`.
    """
    triangles = torch.zeros(2, number_of_meshes[0], number_of_meshes[1], 3, 3)
    step = [size[0] / number_of_meshes[0], size[1] / number_of_meshes[1]]
    for i in range(0, number_of_meshes[0] - 1):
        for j in range(0, number_of_meshes[1] - 1):
            first_triangle = torch.tensor([
                                           [       -size[0] / 2. + step[0] * i,       -size[1] / 2. + step[0] * j, 0.],
                                           [ -size[0] / 2. + step[0] * (i + 1),       -size[1] / 2. + step[0] * j, 0.],
                                           [       -size[0] / 2. + step[0] * i, -size[1] / 2. + step[0] * (j + 1), 0.]
                                          ])
            second_triangle = torch.tensor([
                                            [ -size[0] / 2. + step[0] * (i + 1), -size[1] / 2. + step[0] * (j + 1), 0.],
                                            [ -size[0] / 2. + step[0] * (i + 1),       -size[1] / 2. + step[0] * j, 0.],
                                            [       -size[0] / 2. + step[0] * i, -size[1] / 2. + step[0] * (j + 1), 0.]
                                           ])
            triangles[0, i, j], _, _, _ = rotate_points(first_triangle, angles = angles)
            triangles[1, i, j], _, _, _ = rotate_points(second_triangle, angles = angles)
    triangles = triangles.view(-1, 3, 3) + offset
    return triangles


def center_of_triangle(triangle):
    """
    Definition to calculate center of a triangle.

    Parameters
    ----------
    triangle      : torch.tensor
                    An array that contains three points defining a triangle (Mx3). 
                    It can also parallel process many triangles (NxMx3).

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
    Definition to check if a given point is inside a triangle. 
    If the given point is inside a defined triangle, this definition returns True.
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


def is_it_on_triangle_batch(point_to_check, triangle):
    """
    Definition to check if given points are inside triangles. If the given points are inside defined triangles, this definition returns True.

    Parameters
    ----------
    point_to_check  : torch.tensor
                      Points to check (m x n x 3).
    triangle        : torch.tensor 
                      Triangles (m x 3 x 3).

    Returns
    ----------
    result          : torch.tensor (m x n)
                        
    """
    if len(point_to_check.shape) == 1:
        point_to_check = point_to_check.unsqueeze(0)
    if len(triangle.shape) == 2:
        triangle = triangle.unsqueeze(0)
    v0 = triangle[:, 2] - triangle[:, 0]
    v1 = triangle[:, 1] - triangle[:, 0]
    v2 = point_to_check - triangle[:, None, 0]
    if len(v0.shape) == 1:
        v0 = v0.unsqueeze(0)
    if len(v1.shape) == 1:
        v1 = v1.unsqueeze(0)
    if len(v2.shape) == 1:
        v2 = v2.unsqueeze(0)

    dot00 = torch.bmm(v0.unsqueeze(1), v0.unsqueeze(1).permute(0, 2, 1)).squeeze(1)
    dot01 = torch.bmm(v0.unsqueeze(1), v1.unsqueeze(1).permute(0, 2, 1)).squeeze(1)
    dot02 = torch.bmm(v0.unsqueeze(1), v2.permute(0, 2, 1)).squeeze(1)
    dot11 = torch.bmm(v1.unsqueeze(1), v1.unsqueeze(1).permute(0, 2, 1)).squeeze(1)
    dot12 = torch.bmm(v1.unsqueeze(1), v2.permute(0, 2, 1)).squeeze(1)
    invDenom = 1. / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    result = (u >= 0.) & (v >= 0.) & ((u + v) < 1)

    return result

def define_sphere(center = torch.tensor([[0., 0., 0.]]), radius = torch.tensor([1.])):
    """
    Definition to define a sphere.

    Parameters
    ----------
    center      : torch.tensor
                  Center of the sphere(s) along XYZ axes.
                  Expected size is [3], [1, 3] or [m, 3].
    radius      : torch.tensor
                  Radius of that sphere(s).
                  Expected size is [1], [1, 1], [m] or [m, 1].

    Returns
    -------
    parameters  : torch.tensor
                  Parameters of defined sphere(s).
                  Expected size is [1, 3] or [m x 3].
    """
    if len(radius.shape) == 1:
        radius = radius.unsqueeze(0)
    if len(center.shape) == 1:
        center = center.unsqueeze(1)
    parameters = torch.cat((center, radius), dim = 1)
    return parameters
                  

def define_circle(center, radius, angles):
    """
    Definition to describe a circle in a single variable packed form.

    Parameters
    ----------
    center  : torch.Tensor
              Center of a circle to be defined in 3D space.
    radius  : float
              Radius of a circle to be defined.
    angles  : torch.Tensor
              Angular tilt of a circle represented by rotations about x, y, and z axes.

    Returns
    ----------
    circle  : list
              Single variable packed form.
    """
    points = define_plane(center, angles=angles)
    circle = [
        points,
        center,
        torch.tensor([radius])
    ]
    return circle
