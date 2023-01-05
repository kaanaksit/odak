import numpy as np
from . import create_ray_from_angles
from ..tools.transformation import rotate_point, rotate_points
from ..tools.vector import same_side, point_to_ray_distance


def define_plane(point, angles=[0., 0., 0.]):
    """ 
    Definition to generate a rotation matrix along X axis.

    Parameters
    ----------
    point        : ndarray
                   A point that is at the center of a plane.
    angles       : list
                   Rotation angles in degrees.

    Returns
    ----------
    plane        : ndarray
                   Points defining plane.
    """
    plane = np.array([
        [10., 10., 0.],
        [0., 10., 0.],
        [0.,  0., 0.]
    ], dtype=np.float64)
    point = np.asarray(point)
    for i in range(0, plane.shape[0]):
        plane[i], _, _, _ = rotate_point(plane[i], angles=angles)
        plane[i] = plane[i]+point
    return plane


def bring_plane_to_origin(point, plane, shape=[10., 10.], center=[0., 0., 0.], angles=[0., 0., 0.], mode='XYZ'):
    """
    Definition to bring points back to reference origin with respect to a plane.

    Parameters
    ----------
    point              : ndarray
                         Point(s) to be tested.
    shape              : list
                         Dimensions of the rectangle along X and Y axes.
    center             : list
                         Center of the rectangle.
    angles             : list
                         Rotation angle of the rectangle.
    mode               : str
                         Rotation mode of the rectangle, for more see odak.tools.rotate_point and odak.tools.rotate_points.

    Returns
    ----------
    transformed_points : ndarray
                         Point(s) that are brought back to reference origin with respect to given plane.
    """
    if point.shape[0] == 3:
        point = point.reshape((1, 3))
    reverse_mode = mode[::-1]
    angles = [-angles[0], -angles[1], -angles[2]]
    center = np.asarray(center).reshape((1, 3))
    transformed_points = point-center
    transformed_points = rotate_points(
        transformed_points,
        angles=angles,
        mode=reverse_mode,
    )
    if transformed_points.shape[0] == 1:
        transformed_points = transformed_points.reshape((3,))
    return transformed_points


def center_of_triangle(triangle):
    """
    Definition to calculate center of a triangle.

    Parameters
    ----------
    triangle      : ndarray
                    An array that contains three points defining a triangle (Mx3). It can also parallel process many triangles (NxMx3).
    """
    if len(triangle.shape) == 2:
        triangle = triangle.reshape((1, 3, 3))
    center = np.mean(triangle, axis=1)
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
    pointtocheck = np.asarray(pointtocheck).reshape(3)
    point0 = np.asarray(point0)
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    side0 = same_side(pointtocheck, point0, point1, point2)
    side1 = same_side(pointtocheck, point1, point0, point2)
    side2 = same_side(pointtocheck, point2, point0, point1)
    if side0 == True and side1 == True and side2 == True:
        return True
    return False


def define_circle(center, radius, angles):
    """
    Definition to describe a circle in a single variable packed form.

    Parameters
    ----------
    center  : float
              Center of a circle to be defined.
    radius  : float
              Radius of a circle to be defined.
    angles  : float
              Angular tilt of a circle.

    Returns
    ----------
    circle  : list
              Single variable packed form.
    """
    points = define_plane(center, angles=angles)
    circle = [
        points,
        center,
        radius
    ]
    return circle


def define_sphere(center, radius):
    """
    Definition to define a sphere.

    Parameters
    ----------
    center     : ndarray
                 Center of a sphere in X,Y,Z.
    radius     : float
                 Radius of a sphere.

    Returns
    ----------
    sphere     : ndarray
                 Single variable packed form.
    """
    sphere = np.array(
        [center[0], center[1], center[2], radius], dtype=np.float64)
    return sphere


def sphere_function(point, sphere):
    """
    Definition of a sphere function. Evaluate a point against a sphere function.

    Parameters
    ----------
    sphere     : ndarray
                 Sphere parameters, XYZ center and radius.
    point      : ndarray
                 Point in XYZ.

    Return
    ----------
    result     : float
                 Result of the evaluation. Zero if point is on sphere.
    """
    point = np.asarray(point)
    if len(point.shape) == 1:
        point = point.reshape((1, 3))
    result = (point[:, 0]-sphere[0])**2 + (point[:, 1]-sphere[1]
                                           )**2 + (point[:, 2]-sphere[2])**2 - sphere[3]**2
    return result


def define_cylinder(center, radius, rotation=[0., 0., 0.]):
    """
    Definition to define a cylinder

    Parameters
    ----------
    center     : ndarray
                 Center of a cylinder in X,Y,Z.
    radius     : float
                 Radius of a cylinder along X axis.
    rotation   : list
                 Direction angles in degrees for the orientation of a cylinder.

    Returns
    ----------
    cylinder   : ndarray
                 Single variable packed form.
    """
    cylinder_ray = create_ray_from_angles(
        np.asarray(center), np.asarray(rotation))
    cylinder = np.array(
        [
            center[0],
            center[1],
            center[2],
            radius,
            center[0]+cylinder_ray[1, 0],
            center[1]+cylinder_ray[1, 1],
            center[2]+cylinder_ray[1, 2]
        ],
        dtype=np.float64
    )
    return cylinder


def cylinder_function(point, cylinder):
    """
    Definition of a cylinder function. Evaluate a point against a cylinder function. Inspired from https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

    Parameters
    ----------
    sphere     : ndarray
                 Sphere parameters, XYZ center and radius.
    point      : ndarray
                 Point in XYZ.

    Return
    ----------
    result     : float
                 Result of the evaluation. Zero if point is on sphere.
    """
    point = np.asarray(point)
    if len(point.shape) == 1:
        point = point.reshape((1, 3))
    distance = point_to_ray_distance(
        point,
        np.array([cylinder[0], cylinder[1], cylinder[2]], dtype=np.float64),
        np.array([cylinder[4], cylinder[5], cylinder[6]], dtype=np.float64)
    )
    r = cylinder[3]
    result = distance-r**2
    return result
