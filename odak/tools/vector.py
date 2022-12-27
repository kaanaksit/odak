import numpy as np


def cross_product(vector1, vector2):
    """
    Definition to cross product two vectors and return the resultant vector. Used method described under: http://en.wikipedia.org/wiki/Cross_product

    Parameters
    ----------
    vector1      : ndarray
                   A vector/ray.
    vector2      : ndarray
                   A vector/ray.

    Returns
    ----------
    ray          : ndarray
                   Array that contains starting points and cosines of a created ray.
    """
    angle = np.cross(vector1[1].T, vector2[1].T)
    angle = np.asarray(angle)
    ray = np.array([vector1[0], angle], dtype=np.float32)
    return ray


def same_side(p1, p2, a, b):
    """
    Definition to figure which side a point is on with respect to a line and a point. See http://www.blackpawn.com/texts/pointinpoly/ for more. If p1 and p2 are on the sameside, this definition returns True.

    Parameters
    ----------
    p1          : list
                  Point(s) to check.
    p2          : list
                  This is the point check against.
    a           : list
                  First point that forms the line.
    b           : list
                  Second point that forms the line.
    """
    ba = np.subtract(b, a)
    p1a = np.subtract(p1, a)
    p2a = np.subtract(p2, a)
    cp1 = np.cross(ba, p1a)
    cp2 = np.cross(ba, p2a)
    test = np.dot(cp1, cp2)
    if len(p1.shape) > 1:
        return test >= 0
    if test >= 0:
        return True
    return False


def distance_between_point_clouds(points0, points1):
    """
    A definition to find distance between every point in one cloud to other points in the other point cloud.
    Parameters
    ----------
    points0     : ndarray
                  Mx3 points.
    points1     : ndarray
                  Nx3 points.

    Returns
    ----------
    distances   : ndarray
                  MxN distances.
    """
    c = points1.reshape((1, points1.shape[0], points1.shape[1]))
    a = np.repeat(c, points0.shape[0], axis=0)
    b = points0.reshape((points0.shape[0], 1, points0.shape[1]))
    b = np.repeat(b, a.shape[1], axis=1)
    distances = np.sqrt(np.sum((a-b)**2, axis=2))
    return distances


def distance_between_two_points(point1, point2):
    """
    Definition to calculate distance between two given points.

    Parameters
    ----------
    point1      : list
                  First point in X,Y,Z.
    point2      : list
                  Second point in X,Y,Z.

    Returns
    ----------
    distance    : float
                  Distance in between given two points.
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    if len(point1.shape) == 1 and len(point2.shape) == 1:
        distance = np.sqrt(np.sum((point1-point2)**2))
    elif len(point1.shape) == 2 or len(point2.shape) == 2:
        distance = np.sqrt(np.sum((point1-point2)**2, axis=1))
    return distance


def closest_point_to_a_ray(point, ray):
    """
    Definition to calculate the point on a ray that is closest to given point.

    Parameters
    ----------
    point         : list
                    Given point in X,Y,Z.
    ray           : ndarray
                    Given ray.

    Returns
    ---------
    closest_point : ndarray
                    Calculated closest point.
    """
    from odak.raytracing import propagate_a_ray
    if len(ray.shape) == 2:
        ray = ray.reshape((1, 2, 3))
    p0 = ray[:, 0]
    p1 = propagate_a_ray(ray, 1.)
    if len(p1.shape) == 2:
        p1 = p1.reshape((1, 2, 3))
    p1 = p1[:, 0]
    p1 = p1.reshape(3)
    p0 = p0.reshape(3)
    point = point.reshape(3)
    closest_distance = -np.dot((p0-point), (p1-p0))/np.sum((p1-p0)**2)
    closest_point = propagate_a_ray(ray, closest_distance)[0]
    return closest_point


def point_to_ray_distance(point, ray_point_0, ray_point_1):
    """
    Definition to find point's closest distance to a line represented with two points.

    Parameters
    ----------
    point       : ndarray
                  Point to be tested.
    ray_point_0 : ndarray
                  First point to represent a line.
    ray_point_1 : ndarray
                  Second point to represent a line.

    Returns
    ----------
    distance    : float
                  Calculated distance.
    """
    distance = np.sum(np.cross((point-ray_point_0), (point-ray_point_1))
                      ** 2)/np.sum((ray_point_1-ray_point_0)**2)
    return distance
