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
    ray          : ndarray
                   Array that contains starting points and cosines of a created ray.
    """
    # Due to Python 2 -> Python 3.
    x0, y0, z0 = x0y0z0
    alpha, beta, gamma = abg
    # Create a vector with the given points and angles in each direction
    point = np.array([x0, y0, z0], dtype=np.float64)
    alpha = np.cos(np.radians(alpha))
    beta = np.cos(np.radians(beta))
    gamma = np.cos(np.radians(gamma))
    # Cosines vector.
    cosines = np.array([alpha, beta, gamma], dtype=np.float64)
    ray = np.array([point, cosines], dtype=np.float64)
    return ray


def create_ray_from_two_points(x0y0z0, x1y1z1):
    """
    Definition to create a ray from two given points. Note that both inputs must match in shape.

    Parameters
    ----------
    x0y0z0       : list
                   List that contains X,Y and Z start locations of a ray (3). It can also be a list of points as well (mx3). This is the starting point.
    x1y1z1       : list
                   List that contains X,Y and Z ending locations of a ray (3). It can also be a list of points as well (mx3). This is the end point.

    Returns
    ----------
    ray          : ndarray
                   Array that contains starting points and cosines of a created ray.
    """
    x0y0z0 = np.asarray(x0y0z0, dtype=np.float64)
    x1y1z1 = np.asarray(x1y1z1, dtype=np.float64)
    if len(x0y0z0.shape) == 1:
        x0y0z0 = x0y0z0.reshape((1, 3))
    if len(x1y1z1.shape) == 1:
        x1y1z1 = x1y1z1.reshape((1, 3))
    xdiff = x1y1z1[:, 0] - x0y0z0[:, 0]
    ydiff = x1y1z1[:, 1] - x0y0z0[:, 1]
    zdiff = x1y1z1[:, 2] - x0y0z0[:, 2]
    s = np.sqrt(xdiff ** 2 + ydiff ** 2 + zdiff ** 2)
    s[s == 0] = np.nan
    cosines = np.zeros((xdiff.shape[0], 3))
    cosines[:, 0] = xdiff/s
    cosines[:, 1] = ydiff/s
    cosines[:, 2] = zdiff/s
    ray = np.zeros((xdiff.shape[0], 2, 3), dtype=np.float64)
    ray[:, 0] = x0y0z0
    ray[:, 1] = cosines
    if ray.shape[0] == 1:
        ray = ray.reshape((2, 3))
    return ray


def create_ray_from_angles(point, angles, mode='XYZ'):
    """
    Definition to create a ray from a point and angles.

    Parameters
    ----------
    point      : ndarray
                 Point in X,Y and Z.
    angles     : ndarray
                 Angles with X,Y,Z axes in degrees. All zeros point Z axis.
    mode       : str
                 Rotation mode determines ordering of the rotations at each axis. There are XYZ,YXZ    ,ZXY and ZYX modes.

    Returns
    ----------
    ray        : ndarray
                 Created ray.
    """
    if len(point.shape) == 1:
        point = point.reshape((1, 3))
    new_point = np.zeros(point.shape)
    new_point[:, 2] += 5.
    new_point = rotate_points(new_point, angles, mode=mode, offset=point[:, 0])
    ray = create_ray_from_two_points(point, new_point)
    if ray.shape[0] == 1:
        ray = ray.reshape((2, 3))
    return ray


def propagate_a_ray(ray, distance):
    """
    Definition to propagate a ray at a certain given distance.

    Parameters
    ----------
    ray        : ndarray
                 A ray.
    distance   : float
                 Distance.

    Returns
    ----------
    new_ray    : ndarray
                 Propagated ray.
    """
    if len(ray.shape) == 2:
        ray = ray.reshape((1, 2, 3))
    new_ray = np.copy(ray)
    new_ray[:, 0, 0] = distance*new_ray[:, 1, 0] + new_ray[:, 0, 0]
    new_ray[:, 0, 1] = distance*new_ray[:, 1, 1] + new_ray[:, 0, 1]
    new_ray[:, 0, 2] = distance*new_ray[:, 1, 2] + new_ray[:, 0, 2]
    if new_ray.shape[0] == 1:
        new_ray = new_ray.reshape((2, 3))
    return new_ray


def calculate_intersection_of_two_rays(ray0, ray1):
    """
    Definition to calculate the intersection of two rays.

    Parameters
    ----------
    ray0       : ndarray
                 A ray.
    ray1       : ndarray
                 A ray.

    Returns
    ----------
    point      : ndarray
                 Point in X,Y,Z.
    distances  : ndarray
                 Distances.
    """
    A = np.array([
        [float(ray0[1][0]), float(ray1[1][0])],
        [float(ray0[1][1]), float(ray1[1][1])],
        [float(ray0[1][2]), float(ray1[1][2])]
    ])
    B = np.array([
        ray0[0][0]-ray1[0][0],
        ray0[0][1]-ray1[0][1],
        ray0[0][2]-ray1[0][2]
    ])
    distances = np.linalg.lstsq(A, B, rcond=None)[0]
    if np.allclose(np.dot(A, distances), B) == False:
        distances = np.array([0, 0])
    distances = distances[np.argsort(-distances)]
    point = propagate_a_ray(ray0, distances[0])[0]
    return point, distances


def find_nearest_points(ray0, ray1):
    """
    Find the nearest points on given rays with respect to the other ray.

    Parameters
    ----------
    ray0       : ndarray
                 A ray.
    ray1       : ndarray
                 A ray.

    Returns
    ----------
    c0         : ndarray
                 Closest point on ray0.
    c1         : ndarray
                 Closest point on ray1.
    """
    p0 = ray0[0].reshape(3,)
    d0 = ray0[1].reshape(3,)
    p1 = ray1[0].reshape(3,)
    d1 = ray1[1].reshape(3,)
    n = np.cross(d0, d1)
    if np.all(n) == 0:
        point, distances = calculate_intersection_of_two_rays(ray0, ray1)
        c0 = c1 = point
    else:
        n0 = np.cross(d0, n)
        n1 = np.cross(d1, n)
        c0 = p0+(np.dot((p1-p0), n1)/np.dot(d0, n1))*d0
        c1 = p1+(np.dot((p0-p1), n0)/np.dot(d1, n0))*d1
    return c0, c1
