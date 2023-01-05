import math
import numpy as np


def rotmatx(angle):
    """
    Definition to generate a rotation matrix along X axis.

    Parameters
    ----------
    angles       : list
                   Rotation angles in degrees.

    Returns
    ----------
    rotx         : ndarray
                    Rotation matrix along X axis.
    """
    angle = np.float64(angle)
    angle = np.radians(angle)
    rotx = np.array([
        [1.,               0.,               0.],
        [0.,  math.cos(angle), -math.sin(angle)],
        [0.,  math.sin(angle),  math.cos(angle)]
    ], dtype=np.float64)
    return rotx


def rotmaty(angle):
    """
    Definition to generate a rotation matrix along Y axis.

    Parameters
    ----------
    angles       : list
                   Rotation angles in degrees.

    Returns
    ----------
    roty         : ndarray
                   Rotation matrix along Y axis.
    """
    angle = np.radians(angle)
    roty = np.array([
        [math.cos(angle),  0., math.sin(angle)],
        [0.,               1.,              0.],
        [-math.sin(angle), 0., math.cos(angle)]
    ], dtype=np.float64)
    return roty


def rotmatz(angle):
    """
    Definition to generate a rotation matrix along Z axis.

    Parameters
    ----------
    angles       : list
                   Rotation angles in degrees.

    Returns
    ----------
    rotz         : ndarray
                   Rotation matrix along Z axis.
    """
    angle = np.radians(angle)
    rotz = np.array([
        [math.cos(angle), -math.sin(angle), 0.],
        [math.sin(angle),  math.cos(angle), 0.],
        [0.,               0., 1.]
    ], dtype=np.float64)

    return rotz


def rotate_point(point, angles=[0, 0, 0], mode='XYZ', origin=[0, 0, 0], offset=[0, 0, 0]):
    """
    Definition to rotate a given point. Note that rotation is always with respect to 0,0,0.

    Parameters
    ----------
    point        : ndarray
                   A point.
    angles       : list
                   Rotation angles in degrees. 
    mode         : str
                   Rotation mode determines ordering of the rotations at each axis. There are XYZ,YXZ,ZXY and ZYX modes.
    origin       : list
                   Reference point for a rotation.
    offset       : list
                   Shift with the given offset.

    Returns
    ----------
    result       : ndarray
                   Result of the rotation
    rotx         : ndarray
                   Rotation matrix along X axis.
    roty         : ndarray
                   Rotation matrix along Y axis.
    rotz         : ndarray
                   Rotation matrix along Z axis.
    """
    point = np.asarray(point)
    point -= np.asarray(origin)
    rotx = rotmatx(angles[0])
    roty = rotmaty(angles[1])
    rotz = rotmatz(angles[2])
    if mode == 'XYZ':
        result = np.dot(rotz, np.dot(roty, np.dot(rotx, point)))
    elif mode == 'XZY':
        result = np.dot(roty, np.dot(rotz, np.dot(rotx, point)))
    elif mode == 'YXZ':
        result = np.dot(rotz, np.dot(rotx, np.dot(roty, point)))
    elif mode == 'ZXY':
        result = np.dot(roty, np.dot(rotx, np.dot(rotz, point)))
    elif mode == 'ZYX':
        result = np.dot(rotx, np.dot(roty, np.dot(rotz, point)))
    result += np.asarray(origin)
    result += np.asarray(offset)
    return result, rotx, roty, rotz


def rotate_points(points, angles=[0, 0, 0], mode='XYZ', origin=[0, 0, 0], offset=[0, 0, 0]):
    """
    Definition to rotate points.

    Parameters
    ----------
    points       : ndarray
                   Points.
    angles       : list
                   Rotation angles in degrees. 
    mode         : str
                   Rotation mode determines ordering of the rotations at each axis. There are XYZ,YXZ,ZXY and ZYX modes.
    origin       : list
                   Reference point for a rotation.
    offset       : list
                   Shift with the given offset.

    Returns
    ----------
    result       : ndarray
                   Result of the rotation   
    """
    points = np.asarray(points)
    if angles[0] == 0 and angles[1] == 0 and angles[2] == 0:
        result = np.array(offset) + points
        return result
    points -= np.array(origin)
    rotx = rotmatx(angles[0])
    roty = rotmaty(angles[1])
    rotz = rotmatz(angles[2])
    if mode == 'XYZ':
        result = np.dot(rotz, np.dot(roty, np.dot(rotx, points.T))).T
    elif mode == 'XZY':
        result = np.dot(roty, np.dot(rotz, np.dot(rotx, points.T))).T
    elif mode == 'YXZ':
        result = np.dot(rotz, np.dot(rotx, np.dot(roty, points.T))).T
    elif mode == 'ZXY':
        result = np.dot(roty, np.dot(rotx, np.dot(rotz, points.T))).T
    elif mode == 'ZYX':
        result = np.dot(rotx, np.dot(roty, np.dot(rotz, points.T))).T
    result += np.array(origin)
    result += np.array(offset)
    return result


def tilt_towards(location, lookat):
    """
    Definition to tilt surface normal of a plane towards a point.

    Parameters
    ----------
    location     : list
                   Center of the plane to be tilted.
    lookat       : list
                   Tilt towards this point.

    Returns
    ----------
    angles       : list
                   Rotation angles in degrees.
    """
    dx = location[0]-lookat[0]
    dy = location[1]-lookat[1]
    dz = location[2]-lookat[2]
    dist = np.sqrt(dx**2+dy**2+dz**2)
    phi = np.arctan2(dy, dx)
    theta = np.arccos(dz/dist)
    angles = [
        0,
        np.degrees(theta).tolist(),
        np.degrees(phi).tolist()
    ]
    return angles
