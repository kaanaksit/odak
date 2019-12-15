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
    angle = np.float(angle)
    angle = angle_to_radians(angle)
    rotx  = np.array([
                      [1.,            0.  ,           0.],
                      [0.,  math.cos(angle), -math.sin(angle)],
                      [0.,  math.sin(angle), math.cos(angle)]
                     ],dtype=np.float)
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
    angle = angle_to_radians(angle)
    roty  = np.array([
                      [math.cos(angle),  0., math.sin(angle)],
                      [0.,             1.,            0.],
                      [-math.sin(angle), 0., math.cos(angle)]
                     ],dtype=np.float)
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
    angle = angle_to_radians(angle)
    rotz  = np.array([
                      [ math.cos(angle), -math.sin(angle), 0.],
                      [ math.sin(angle),  math.cos(angle), 0.],
                      [            0.,            0., 1.]
                     ],dtype=np.float)
    return rotz

def rotate_point(point,angles=[0,0,0],mode='XYZ'):
    """
    Definition to rotate a given point.

    Parameters
    ----------
    point        : ndarray
                   A point.
    angles       : list
                   Rotation angles in degrees.
    mode         : str
                   Rotation mode determines ordering of the rotations at each axis. There are XYZ,YXZ,ZXY and ZYX modes.

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
    rotx   = rotmatx(angles[0])
    roty   = rotmaty(angles[1])
    rotz   = rotmatz(angles[2])
    if mode == 'XYZ':
        result = np.dot(rotz,np.dot(roty,np.dot(rotx,point)))
    elif mode == 'XZY':
        result = np.dot(roty,np.dot(rotz,np.dot(rotx,point)))
    elif mode == 'YXZ':
        result = np.dot(rotz,np.dot(rotx,np.dot(roty,point)))
    elif mode == 'ZXY':
        result = np.dot(roty,np.dot(rotx,np.dot(rotz,point)))
    elif mode == 'ZYX':
        result = np.dot(rotx,np.dot(roty,np.dot(rotz,point)))
    return result,rotx,roty,rotz
