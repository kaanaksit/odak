import math
import torch


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
    angle = torch.deg2rad(torch.tensor(angle))
    rotx = torch.tensor([
        [1.,               0.,               0.],
        [0.,  math.cos(angle), -math.sin(angle)],
        [0.,  math.sin(angle),  math.cos(angle)]
    ])
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
    angle = torch.deg2rad(torch.tensor(angle))
    roty = torch.tensor([
        [math.cos(angle),  0., math.sin(angle)],
        [0.,               1.,              0.],
        [-math.sin(angle), 0., math.cos(angle)]
    ])
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
    angle = torch.deg2rad(torch.tensor(angle))
    rotz = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.],
        [math.sin(angle),  math.cos(angle), 0.],
        [0.,               0., 1.]
    ])
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
    rotx = rotmatx(angles[0])
    roty = rotmaty(angles[1])
    rotz = rotmatz(angles[2])
    if angles[0] == 0 and angles[1] == 0 and angles[2] == 0:
        result = torch.tensor(offset).to(point.device) + point
        return result, rotx, roty, rotz
    point -= torch.tensor(origin)
    point = point.view(1, 3)
    if mode == 'XYZ':
        result = torch.mm(rotz, torch.mm(roty, torch.mm(rotx, point.T))).T
    elif mode == 'XZY':
        result = torch.mm(roty, torch.mm(rotz, torch.mm(rotx, point.T))).T
    elif mode == 'YXZ':
        result = torch.mm(rotz, torch.mm(rotx, torch.mm(roty, point.T))).T
    elif mode == 'ZXY':
        result = torch.mm(roty, torch.mm(rotx, torch.mm(rotz, point.T))).T
    elif mode == 'ZYX':
        result = torch.mm(rotx, torch.mm(roty, torch.mm(rotz, point.T))).T
    point = point.view(3)
    result += torch.tensor(origin)
    result += torch.tensor(offset)
    return result.to(point.device), rotx, roty, rotz


def get_rotation_matrix(tilt_angles=[0., 0., 0.], tilt_order='XYZ'):
    """
    Function to generate rotation matrix for given tilt angles and tilt order.


    Parameters
    ----------
    tilt_angles        : list
                         Tilt angles in degrees along XYZ axes.
    tilt_order         : str
                         Rotation order (e.g., XYZ, XZY, ZXY, YXZ, ZYX).

    Returns
    -------
    rotmat             : torch.tensor
                         Rotation matrix.
    """
    rotx = rotmatx(tilt_angles[0])
    roty = rotmaty(tilt_angles[1])
    rotz = rotmatz(tilt_angles[2])
    if tilt_order =='XYZ':
        rotmat = torch.mm(rotz,torch.mm(roty, rotx))
    elif tilt_order == 'XZY':
        rotmat = torch.mm(roty,torch.mm(rotz, rotx))
    elif tilt_order == 'ZXY':
        rotmat = torch.mm(roty,torch.mm(rotx, rotz))
    elif tilt_order == 'YXZ':
        rotmat = torch.mm(rotz,torch.mm(rotx, roty))
    elif tilt_order == 'ZYX':
         rotmat = torch.mm(rotx,torch.mm(roty, rotz))
    return rotmat


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
    rotx         : torch.tensor
                   Rotation matrix at X axis.
    roty         : torch.tensor
                   Rotation matrix at Y axis.
    rotz         : torch.tensor
                   Rotation matrix at Z axis.
    """
    rotx = rotmatx(angles[0])
    roty = rotmaty(angles[1])
    rotz = rotmatz(angles[2])
    if angles[0] == 0 and angles[1] == 0 and angles[2] == 0:
        result = torch.tensor(offset) + points
        return result, rotx, roty, rotz
    points -= torch.tensor(origin)
    if mode == 'XYZ':
        result = torch.mm(rotz, torch.mm(roty, torch.mm(rotx, points.T))).T
    elif mode == 'XZY':
        result = torch.mm(roty, torch.mm(rotz, torch.mm(rotx, points.T))).T
    elif mode == 'YXZ':
        result = torch.mm(rotz, torch.mm(rotx, torch.mm(roty, points.T))).T
    elif mode == 'ZXY':
        result = torch.mm(roty, torch.mm(rotx, torch.mm(rotz, points.T))).T
    elif mode == 'ZYX':
        result = torch.mm(rotx, torch.mm(roty, torch.mm(rotz, points.T))).T
    result += torch.tensor(origin)
    result += torch.tensor(offset)
    return result, rotx, roty, rotz


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
    dist = torch.sqrt(torch.tensor(dx**2+dy**2+dz**2))
    phi = torch.atan2(torch.tensor(dy), torch.tensor(dx))
    theta = torch.arccos(dz/dist)
    angles = [
        0,
        float(torch.rad2deg(theta)),
        float(torch.rad2deg(phi))
    ]
    return angles
