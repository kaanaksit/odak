from odak import np
from odak.tools.transformation import rotate_point

def define_plane(point,angles=[0.,0.,0.]):
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
                      [ 10., 10., 0.],
                      [  0., 10., 0.],
                      [  0.,  0., 0.]
                     ])
    for i in range(0,plane.shape[0]):
        plane[i],_,_,_  = rotate_point(plane[i],angles=angles)
        plane[i]        = plane[i]+point
    return plane
