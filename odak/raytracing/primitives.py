from odak import np
from odak.tools.transformation import rotate_point
from odak.tools.vector import same_side

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

def is_it_on_triangle(pointtocheck,point0,point1,point2):
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
    side0        = same_side(pointtocheck,point0,point1,point2)
    side1        = same_side(pointtocheck,point1,point0,point2)
    side2        = same_side(pointtocheck,point2,point0,point1)
    if side0 == True and side1 == True and side2 == True:
        return True
    return False
