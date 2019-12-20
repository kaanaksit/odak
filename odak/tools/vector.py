from odak import np

def cross_product(vector1,vector2):
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
    angle  = np.cross(vector1[1].T,vector2[1].T)
    angle  = np.asarray(angle)
    angle /= np.amax(angle)
    ray    = np.array([vector1[0],angle],dtype=np.float)
    return ray

def same_side(p1,p2,a,b):
    """
    Definition to figure which side a point is on with respect to a line and a point. See http://www.blackpawn.com/texts/pointinpoly/ for more. If p1 and p2 are on the sameside, this definition returns True.

    Parameters
    ----------
    p1          : list
                  Point to check.
    p2          : list
                  This is the point check against.
    a           : list
                  First point that forms the line.
    b           : list
                  Second point that forms the line.
    """
    ba    = np.subtract(b,a)
    p1a   = np.subtract(p1,a)
    p2a   = np.subtract(p2,a)
    cp1   = np.cross(ba,p1a)
    cp2   = np.cross(ba,p2a)
    if np.dot(cp1,cp2) >= 0:
        return True
    return False

def  distance_between_two_points(point1,point2):
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
    distance = np.sqrt(np.sum((np.array(point1)-np.array(point2))**2))
    return distance
