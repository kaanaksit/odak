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
