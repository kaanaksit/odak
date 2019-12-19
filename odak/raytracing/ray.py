from odak import np

def create_ray(x0y0z0,abg):
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
    x0,y0,z0         = x0y0z0
    alpha,beta,gamma = abg
    # Create a vector with the given points and angles in each direction
    point            = np.array([x0,y0,z0],dtype=np.float)
    alpha            = np.cos(np.radians(alpha))
    beta             = np.cos(np.radians(beta))
    gamma            = np.cos(np.radians(gamma))
    # Cosines vector.
    cosines          = np.array([alpha,beta,gamma],dtype=np.float)
    ray              = np.array([point,cosines],dtype=np.float)
    return ray

def create_ray_from_two_points(x0y0z0,x1y1z1):
    """
    Definition to create a ray from two given points.

    Parameters
    ----------
    x0y0z0       : list
                   List that contains X,Y and Z start locations of a ray.
    x1y1z1       : list
                   List that contains X,Y and Z ending locations of a ray.

    Returns
    ----------
    ray          : ndarray
                   Array that contains starting points and cosines of a created ray.
    """
    # Because of Python 2 -> Python 3.
    x0,y0,z0  = x0y0z0
    x1,y1,z1  = x1y1z1
    x0 = float(x0); y0 = float(y0); z0 = float(z0)
    x1 = float(x1); y1 = float(y1); z1 = float(z1)
    # Create a vector from two given points.
    point     = np.array([x0,y0,z0],dtype=np.float)
    # Distance between two points.
    s         = np.sqrt( (x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2 )
    if s != 0:
        alpha = (x1-x0)/s
        beta  = (y1-y0)/s
        gamma = (z1-z0)/s
    elif s == 0:
        alpha = float('nan')
        beta  = float('nan')
        gamma = float('nan')
    # Cosines vector
    cosines   = np.array([alpha,beta,gamma],dtype=np.float)
    # Returns vector and the distance.
    ray       = np.array([point,cosines],dtype=np.float)
    return ray

