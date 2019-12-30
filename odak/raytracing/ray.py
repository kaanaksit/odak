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
    Definition to create a ray from two given points. Note that both inputs must match in shape.

    Parameters
    ----------
    x0y0z0       : list
                   List that contains X,Y and Z start locations of a ray (3). It can also be a list of points as well (mx3).
    x1y1z1       : list
                   List that contains X,Y and Z ending locations of a ray (3). It can also be a list of points as well (mx3).

    Returns
    ----------
    ray          : ndarray
                   Array that contains starting points and cosines of a created ray.
    """
    x0y0z0 = np.asarray(x0y0z0,dtype=np.float)
    x1y1z1 = np.asarray(x1y1z1,dtype=np.float)
    if len(x0y0z0.shape) == 1:
        x0y0z0 = x0y0z0.reshape((1,3))
    if len(x1y1z1.shape) == 1:
        x1y1z1 = x1y1z1.reshape((1,3))
    xdiff        = x0y0z0[:,0]-x1y1z1[:,0]
    ydiff        = x0y0z0[:,1]-x1y1z1[:,1]
    zdiff        = x0y0z0[:,2]-x1y1z1[:,2]
    s            = np.sqrt(xdiff**2+ydiff**2+zdiff**2)
    s[s==0]      = np.NaN
    cosines      = np.zeros((xdiff.shape[0],3))
    cosines[:,0] = xdiff/s 
    cosines[:,1] = ydiff/s
    cosines[:,2] = zdiff/s
    ray          = np.zeros((xdiff.shape[0],2,3),dtype=np.float)
    ray[:,0]     = x0y0z0
    ray[:,1]     = cosines
    if ray.shape[0] == 1:
        ray = ray.reshape((2,3))
    return ray

