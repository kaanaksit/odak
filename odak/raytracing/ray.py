from odak import np
from odak.tools import rotate_points

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
                   List that contains X,Y and Z start locations of a ray (3). It can also be a list of points as well (mx3). This is the starting point.
    x1y1z1       : list
                   List that contains X,Y and Z ending locations of a ray (3). It can also be a list of points as well (mx3). This is the end point.

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
    xdiff        = x1y1z1[:,0]-x0y0z0[:,0]
    ydiff        = x1y1z1[:,1]-x0y0z0[:,1]
    zdiff        = x1y1z1[:,2]-x0y0z0[:,2]
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

def create_ray_from_angles(point,angles,mode='XYZ'):
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
        point  = point.reshape((1,3))
    new_point       = np.zeros(point.shape)
    new_point[:,2] += 5.
    new_point       = rotate_points(new_point,angles,mode=mode,offset=point[:,0])
    ray             = create_ray_from_two_points(point,new_point)
    if ray.shape[0] == 1:
        ray = ray.reshape((2,3))
    return ray

def propagate_a_ray(ray,distance):
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
        ray = ray.reshape((1,2,3))
    new_ray        = np.copy(ray)
    new_ray[:,0,0] = distance*new_ray[:,1,0] + new_ray[:,0,0]
    new_ray[:,0,1] = distance*new_ray[:,1,1] + new_ray[:,0,1]
    new_ray[:,0,2] = distance*new_ray[:,1,2] + new_ray[:,0,2]
    if new_ray.shape[0] == 1:
        new_ray = new_ray.reshape((2,3))
    return new_ray

def findNearestPoints(vec1, vec2, ray):
  # written by praneeth chakravarthula
    # Refer to the concept of skew lines and line-plane intersection for the following math.
    p1 = vec1[0].reshape(3,)
    d1 = vec1[1].reshape(3,)
    p2 = vec2[0].reshape(3,)
    d2 = vec2[1].reshape(3,)
    # normal to both vectors
    n = np.cross(d1, d2)
    # if the rays intersect
    if np.all(n)==0:
      point, distances = ray.CalculateIntersectionOfTwoVectors(vec1, vec2)
      c1 = c2 = point
    else:
      # normal to plane formed by vectors n and d1
      n1 = np.cross(d1, n)
      # normal to plane formed by vectors n and d2
      n2 = np.cross(d2, n)
      # nearest distance point to vec2 along vec1 is equal to
      # intersection of vec1 with plane formed by vec2 and normal n
      c1 = p1 + (np.dot((p2-p1), n2)/np.dot(d1, n2))*d1
      # nearest distance point to vec1 along vec2 is equal to
      # intersection of vec2 with plane formed by vec1 and normal n
      c2 = p2 + (np.dot((p1-p2), n1)/np.dot(d2, n1))*d2
    return c1, c2
