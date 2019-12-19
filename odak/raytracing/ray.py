from odak import np
from odak.tools.transformation import rotate_point

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

def multiply_two_vectors(vector1,vector2):
    """
    Definition to Multiply two vectors and return the resultant vector. Used method described under: http://en.wikipedia.org/wiki/Cross_product

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
    ray    = np.array([vector1[0],angle],dtype=np.float)
    return ray

def find_intersection_w_surface(ray,points):
    """
    Definition to find intersection point inbetween a surface and a ray. For more see: http://geomalgorithms.com/a06-_intersect-2.html

    Parameters
    ----------
    vector       : ndarray
                   A vector/ray.
    points       : ndarray
                   Set of points in X,Y and Z to define a planar surface.

    Returns
    ----------
    normal       : ndarray
                   Surface normal at the point of intersection.
    distance     : float
                   Distance in between starting point of a ray with it's intersection with a planar surface.
    """
    point0,point1,point2 = points
    vector0              = create_ray_from_two_points(point0,point1)
    vector1              = create_ray_from_two_points(point1,point2)
    vector2              = create_ray_from_two_points(point0,point2)
    normal               = multiply_two_vectors(vector0,vector2)
    f                    = point0-ray[0]
    n                    = normal[1].copy()
    distance             = np.dot(n,f)/np.dot(n,ray[1])
    normal[0][0]         = ray[0][0]+distance*ray[1][0]
    normal[0][1]         = ray[0][1]+distance*ray[1][1]
    normal[0][2]         = ray[0][2]+distance*ray[1][2]
    return normal,distance

def reflect(input_ray,normal):
    """
    Definition to reflect an incoming ray from a surface defined by a surface normal. Used method described in G.H. Spencer and M.V.R.K. Murty, "General Ray-Tracing Procedure", 1961.

    Parameters
    ----------
    input_ray    : ndarray
                   A vector/ray.
    normal       : ndarray
                   A surface normal.

    Returns
    ----------
    output_ray   : ndarray
                   Array that contains starting points and cosines of a reflected ray.
    """
    mu               = 1
    div              = pow(normal[1,0],2)  + pow(normal[1,1],2) + pow(normal[1,2],2)
    a                = mu* ( input_ray[1][0]*normal[1][0]
                           + input_ray[1][1]*normal[1][1]
                           + input_ray[1][2]*normal[1][2]) / div
    output_ray       = input_ray.copy()
    output_ray[0][0] = normal[0][0]
    output_ray[0][1] = normal[0][1]
    output_ray[0][2] = normal[0][2]
    output_ray[1][0] = input_ray[1][0] - 2*a*normal[1][0]
    output_ray[1][1] = input_ray[1][1] - 2*a*normal[1][1]
    output_ray[1][2] = input_ray[1][2] - 2*a*normal[1][2]
    return output_ray

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
                      [10., 10., 0.],
                      [ 0., 10., 0.],
                      [ 0.,  0., 0.]
                     ])
    for i in range(0,plane.shape[0]):
        plane[i],_,_,_  = rotate_point(plane[i],angles=angles)
        plane[i]        = plane[i]+point
    return plane
