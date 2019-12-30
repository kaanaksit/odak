from odak import np
from odak.tools.vector import cross_product,distance_between_two_points
from odak.raytracing.ray import create_ray_from_two_points
from odak.raytracing.primitives import is_it_on_triangle,center_of_triangle

def reflect(input_ray,normal):
    """ 
    Definition to reflect an incoming ray from a surface defined by a surface normal. Used method described in G.H. Spencer and M.V.R.K. Murty, "General Ray-Tracing Procedure", 1961.

    Parameters
    ----------
    input_ray    : ndarray
                   A vector/ray (2x3). It can also be a list of rays (nx2x3).
    normal       : ndarray
                   A surface normal (2x3). It also be a list of normals (nx2x3).

    Returns
    ----------
    output_ray   : ndarray
                   Array that contains starting points and cosines of a reflected ray.
    """
    if len(input_ray.shape) == 2:
        input_ray = input_ray.reshape((1,2,3))
    if len(normal.shape) == 2:
        normal    = normal.reshape((1,2,3))
    mu              = 1
    div             = normal[:,1,0]**2  + normal[:,1,1]**2 + normal[:,1,2]**2
    a               = mu* ( input_ray[:,1,0]*normal[:,1,0]
                          + input_ray[:,1,1]*normal[:,1,1]
                          + input_ray[:,1,2]*normal[:,1,2]) / div 
    n               = np.amax([normal.shape[0],input_ray.shape[0]])
    output_ray      = np.zeros((n,2,3))
    output_ray[:,0] = normal[:,0]
    output_ray[:,1] = input_ray[:,1]-2*normal[:,1]
    if output_ray.shape[0] == 1:
       output_ray = output_ray.reshape((2,3))
    return output_ray

def intersect_w_surface(ray,points):
    """
    Definition to find intersection point inbetween a surface and a ray. For more see: http://geomalgorithms.com/a06-_intersect-2.html

    Parameters
    ----------
    ray          : ndarray
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
    normal               = get_triangle_normal(points)
    if len(ray.shape) == 2:
        ray = ray.reshape((1,2,3))
    if len(points) == 2:
        points = points.reshape((1,3,3))
    if len(normal.shape) == 2:
        normal = normal.reshape((1,2,3))
    f                    = points[:,0]-ray[:,0]
    distance             = np.dot(normal[:,1],f.T)/np.dot(normal[:,1],ray[:,1].T)
    n                    = np.amax([ray.shape[0],normal.shape[0]])
    normal               = np.zeros((n,2,3))
    normal[:,0]          = ray[:,0]+distance[:,0]*ray[:,1]
    distance             = np.abs(distance)
    if normal.shape[0] == 1:
        normal   = normal.reshape((2,3))
        distance = distance.reshape((1))
    if distance.shape[0] == 1:
        distance = distance.reshape((distance.shape[1]))
    return normal,distance

def get_triangle_normal(triangle,triangle_center=None):
    """
    Definition to calculate surface normal of a triangle.

    Parameters
    ----------
    triangle        : ndarray
                      Set of points in X,Y and Z to define a planar surface (3,3). It can also be list of triangles (mx3x3).
    triangle_center : ndarray
                      Center point of the given triangle. See odak.raytracing.center_of_triangle for more. In many scenarios you can accelerate things by precomputing triangle centers.

    Returns
    ----------
    normal          : ndarray
                      Surface normal at the point of intersection.
    """
    triangle  = np.asarray(triangle)
    if len(triangle.shape) == 2:
        triangle  = triangle.reshape((1,3,3))
    normal    = np.zeros((triangle.shape[0],2,3))
    direction = np.cross(triangle[:,0]-triangle[:,1],triangle[:,2]-triangle[:,1])
    if type(triangle_center) == type(None):
        normal[:,0] = center_of_triangle(triangle)
    else:
        normal[:,0] = triangle_center
    normal[:,1] = direction/np.sum(direction,axis=1)[0]
    if normal.shape[0] == 1:
        normal = normal.reshape((2,3))
    return normal

def intersect_w_circle(ray,circle):
    """
    Definition to find intersection point of a ray with a circle. Returns False for each variable if the ray doesn't intersect with a given circle. Returns distance as zero if there isn't an intersection.

    Parameters
    ----------
    ray          : ndarray
                   A vector/ray.
    circle       : list
                   A list that contains (0) Set of points in X,Y and Z to define plane of a circle, (1) circle center, and (2) circle radius.

    Returns
    ----------
    normal       : ndarray
                   Surface normal at the point of intersection.
    distance     : float
                   Distance in between a starting point of a ray and the intersection point with a given triangle.
    """
    normal,distance    = intersect_w_surface(ray,circle[0])
    if len(normal.shape) == 2:
        normal = normal.reshape((1,2,3))
    distance_to_center                     = distance_between_two_points(normal[:,0],circle[1])
    distance[np.nonzero(distance_to_center>circle[2])] = 0
    return normal,distance

def intersect_w_triangle(ray,triangle):
    """
    Definition to find intersection point of a ray with a triangle. Returns False for each variable if the ray doesn't intersect with a given triangle.

    Parameters
    ----------
    ray          : ndarray
                   A vector/ray.
    triangle     : ndarray
                   Set of points in X,Y and Z to define a single triangle.

    Returns
    ----------
    normal       : ndarray
                   Surface normal at the point of intersection.
    distance     : float
                   Distance in between a starting point of a ray and the intersection point with a given triangle.
    """
    normal,distance = intersect_w_surface(ray,triangle)
    if is_it_on_triangle(normal[0],triangle[0],triangle[1],triangle[2]) == False:
        return False,False
    return normal,distance
