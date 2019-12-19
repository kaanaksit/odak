from odak import np
from odak.tools.vector import cross_product
from odak.raytracing.ray import create_ray_from_two_points

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
    normal               = cross_product(vector0,vector2)
    f                    = point0-ray[0].T
    n                    = normal[1].copy()
    distance             = np.dot(n.T,f.T)/np.dot(n.T,ray[1])
    normal[0][0]         = ray[0][0]+distance*ray[1][0]
    normal[0][1]         = ray[0][1]+distance*ray[1][1]
    normal[0][2]         = ray[0][2]+distance*ray[1][2]
    return normal,distance

