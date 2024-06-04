import numpy as np
from ..tools.vector import distance_between_two_points, closest_point_to_a_ray
from ..raytracing.primitives import is_it_on_triangle, center_of_triangle, sphere_function, cylinder_function
from ..raytracing.ray import create_ray_from_two_points, propagate_a_ray


def reflect(input_ray, normal):
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
    input_ray = np.asarray(input_ray)
    normal = np.asarray(normal)
    if len(input_ray.shape) == 2:
        input_ray = input_ray.reshape((1, 2, 3))
    if len(normal.shape) == 2:
        normal = normal.reshape((1, 2, 3))
    mu = 1
    div = normal[:, 1, 0]**2 + normal[:, 1, 1]**2 + normal[:, 1, 2]**2
    a = mu * (input_ray[:, 1, 0]*normal[:, 1, 0]
              + input_ray[:, 1, 1]*normal[:, 1, 1]
              + input_ray[:, 1, 2]*normal[:, 1, 2]) / div
    n = np.int64(np.amax(np.array([normal.shape[0], input_ray.shape[0]])))
    output_ray = np.zeros((n, 2, 3))
    output_ray[:, 0] = normal[:, 0]
    output_ray[:, 1] = input_ray[:, 1]-2*a*normal[:, 1]
    if output_ray.shape[0] == 1:
        output_ray = output_ray.reshape((2, 3))
    return output_ray


def intersect_w_surface(ray, points):
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
    points = np.asarray(points)
    normal = get_triangle_normal(points)
    if len(ray.shape) == 2:
        ray = ray.reshape((1, 2, 3))
    if len(points) == 2:
        points = points.reshape((1, 3, 3))
    if len(normal.shape) == 2:
        normal = normal.reshape((1, 2, 3))
    f = normal[:, 0]-ray[:, 0]
    distance = np.dot(normal[:, 1], f.T)/np.dot(normal[:, 1], ray[:, 1].T)
    n = np.int64(np.amax(np.array([ray.shape[0], normal.shape[0]])))
    normal = np.zeros((n, 2, 3))
    normal[:, 0] = ray[:, 0]+distance.T*ray[:, 1]
    distance = np.abs(distance)
    if normal.shape[0] == 1:
        normal = normal.reshape((2, 3))
        distance = distance.reshape((1))
    if distance.shape[0] == 1 and len(distance.shape) > 1:
        distance = distance.reshape((distance.shape[1]))
    return normal, distance


def get_triangle_normal(triangle, triangle_center=None):
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
    triangle = np.asarray(triangle)
    if len(triangle.shape) == 2:
        triangle = triangle.reshape((1, 3, 3))
    normal = np.zeros((triangle.shape[0], 2, 3))
    direction = np.cross(
        triangle[:, 0]-triangle[:, 1], triangle[:, 2]-triangle[:, 1])
    if type(triangle_center) == type(None):
        normal[:, 0] = center_of_triangle(triangle)
    else:
        normal[:, 0] = triangle_center
    normal[:, 1] = direction/np.sum(direction, axis=1)[0]
    if normal.shape[0] == 1:
        normal = normal.reshape((2, 3))
    return normal


def intersect_w_circle(ray, circle):
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
    normal, distance = intersect_w_surface(ray, circle[0])
    if len(normal.shape) == 2:
        normal = normal.reshape((1, 2, 3))
    distance_to_center = distance_between_two_points(normal[:, 0], circle[1])
    distance[np.nonzero(distance_to_center > circle[2])] = 0
    if len(ray.shape) == 2:
        normal = normal.reshape((2, 3))
    return normal, distance


def intersect_w_triangle(ray, triangle):
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
    normal, distance = intersect_w_surface(ray, triangle)
    if is_it_on_triangle(normal[0], triangle[0], triangle[1], triangle[2]) == False:
        return 0, 0
    return normal, distance


def get_sphere_normal(point, sphere):
    """
    Definition to get a normal of a point on a given sphere.

    Parameters
    ----------
    point         : ndarray
                    Point on sphere in X,Y,Z.
    sphere        : ndarray
                    Center defined in X,Y,Z and radius.

    Returns
    ----------
    normal_vector : ndarray
                    Normal vector.
    """
    if len(point.shape) == 1:
        point = point.reshape((1, 3))
    normal_vector = create_ray_from_two_points(point, sphere[0:3])
    return normal_vector


def get_cylinder_normal(point, cylinder):
    """
    Parameters
    ----------
    point         : ndarray
                    Point on a cylinder defined in X,Y,Z.

    Returns
    ----------
    normal_vector : ndarray
                    Normal vector.
    """
    cylinder_ray = create_ray_from_two_points(cylinder[0:3], cylinder[4:7])
    closest_point = closest_point_to_a_ray(
        point,
        cylinder_ray
    )
    normal_vector = create_ray_from_two_points(closest_point, point)
    return normal_vector


def intersection_kernel_for_parametric_surfaces(distance, ray, parametric_surface, surface_function):
    """
    Definition for the intersection kernel when dealing with parametric surfaces.

    Parameters
    ----------
    distance           : float
                         Distance.
    ray                : ndarray
                         Ray.
    parametric_surface : ndarray
                         Array that defines a parametric surface.
    surface_function   : ndarray
                         Function to evaluate a point against a parametric surface.

    Returns
    ----------
    point              : ndarray
                         Location in X,Y,Z after propagation.
    error              : float
                         Error.
    """
    new_ray = propagate_a_ray(ray, distance)
    if len(new_ray) == 2:
        new_ray = new_ray.reshape((1, 2, 3))
    point = new_ray[:, 0]
    error = surface_function(point, parametric_surface)
    return error, point


def propagate_parametric_intersection_error(distance, error):
    """
    Definition to propagate the error in parametric intersection to find the next distance to try.

    Parameters
    ----------
    distance     : list
                   List that contains the new and the old distance.
    error        : list
                   List that contains the new and the old error.

    Returns
    ----------
    distance     : list
                   New distance.
    error        : list
                   New error.
    """
    new_distance = distance[1]-error[1] * \
        (distance[1]-distance[0])/(error[1]-error[0])
    distance[0] = distance[1]
    distance[1] = np.abs(new_distance)
    error[0] = error[1]
    return distance, error


def intersect_parametric(ray, parametric_surface, surface_function, surface_normal_function, target_error=0.00000001, iter_no_limit=100000):
    """
    Definition to intersect a ray with a parametric surface.

    Parameters
    ----------
    ray                     : ndarray
                              Ray.
    parametric_surface      : ndarray
                              Parameters of the surfaces.
    surface_function        : function
                              Function to evaluate a point against a surface.
    surface_normal_function : function
                              Function to calculate surface normal for a given point on a surface.
    target_error            : float
                              Target error that defines the precision.  
    iter_no_limit           : int
                              Maximum number of iterations.

    Returns
    ----------
    distance                : float
                              Propagation distance.
    normal                  : ndarray
                              Ray that defines a surface normal for the intersection.
    """
    if len(ray.shape) == 2:
        ray = ray.reshape((1, 2, 3))
    error = [150, 100]
    distance = [0, 0.1]
    iter_no = 0
    while np.abs(np.max(np.asarray(error[1]))) > target_error:
        error[1], point = intersection_kernel_for_parametric_surfaces(
            distance[1],
            ray,
            parametric_surface,
            surface_function
        )
        distance, error = propagate_parametric_intersection_error(
            distance,
            error
        )
        iter_no += 1
        if iter_no > iter_no_limit:
            return False, False
        if np.isnan(np.sum(point)):
            return False, False
    normal = surface_normal_function(
        point,
        parametric_surface
    )
    return distance[1], normal


def intersect_w_cylinder(ray, cylinder):
    """
    Definition to intersect a ray with a cylinder.

    Parameters
    ----------
    ray        : ndarray
                 A ray definition.
    cylinder   : ndarray
                 A cylinder defined with a center in XYZ and radius of curvature.

    Returns
    ----------
    normal     : ndarray
                 A ray defining surface normal at the point of intersection.
    distance   : float
                 Total optical propagation distance.
    """
    distance, normal = intersect_parametric(
        ray,
        cylinder,
        cylinder_function,
        get_cylinder_normal
    )
    return normal, distance


def intersect_w_sphere(ray, sphere):
    """
    Definition to intersect a ray with a sphere.

    Parameters
    ----------
    ray        : ndarray
                 A ray definition.
    sphere     : ndarray
                 A sphere defined with a center in XYZ and radius of curvature.

    Returns
    ----------
    normal     : ndarray
                 A ray defining surface normal at the point of intersection.
    distance   : float
                 Total optical propagation distance.
    """
    distance, normal = intersect_parametric(
        ray,
        sphere,
        sphere_function,
        get_sphere_normal
    )
    return normal, distance
