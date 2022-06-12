import torch
from .primitives import is_it_on_triangle, center_of_triangle


def refract(vector, normvector, n1, n2, error=0.01):
    """
    Definition to refract an incoming ray.

    Parameters
    ----------
    vector         : torch.tensor
                     Incoming ray.
    normvector     : torch.tensor
                     Normal vector.
    n1             : float
                     Refractive index of the incoming medium.
    n2             : float
                     Refractive index of the outgoing medium.
    error          : float 
                     Desired error.

    Returns
    -------
    VectorOutput   : torch.tensor
                     Refracted ray.
    """
    mu    = n1 / n2
    div   = normvector[1,0]**2  + normvector[1,1]**2 + normvector[1,2]**2
    a     = mu * (vector[1,0] * normvector[1,0] + vector[1,1] * normvector[1,1] + vector[1,2] * normvector[1,2]) / div
    b     = (mu**2 - 1) / div
    to    = -b * 0.5 / a
    num   = 0
    eps   = error * 2
    while eps > error:
       num   += 1
       oldto  = to
       v      = to**2 + 2 * a * to + b
       deltav = 2 * (to + a)
       to     = to - v / deltav
       eps    = abs(oldto - to)
    if num > 5000:
       return False
    VectorOutput = torch.zeros_like(vector)
    VectorOutput[0,0] = normvector[0,0]
    VectorOutput[0,1] = normvector[0,1]
    VectorOutput[0,2] = normvector[0,2]
    VectorOutput[1,0] = mu * vector[1,0] + to * normvector[1,0]
    VectorOutput[1,1] = mu * vector[1,1] + to * normvector[1,1]
    VectorOutput[1,2] = mu * vector[1,2] + to * normvector[1,2]
    return VectorOutput


def reflect(input_ray, normal):
    """ 
    Definition to reflect an incoming ray from a surface defined by a surface normal. Used method described in G.H. Spencer and M.V.R.K. Murty, "General Ray-Tracing Procedure", 1961.

    Parameters
    ----------
    input_ray    : torch.tensor
                   A vector/ray (2x3). It can also be a list of rays (nx2x3).
    normal       : torch.tensor
                   A surface normal (2x3). It also be a list of normals (nx2x3).

    Returns
    ----------
    output_ray   : torch.tensor
                   Array that contains starting points and cosines of a reflected ray.
    """
    if len(input_ray.shape) == 2:
        input_ray = input_ray.view((1, 2, 3))
    if len(normal.shape) == 2:
        normal = normal.view((1, 2, 3))
    mu = 1
    div = normal[:, 1, 0]**2 + normal[:, 1, 1]**2 + normal[:, 1, 2]**2 + 1e-8
    a = mu * (input_ray[:, 1, 0] * normal[:, 1, 0]
              + input_ray[:, 1, 1] * normal[:, 1, 1]
              + input_ray[:, 1, 2] * normal[:, 1, 2]) / div
    a = a.unsqueeze(1)
    n = int(torch.amax(torch.tensor([normal.shape[0], input_ray.shape[0]])))
    output_ray = torch.zeros((n, 2, 3)).to(input_ray.device)
    output_ray[:, 0] = normal[:, 0]
    output_ray[:, 1] = input_ray[:, 1] - 2 * a * normal[:, 1]
    if output_ray.shape[0] == 1:
        output_ray = output_ray.view((2, 3))
    return output_ray


def intersect_w_triangle(ray, triangle):
    """
    Definition to find intersection point of a ray with a triangle. Returns False for each variable if the ray doesn't intersect with a given triangle.

    Parameters
    ----------
    ray          : torch.tensor
                   A vector/ray.
    triangle     : torch.tensor
                   Set of points in X,Y and Z to define a single triangle.

    Returns
    ----------
    normal       : torch.tensor
                   Surface normal at the point of intersection.
    distance     : float
                   Distance in between a starting point of a ray and the intersection point with a given triangle.
    """
    normal, distance = intersect_w_surface(ray, triangle)
    if is_it_on_triangle(normal[0], triangle[0], triangle[1], triangle[2]) == False:
        return None, None
    return normal, distance


def intersect_w_surface(ray, points):
    """
    Definition to find intersection point inbetween a surface and a ray. For more see: http://geomalgorithms.com/a06-_intersect-2.html

    Parameters
    ----------
    ray          : torch.tensor
                   A vector/ray.
    points       : torch.tensor
                   Set of points in X,Y and Z to define a planar surface.

    Returns
    ----------
    normal       : torch.tensor
                   Surface normal at the point of intersection.
    distance     : float
                   Distance in between starting point of a ray with it's intersection with a planar surface.
    """
    normal = get_triangle_normal(points)
    if len(ray.shape) == 2:
        ray = ray.view((1, 2, 3))
    if len(points) == 2:
        points = points.view((1, 3, 3))
    if len(normal.shape) == 2:
        normal = normal.view((1, 2, 3))
    f = normal[:, 0] - ray[:, 0]
    distance = (torch.mm(normal[:, 1], f.T) / torch.mm(normal[:, 1], ray[:, 1].T)).T
    new_normal = torch.zeros_like(ray)
    new_normal[:, 0] = ray[:, 0] + distance * ray[:, 1]
    new_normal[:, 1] = normal[:, 1]
    if new_normal.shape[0] == 1:
        new_normal = new_normal.view((2, 3))
        distance = distance.view((distance.shape[0]))
    if len(distance.shape) > 1:
        distance = distance.view((distance.shape[0]))
    return new_normal, distance


def get_triangle_normal(triangle, triangle_center=None):
    """
    Definition to calculate surface normal of a triangle.

    Parameters
    ----------
    triangle        : torch.tensor
                      Set of points in X,Y and Z to define a planar surface (3,3). It can also be list of triangles (mx3x3).
    triangle_center : torch.tensor
                      Center point of the given triangle. See odak.learn.raytracing.center_of_triangle for more. In many scenarios you can accelerate things by precomputing triangle centers.

    Returns
    ----------
    normal          : torch.tensor
                      Surface normal at the point of intersection.
    """
    if len(triangle.shape) == 2:
        triangle = triangle.view((1, 3, 3))
    normal = torch.zeros((triangle.shape[0], 2, 3)).to(triangle.device)
    direction = torch.cross(
                            triangle[:, 0] - triangle[:, 1], 
                            triangle[:, 2] - triangle[:, 1]
                           )
    if type(triangle_center) == type(None):
        normal[:, 0] = center_of_triangle(triangle)
    else:
        normal[:, 0] = triangle_center
    normal[:, 1] = direction / torch.sum(direction, axis=1)[0]
    if normal.shape[0] == 1:
        normal = normal.view((2, 3))
    return normal


