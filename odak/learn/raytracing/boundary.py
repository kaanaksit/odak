import torch
from tqdm import tqdm
from .ray import propagate_ray, create_ray_from_two_points
from .primitives import is_it_on_triangle, is_it_on_triangle_batch, center_of_triangle
from ..tools.vector import distance_between_two_points


def refract(vector, normvector, n1, n2, error = 0.01):
    """
    Definition to refract an incoming ray.
    Used method described in G.H. Spencer and M.V.R.K. Murty, "General Ray-Tracing Procedure", 1961.


    Parameters
    ----------
    vector         : torch.tensor
                     Incoming ray.
                     Expected size is [2, 3], [1, 2, 3] or [m, 2, 3].
    normvector     : torch.tensor
                     Normal vector.
                     Expected size is [2, 3], [1, 2, 3] or [m, 2, 3]].
    n1             : float
                     Refractive index of the incoming medium.
    n2             : float
                     Refractive index of the outgoing medium.
    error          : float 
                     Desired error.

    Returns
    -------
    output         : torch.tensor
                     Refracted ray.
                     Expected size is [1, 2, 3]
    """
    if len(vector.shape) == 2:
        vector = vector.unsqueeze(0)
    if len(normvector.shape) == 2:
        normvector = normvector.unsqueeze(0)
    mu    = n1 / n2
    div   = normvector[:, 1, 0] ** 2  + normvector[:, 1, 1] ** 2 + normvector[:, 1, 2] ** 2
    a     = mu * (vector[:, 1, 0] * normvector[:, 1, 0] + vector[:, 1, 1] * normvector[:, 1, 1] + vector[:, 1, 2] * normvector[:, 1, 2]) / div
    b     = (mu ** 2 - 1) / div
    to    = - b * 0.5 / a
    num   = 0
    eps   = torch.ones(vector.shape[0], device = vector.device) * error * 2
    while len(eps[eps > error]) > 0:
       num   += 1
       oldto  = to
       v      = to ** 2 + 2 * a * to + b
       deltav = 2 * (to + a)
       to     = to - v / deltav
       eps    = abs(oldto - to)
    output = torch.zeros_like(vector)
    output[:, 0, 0] = normvector[:, 0, 0]
    output[:, 0, 1] = normvector[:, 0, 1]
    output[:, 0, 2] = normvector[:, 0, 2]
    output[:, 1, 0] = mu * vector[:, 1, 0] + to * normvector[:, 1, 0]
    output[:, 1, 1] = mu * vector[:, 1, 1] + to * normvector[:, 1, 1]
    output[:, 1, 2] = mu * vector[:, 1, 2] + to * normvector[:, 1, 2]
    return output


def reflect(input_ray, normal):
    """ 
    Definition to reflect an incoming ray from a surface defined by a surface normal. 
    Used method described in G.H. Spencer and M.V.R.K. Murty, "General Ray-Tracing Procedure", 1961.


    Parameters
    ----------
    input_ray    : torch.tensor
                   A ray or rays.
                   Expected size is [2 x 3], [1 x 2 x 3] or [m x 2 x 3].
    normal       : torch.tensor
                   A surface normal(s).
                   Expected size is [2 x 3], [1 x 2 x 3] or [m x 2 x 3].

    Returns
    ----------
    output_ray   : torch.tensor
                   Array that contains starting points and cosines of a reflected ray.
                   Expected size is [1 x 2 x 3] or [m x 2 x 3].
    """
    if len(input_ray.shape) == 2:
        input_ray = input_ray.unsqueeze(0)
    if len(normal.shape) == 2:
        normal = normal.unsqueeze(0)
    mu = 1
    div = normal[:, 1, 0]**2 + normal[:, 1, 1]**2 + normal[:, 1, 2]**2 + 1e-8
    a = mu * (input_ray[:, 1, 0] * normal[:, 1, 0] + input_ray[:, 1, 1] * normal[:, 1, 1] + input_ray[:, 1, 2] * normal[:, 1, 2]) / div
    a = a.unsqueeze(1)
    n = int(torch.amax(torch.tensor([normal.shape[0], input_ray.shape[0]])))
    output_ray = torch.zeros((n, 2, 3)).to(input_ray.device)
    output_ray[:, 0] = normal[:, 0]
    output_ray[:, 1] = input_ray[:, 1] - 2 * a * normal[:, 1]
    return output_ray


def intersect_w_sphere(ray, sphere, learning_rate = 2e-1, number_of_steps = 5000, error_threshold = 1e-2):
    """
    Definition to find the intersection between ray(s) and sphere(s).

    Parameters
    ----------
    ray                 : torch.tensor
                          Input ray(s).
                          Expected size is [1 x 2 x 3] or [m x 2 x 3].
    sphere              : torch.tensor
                          Input sphere.
                          Expected size is [1 x 4].
    learning_rate       : float
                          Learning rate used in the optimizer for finding the propagation distances of the rays.
    number_of_steps     : int
                          Number of steps used in the optimizer.
    error_threshold     : float
                          The error threshold that will help deciding intersection or no intersection.

    Returns
    -------
    intersecting_ray    : torch.tensor
                          Ray(s) that intersecting with the given sphere.
                          Expected size is [n x 2 x 3], where n could be any real number.
    intersecting_normal : torch.tensor
                          Normal(s) for the ray(s) intersecting with the given sphere
                          Expected size is [n x 2 x 3], where n could be any real number.

    """
    if len(ray.shape) == 2:
        ray = ray.unsqueeze(0)
    if len(sphere.shape) == 1:
        sphere = sphere.unsqueeze(0)
    distance = torch.zeros(ray.shape[0], device = ray.device, requires_grad = True)
    loss_l2 = torch.nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.AdamW([distance], lr = learning_rate)    
    t = tqdm(range(number_of_steps), leave = False, dynamic_ncols = True)
    for step in t:
        optimizer.zero_grad()
        propagated_ray = propagate_ray(ray, distance)
        test = torch.abs((propagated_ray[:, 0, 0] - sphere[:, 0]) ** 2 + (propagated_ray[:, 0, 1] - sphere[:, 1]) ** 2 + (propagated_ray[:, 0, 2] - sphere[:, 2]) ** 2 - sphere[:, 3] ** 2)
        loss = loss_l2(
                       test,
                       torch.zeros_like(test)
                      )
        loss.backward(retain_graph = True)
        optimizer.step()
        t.set_description('Sphere intersection loss: {}'.format(loss.item()))
    check = test < error_threshold
    intersecting_ray = propagate_ray(ray[check == True], distance[check == True])
    intersecting_normal = create_ray_from_two_points(
                                                     sphere[:, 0:3],
                                                     intersecting_ray[:, 0]
                                                    )
    return intersecting_ray, intersecting_normal, distance, check


def intersect_w_triangle(ray, triangle):
    """
    Definition to find intersection point of a ray with a triangle. 

    Parameters
    ----------
    ray                 : torch.tensor
                          A ray [1 x 2 x 3] or a batch of ray [m x 2 x 3].
    triangle            : torch.tensor
                          Set of points in X,Y and Z to define a single triangle [1 x 3 x 3].

    Returns
    ----------
    normal              : torch.tensor
                          Surface normal at the point of intersection with the surface of triangle.
                          This could also involve surface normals that are not on the triangle.
                          Expected size is [1 x 2 x 3] or [m x 2 x 3] depending on the input.
    distance            : float
                          Distance in between a starting point of a ray and the intersection point with a given triangle.
                          Expected size is [1 x 1] or [m x 1] depending on the input.
    intersecting_ray    : torch.tensor
                          Rays that intersect with the triangle plane and on the triangle.
                          Expected size is [1 x 2 x 3] or [m x 2 x 3] depending on the input.
    intersecting_normal : torch.tensor
                          Normals that intersect with the triangle plane and on the triangle.
                          Expected size is [1 x 2 x 3] or [m x 2 x 3] depending on the input.
    check               : torch.tensor
                          A list that provides a bool as True or False for each ray used as input.
                          A test to see is a ray could be on the given triangle.
                          Expected size is [1] or [m].
    """
    if len(triangle.shape) == 2:
       triangle = triangle.unsqueeze(0)
    if len(ray.shape) == 2:
       ray = ray.unsqueeze(0)
    normal, distance = intersect_w_surface(ray, triangle)
    check = is_it_on_triangle(normal[:, 0], triangle)
    intersecting_ray = ray.unsqueeze(0)
    intersecting_ray = intersecting_ray.repeat(triangle.shape[0], 1, 1, 1)
    intersecting_ray = intersecting_ray[check == True]
    intersecting_normal = normal.unsqueeze(0)
    intersecting_normal = intersecting_normal.repeat(triangle.shape[0], 1, 1, 1)
    intersecting_normal = intersecting_normal[check ==  True]
    return normal, distance, intersecting_ray, intersecting_normal, check


def intersect_w_triangle_batch(ray, triangle):
    """
    Definition to find intersection points of rays with triangles. Returns False for each variable if the rays doesn't intersect with given triangles.

    Parameters
    ----------
    ray          : torch.tensor
                   vectors/rays (n x 2 x 3).
    triangle     : torch.tensor
                   Set of points in X,Y and Z to define triangles (m x 3 x 3).

    Returns
    ----------
    normal          : torch.tensor
                      Surface normal at the point of intersection (m x n x 2 x 3).
    distance        : List
                      Distance in between starting point of a ray with it's intersection with a planar surface (m x n).
    intersect_ray   : List
                      List of intersecting rays (k x 2 x 3) where k <= n.
    intersect_normal: List
                      List of intersecting normals (k x 2 x 3) where k <= n*m.
    check           : torch.tensor
                      Boolean tensor (m x n) indicating whether each ray intersects with a triangle or not.
    """
    if len(triangle.shape) == 2:
       triangle = triangle.unsqueeze(0)
    if len(ray.shape) == 2:
       ray = ray.unsqueeze(0)

    normal, distance = intersect_w_surface_batch(ray, triangle)

    check = is_it_on_triangle_batch(normal[:, :, 0], triangle)

    flat_check = check.flatten()
    flat_normal = normal.view(-1, normal.size(-2), normal.size(-1))
    flat_ray = ray.repeat(normal.size(0), 1, 1)
    flat_distance = distance.flatten()

    filtered_normal = torch.masked_select(flat_normal, flat_check.unsqueeze(-1).unsqueeze(-1).repeat(1, 2, 3))
    filtered_ray = torch.masked_select(flat_ray, flat_check.unsqueeze(-1).unsqueeze(-1).repeat(1, 2, 3))
    filtered_distnace = torch.masked_select(flat_distance, flat_check)

    check_count = check.sum(dim=1).tolist()
    split_size_ray_and_normal = [count * 2 * 3 for count in check_count]
    split_size_distance = [count for count in check_count]

    normal_grouped = torch.split(filtered_normal, split_size_ray_and_normal)
    ray_grouped = torch.split(filtered_ray, split_size_ray_and_normal)
    distance_grouped = torch.split(filtered_distnace, split_size_distance)

    intersecting_normal = [g.view(-1, 2, 3) for g in normal_grouped if g.numel() > 0]
    intersecting_ray = [g.view(-1, 2, 3) for g in ray_grouped if g.numel() > 0]
    new_distance = [g for g in distance_grouped if g.numel() > 0]

    return normal, new_distance, intersecting_ray, intersecting_normal, check


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
        ray = ray.unsqueeze(0)
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    if len(normal.shape) == 2:
        normal = normal.unsqueeze(0)
    f = normal[:, 0] - ray[:, 0]
    distance = (torch.mm(normal[:, 1], f.T) / torch.mm(normal[:, 1], ray[:, 1].T)).T
    new_normal = torch.zeros_like(ray)
    new_normal[:, 0] = ray[:, 0] + distance * ray[:, 1]
    new_normal[:, 1] = normal[:, 1]
    new_normal = torch.nan_to_num(
                                  new_normal,
                                  nan = float('nan'),
                                  posinf = float('nan'),
                                  neginf = float('nan')
                                 )
    distance = torch.nan_to_num(
                                distance,
                                nan = float('nan'),
                                posinf = float('nan'),
                                neginf = float('nan')
                               )
    return new_normal, distance


def intersect_w_surface_batch(ray, triangle):
    """
    Parameters
    ----------
    ray          : torch.tensor
                   A vector/ray (2 x 3). It can also be a list of rays (n x 2 x 3).
    triangle     : torch.tensor
                   Set of points in X,Y and Z to define a planar surface. It can also be a list of triangles (m x 3 x 3).

    Returns
    ----------
    normal       : torch.tensor
                   Surface normal at the point of intersection (m x n x 2 x 3).
    distance     : torch.tensor
                   Distance in between starting point of a ray with it's intersection with a planar surface (m x n).
    """
    normal = get_triangle_normal(triangle)
    if len(ray.shape) == 2:
        ray = ray.unsqueeze(0)
    if len(triangle.shape) == 2:
        triangle = triangle.unsqueeze(0)
    if len(normal.shape) == 2:
        normal = normal.unsqueeze(0)

    f = normal[:, None, 0] - ray[None, :, 0]
    distance = (torch.bmm(normal[:, None, 1], f.permute(0, 2, 1)).squeeze(1) / torch.mm(normal[:, 1], ray[:, 1].T)).T

    new_normal = torch.zeros((triangle.shape[0], )+ray.shape)
    new_normal[:, :, 0] = ray[None, :, 0] + (distance[:, :, None] * ray[:, None, 1]).permute(1, 0, 2)
    new_normal[:, :, 1] = normal[:, None, 1]
    new_normal = torch.nan_to_num(
                                  new_normal,
                                  nan = float('nan'),
                                  posinf = float('nan'),
                                  neginf = float('nan')
                                 )
    distance = torch.nan_to_num(
                                distance,
                                nan = float('nan'),
                                posinf = float('nan'),
                                neginf = float('nan')
                               )
    return new_normal, distance.T

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
    direction = torch.linalg.cross(
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


def get_sphere_normal_torch(point, sphere):
    """
    Definition to get a normal of a point on a given sphere.

    Parameters
    ----------
    point         : torch.tensor
                    Point on sphere in X,Y,Z.
    sphere        : torch.tensor
                    Center defined in X,Y,Z and radius.

    Returns
    ----------
    normal_vector : torch.tensor
                    Normal vector.
    """
    if len(point.shape) == 1:
        point = point.reshape((1, 3))
    normal_vector = create_ray_from_two_points(point, sphere[0:3])
    return normal_vector

def intersect_w_circle(ray, circle):
    """
    Definition to find intersection point of a ray with a circle. 
    Returns distance as zero if there isn't an intersection.

    Parameters
    ----------
    ray          : torch.Tensor
                   A vector/ray.
    circle       : list
                   A list that contains (0) Set of points in X,Y and Z to define plane of a circle, (1) circle center, and (2) circle radius.

    Returns
    ----------
    normal       : torch.Tensor
                   Surface normal at the point of intersection.
    distance     : torch.Tensor
                   Distance in between a starting point of a ray and the intersection point with a given triangle.
    """
    normal, distance = intersect_w_surface(ray, circle[0])

    if len(normal.shape) == 2:
        normal = normal.unsqueeze(0)

    distance_to_center = distance_between_two_points(normal[:, 0], circle[1])
    mask = distance_to_center > circle[2]
    distance[mask] = 0
    
    if len(ray.shape) == 2:
        normal = normal.squeeze(0)

    return normal, distance
