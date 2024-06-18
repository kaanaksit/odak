import torch
from ..tools import rotate_points


def create_ray(xyz, abg, direction = False):
    """
    Definition to create a ray.

    Parameters
    ----------
    xyz          : torch.tensor
                   List that contains X,Y and Z start locations of a ray.
                   Size could be [1 x 3], [3], [m x 3].
    abg          : torch.tensor
                   List that contains angles in degrees with respect to the X,Y and Z axes.
                   Size could be [1 x 3], [3], [m x 3].
    direction    : bool
                   If set to True, cosines of `abg` is not calculated.

    Returns
    ----------
    ray          : torch.tensor
                   Array that contains starting points and cosines of a created ray.
                   Size will be either [1 x 3] or [m x 3].
    """
    points = xyz
    angles = abg
    if len(xyz) == 1:
        points = xyz.unsqueeze(0)
    if len(abg) == 1:
        angles = abg.unsqueeze(0)
    ray = torch.zeros(points.shape[0], 2, 3, device = points.device)
    ray[:, 0] = points
    if direction:
        ray[:, 1] = abg
    else:
        ray[:, 1] = torch.cos(torch.deg2rad(abg))
    return ray


def create_ray_from_two_points(x0y0z0, x1y1z1):
    """
    Definition to create a ray from two given points. Note that both inputs must match in shape.

    Parameters
    ----------
    x0y0z0       : torch.tensor
                   List that contains X,Y and Z start locations of a ray.
                   Size could be [1 x 3], [3], [m x 3].
    x1y1z1       : torch.tensor
                   List that contains X,Y and Z ending locations of a ray or batch of rays.
                   Size could be [1 x 3], [3], [m x 3].

    Returns
    ----------
    ray          : torch.tensor
                   Array that contains starting points and cosines of a created ray(s).
    """
    if len(x0y0z0.shape) == 1:
        x0y0z0 = x0y0z0.unsqueeze(0)
    if len(x1y1z1.shape) == 1:
        x1y1z1 = x1y1z1.unsqueeze(0)
    xdiff = x1y1z1[:, 0] - x0y0z0[:, 0]
    ydiff = x1y1z1[:, 1] - x0y0z0[:, 1]
    zdiff = x1y1z1[:, 2] - x0y0z0[:, 2]
    s = (xdiff ** 2 + ydiff ** 2 + zdiff ** 2) ** 0.5
    s[s == 0] = float('nan')
    cosines = torch.zeros_like(x0y0z0 * x1y1z1)
    cosines[:, 0] = xdiff / s
    cosines[:, 1] = ydiff / s
    cosines[:, 2] = zdiff / s
    ray = torch.zeros(xdiff.shape[0], 2, 3, device = x0y0z0.device)
    ray[:, 0] = x0y0z0
    ray[:, 1] = cosines
    return ray

def create_ray_from_all_pairs(x0y0z0, x1y1z1):
    """
    Creates rays from all possible pairs of points in x0y0z0 and x1y1z1.

    Parameters
    ----------
    x0y0z0       : torch.tensor
                   Tensor that contains X, Y, and Z start locations of rays.
                   Size should be [m x 3].
    x1y1z1       : torch.tensor
                   Tensor that contains X, Y, and Z end locations of rays.
                   Size should be [n x 3].

    Returns
    ----------
    rays         : torch.tensor
                   Array that contains starting points and cosines of a created ray(s). Size of [n*m x 2 x 3]
    """

    if len(x0y0z0.shape) == 1:
        x0y0z0 = x0y0z0.unsqueeze(0)
    if len(x1y1z1.shape) == 1:
        x1y1z1 = x1y1z1.unsqueeze(0)
    
    m, n = x0y0z0.shape[0], x1y1z1.shape[0]
    start_points = x0y0z0.unsqueeze(1).expand(-1, n, -1).reshape(-1, 3)
    end_points = x1y1z1.unsqueeze(0).expand(m, -1, -1).reshape(-1, 3)
    
    directions = end_points - start_points
    norms = torch.norm(directions, p=2, dim=1, keepdim=True)
    norms[norms == 0] = float('nan')
    
    normalized_directions = directions / norms

    rays = torch.zeros(m * n, 2, 3, device=x0y0z0.device)
    rays[:, 0, :] = start_points
    rays[:, 1, :] = normalized_directions
    
    return rays

def create_ray_from_grid_w_luminous_angle(center, size, no, tilt, num_ray_per_light, angle_limit):
    """
    Generate a 2D array of lights, each emitting rays within a specified solid angle and tilt.
    
    Parameters:
    ----------
    center              : torch.tensor
                          The center point of the light array, shape [3].
    size                : list[int]
                          The size of the light array [height, width]
    no                  : list[int]
                          The number of the light arary [number of lights in height , number of lights inwidth]
    tilt                : torch.tensor
                          The tilt angles in degrees along x, y, z axes for the rays, shape [3].
    angle_limit         : float
                          The maximum angle in degrees from the initial direction vector within which to emit rays.
    num_rays_per_light  : int
                          The number of rays each light should emit.
    
    Returns:
    ----------
    rays : torch.tensor
           Array that contains starting points and cosines of a created ray(s). Size of [n x 2 x 3]
    """

    samples = torch.zeros((no[0], no[1], 3))

    x = torch.linspace(-size[0] / 2., size[0] / 2., no[0])
    y = torch.linspace(-size[1] / 2., size[1] / 2., no[1])
    X, Y = torch.meshgrid(x, y, indexing='ij')

    samples[:, :, 0] = X.detach().clone()
    samples[:, :, 1] = Y.detach().clone()
    samples = samples.reshape((no[0]*no[1], 3))

    samples, *_ = rotate_points(samples, angles=tilt)

    samples = samples + center
    angle_limit = torch.as_tensor(angle_limit)
    cos_alpha = torch.cos(angle_limit * torch.pi / 180)
    tilt = tilt * torch.pi / 180

    theta = torch.acos(1 - 2 * torch.rand(num_ray_per_light*samples.size(0)) * (1-cos_alpha))
    phi = 2 * torch.pi * torch.rand(num_ray_per_light*samples.size(0))  
    
    directions = torch.stack([
        torch.sin(theta) * torch.cos(phi),  
        torch.sin(theta) * torch.sin(phi),  
        torch.cos(theta)                    
    ], dim=1)
    
    c, s = torch.cos(tilt), torch.sin(tilt)

    Rx = torch.tensor([
        [1, 0, 0],
        [0, c[0], -s[0]],
        [0, s[0], c[0]]
    ])

    Ry = torch.tensor([
        [c[1], 0, s[1]],
        [0, 1, 0],
        [-s[1], 0, c[1]]
    ])

    Rz = torch.tensor([
        [c[2], -s[2], 0],
        [s[2], c[2], 0],
        [0, 0, 1]
    ])

    origins = samples.repeat(num_ray_per_light, 1)

    directions = torch.matmul(directions, (Rz@Ry@Rx).T)

    
    rays = torch.zeros(num_ray_per_light*samples.size(0), 2, 3)
    rays[:, 0, :] = origins
    rays[:, 1, :] = directions

    return rays


def create_ray_from_point_w_luminous_angle(origin, num_ray, tilt, angle_limit):
    """
    Generate rays from a point, tilted by specific angles along x, y, z axes, within a specified solid angle.
    
    Parameters:
    ----------
    origin      : torch.tensor
                  The origin point of the rays, shape [3].
    num_rays    : int
                  The total number of rays to generate.
    tilt        : torch.tensor
                  The tilt angles in degrees along x, y, z axes, shape [3].
    angle_limit : float
                  The maximum angle in degrees from the initial direction vector within which to emit rays.

    Returns:
    ----------
    rays : torch.tensor
           Array that contains starting points and cosines of a created ray(s). Size of [n x 2 x 3]
    """
    angle_limit = torch.as_tensor(angle_limit) 
    cos_alpha = torch.cos(angle_limit * torch.pi / 180)
    tilt = tilt * torch.pi / 180

    theta = torch.acos(1 - 2 * torch.rand(num_ray) * (1-cos_alpha))
    phi = 2 * torch.pi * torch.rand(num_ray)  
    
    
    directions = torch.stack([
        torch.sin(theta) * torch.cos(phi),  
        torch.sin(theta) * torch.sin(phi),  
        torch.cos(theta)                    
    ], dim=1)
    
    c, s = torch.cos(tilt), torch.sin(tilt)

    Rx = torch.tensor([
        [1, 0, 0],
        [0, c[0], -s[0]],
        [0, s[0], c[0]]
    ])

    Ry = torch.tensor([
        [c[1], 0, s[1]],
        [0, 1, 0],
        [-s[1], 0, c[1]]
    ])

    Rz = torch.tensor([
        [c[2], -s[2], 0],
        [s[2], c[2], 0],
        [0, 0, 1]
    ])

    origins = origin.repeat(num_ray, 1)
    directions = torch.matmul(directions, (Rz@Ry@Rx).T)

    
    rays = torch.zeros(num_ray, 2, 3)
    rays[:, 0, :] = origins
    rays[:, 1, :] = directions
    
    return rays


def propagate_ray(ray, distance):
    """
    Definition to propagate a ray at a certain given distance.

    Parameters
    ----------
    ray        : torch.tensor
                 A ray with a size of [2 x 3], [1 x 2 x 3] or a batch of rays with [m x 2 x 3].
    distance   : torch.tensor
                 Distance with a size of [1], [1, m] or distances with a size of [m], [1, m].

    Returns
    ----------
    new_ray    : torch.tensor
                 Propagated ray with a size of [1 x 2 x 3] or batch of rays with [m x 2 x 3].
    """
    if len(ray.shape) == 2:
        ray = ray.unsqueeze(0)
    if len(distance.shape) == 2:
        distance = distance.squeeze(-1)
    new_ray = torch.zeros_like(ray)
    new_ray[:, 0, 0] = distance * ray[:, 1, 0] + ray[:, 0, 0]
    new_ray[:, 0, 1] = distance * ray[:, 1, 1] + ray[:, 0, 1]
    new_ray[:, 0, 2] = distance * ray[:, 1, 2] + ray[:, 0, 2]
    return new_ray
