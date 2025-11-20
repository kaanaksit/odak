import math
import torch
from ...raytracing import center_of_triangle
from ...tools import read_PLY


def rotmatx(angle):
    """
    Definition to generate a rotation matrix along X axis.

    Parameters
    ----------
    angle        : torch.tensor
                   Rotation angles in degrees.

    Returns
    ----------
    rotx         : torch.tensor
                   Rotation matrix along X axis.
    """
    angle = torch.deg2rad(angle)
    rotx = torch.zeros(angle.shape[0], 3, 3, device = angle.device)
    rotx[:, 0, 0] = 1.
    rotx[:, 1, 1] = torch.cos(angle)
    rotx[:, 1, 2] = - torch.sin(angle)
    rotx[:, 2, 1] = torch.sin(angle)
    rotx[:, 2, 2] = torch.cos(angle)
    if rotx.shape[0] == 1:
        rotx = rotx.squeeze(0)
    return rotx


def rotmaty(angle):
    """
    Definition to generate a rotation matrix along Y axis.

    Parameters
    ----------
    angle        : torch.tensor
                   Rotation angles in degrees.

    Returns
    ----------
    roty         : torch.tensor
                   Rotation matrix along Y axis.
    """
    angle = torch.deg2rad(angle)
    roty = torch.zeros(angle.shape[0], 3, 3, device = angle.device)
    roty[:, 0, 0] = torch.cos(angle)
    roty[:, 0, 2] = torch.sin(angle)
    roty[:, 1, 1] = 1.
    roty[:, 2, 0] = - torch.sin(angle)
    roty[:, 2, 2] = torch.cos(angle)
    if roty.shape[0] == 1:
        roty = roty.squeeze(0)
    return roty


def rotmatz(angle):
    """
    Definition to generate a rotation matrix along Z axis.

    Parameters
    ----------
    angle        : torch.tensor
                   Rotation angles in degrees.

    Returns
    ----------
    rotz         : torch.tensor
                   Rotation matrix along Z axis.
    """
    angle = torch.deg2rad(angle)
    rotz = torch.zeros(angle.shape[0], 3, 3, device = angle.device)
    rotz[:, 0, 0] = torch.cos(angle)
    rotz[:, 0, 1] = - torch.sin(angle)
    rotz[:, 1, 0] = torch.sin(angle)
    rotz[:, 1, 1] = torch.cos(angle)
    rotz[:, 2, 2] = 1.
    if rotz.shape[0] == 1:
        rotz = rotz.squeeze(0)
    return rotz


def get_rotation_matrix(tilt_angles = [0., 0., 0.], tilt_order = 'XYZ'):
    """
    Function to generate rotation matrix for given tilt angles and tilt order.


    Parameters
    ----------
    tilt_angles        : list
                         Tilt angles in degrees along XYZ axes.
    tilt_order         : str
                         Rotation order (e.g., XYZ, XZY, ZXY, YXZ, ZYX).

    Returns
    -------
    rotmat             : torch.tensor
                         Rotation matrix.
    """
    rotx = rotmatx(tilt_angles[0])
    roty = rotmaty(tilt_angles[1])
    rotz = rotmatz(tilt_angles[2])
    if tilt_order =='XYZ':
        rotmat = torch.mm(rotz,torch.mm(roty, rotx))
    elif tilt_order == 'XZY':
        rotmat = torch.mm(roty,torch.mm(rotz, rotx))
    elif tilt_order == 'ZXY':
        rotmat = torch.mm(roty,torch.mm(rotx, rotz))
    elif tilt_order == 'YXZ':
        rotmat = torch.mm(rotz,torch.mm(rotx, roty))
    elif tilt_order == 'ZYX':
         rotmat = torch.mm(rotx,torch.mm(roty, rotz))
    return rotmat


def rotate_points(
                 point,
                 angles = torch.zeros(1, 3), 
                 mode = 'XYZ', 
                 origin = torch.zeros(1, 3), 
                 offset = torch.zeros(1, 3),
                ):
    """
    Definition to rotate a given point. Note that rotation is always with respect to 0,0,0.

    Parameters
    ----------
    point        : torch.tensor
                   A point with size of [3] or [1, 3] or [m, 3].
    angles       : torch.tensor
                   Rotation angles in degrees. 
    mode         : str
                   Rotation mode determines ordering of the rotations at each axis.
                   There are XYZ,YXZ,ZXY and ZYX modes.
    origin       : torch.tensor
                   Reference point for a rotation.
                   Expected size is [3] or [1, 3].
    offset       : torch.tensor
                   Shift with the given offset.
                   Expected size is [3] or [1, 3] or [m, 3].

    Returns
    ----------
    result       : torch.tensor
                   Result of the rotation [1 x 3] or [m x 3].
    rotx         : torch.tensor
                   Rotation matrix along X axis [3 x 3].
    roty         : torch.tensor
                   Rotation matrix along Y axis [3 x 3].
    rotz         : torch.tensor
                   Rotation matrix along Z axis [3 x 3].
    """
    origin = origin.to(point.device)
    offset = offset.to(point.device)
    angles = angles.to(point.device)

    if len(point.shape) == 1:
        point = point.unsqueeze(0)
    if len(angles.shape) == 1:
        angles = angles.unsqueeze(0)
    if len(origin.shape) == 1:
        origin = origin.unsqueeze(0)
    if len(offset.shape) == 1:
        offset = offset.unsqueeze(0)        

    rotx = rotmatx(angles[:, 0]).unsqueeze(0)
    roty = rotmaty(angles[:, 1]).unsqueeze(0)
    rotz = rotmatz(angles[:, 2]).unsqueeze(0)

    new_points = (point.unsqueeze(1) - origin.unsqueeze(0)).unsqueeze(-1)

    if mode == 'XYZ':
        result = (rotz @ (roty @ (rotx @ new_points)))
    elif mode == 'XZY':
        result = (roty @ (rotz @ (rotx @ new_points)))
    elif mode == 'YXZ':
        result = (rotz @ (rotx @ (roty @ new_points)))
    elif mode == 'ZXY':
        result = (roty @ (rotx @ (rotz @ new_points)))
    elif mode == 'ZYX':
        result = (rotx @ (roty @ (rotz @ new_points)))

    result = result.squeeze(-1)
    result = result + origin.unsqueeze(0) 
    result = result + offset.unsqueeze(0)
    if result.shape[1] == 1:
        result = result.squeeze(1)
    return result, rotx, roty, rotz


def tilt_towards(location, lookat):
    """
    Definition to tilt surface normal of a plane towards a point.

    Parameters
    ----------
    location     : list
                   Center of the plane to be tilted.
    lookat       : list
                   Tilt towards this point.

    Returns
    ----------
    angles       : list
                   Rotation angles in degrees.
    """
    dx = location[0] - lookat[0]
    dy = location[1] - lookat[1]
    dz = location[2] - lookat[2]
    dist = torch.sqrt(torch.tensor(dx ** 2 + dy ** 2 + dz ** 2))
    phi = torch.atan2(torch.tensor(dy), torch.tensor(dx))
    theta = torch.arccos(dz / dist)
    angles = [0, float(torch.rad2deg(theta)), float(torch.rad2deg(phi))]
    return angles


def point_cloud_to_voxel(
                         points,
                         voxel_size = [0.1, 0.1, 0.1],
                        ):
    """
    Convert a point cloud to a voxel grid representation.

    Parameters
    ----------
    points     : torch.Tensor, shape (N, 3)
                 The input point cloud, where each row is a 3D point.
    voxel_size : list or torch.Tensor, shape (3,), optional
                 The size of each voxel in the x, y, and z directions. Default is [0.1, 0.1, 0.1].

    Returns
    -------
    locations  : torch.Tensor, shape (Gx, Gy, Gz, 3)
                 The coordinates of each voxel center in the grid.
    grid       : torch.Tensor, shape (Gx, Gy, Gz)
                 A binary voxel grid where 1 indicates the presence of at least one point.

    Notes
    -----
    - The voxel grid is constructed by discretizing the space between the minimum and maximum
      coordinates of the point cloud.
    - Only voxels containing at least one point are marked as 1.
    - The output grid is of type float32 and resides on the same device as the input points.
    """
    voxel_size = torch.as_tensor(voxel_size, device = points.device)

    min_coords = points.min(dim = 0).values
    max_coords = points.max(dim = 0).values
    grid_size = ((max_coords - min_coords) / voxel_size).ceil().int()
    points = points - min_coords

    x = torch.linspace(min_coords[0], max_coords[0], grid_size[0], device = points.device)
    y = torch.linspace(min_coords[1], max_coords[1], grid_size[1], device = points.device)
    z = torch.linspace(min_coords[2], max_coords[2], grid_size[2], device = points.device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    locations = torch.stack([X, Y, Z], dim=-1)

    voxel_indices = (points / voxel_size).floor().int()
    mask = (voxel_indices >= 0).all(dim=1) & (voxel_indices < grid_size).all(dim=1)
    voxel_indices = voxel_indices[mask]
    grid = torch.zeros(grid_size.tolist(), dtype=torch.float32, device = points.device)
    grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.

    return locations, grid


def load_voxelized_PLY(
                       ply_filename,
                       voxel_size = [0.05, 0.05, 0.05],
                       device = torch.device('cpu'),
                      ):
    """
    Load a point cloud from a PLY file and convert it into a voxel grid representation.

    Parameters
    ----------
    ply_filename : str or Path
                   The path to the input PLY file containing triangle data.
    voxel_size   : list or tuple, shape (3,), optional
                   The size of each voxel in the x, y, and z directions. Default is [0.05, 0.05, 0.05].
    device       : torch.device, optional
                   The device on which to perform computations. Default is CPU.

    Returns
    -------
    points      : torch.Tensor, shape (N, 3)
                  A tensor containing the coordinates of the voxel centers.
    ground_truth: torch.Tensor, shape (Gx * Gy * Gz,)
                  A binary tensor where each element indicates whether a corresponding voxel contains at least one point.

    Notes
    -----
    - The function reads triangle data from the PLY file and computes the center points of these triangles.
    - These points are then processed to create a normalized point cloud, which is converted into a voxel grid.
    - Only voxels containing at least one point are marked as 1 in `ground_truth`.
    - All operations are performed on the specified device for efficiency.
    """
    triangles = read_PLY(ply_filename)
    points = center_of_triangle(triangles)
    points = torch.as_tensor(points, device = device)
    points = points - points.mean()
    points = points / torch.amax(points)
    ground_truth = torch.ones(points.shape[0], device = device)
    voxel_locations, voxel_grid = point_cloud_to_voxel(points = points, voxel_size = voxel_size,)
    points = voxel_locations.reshape(-1, 3)
    ground_truth = voxel_grid.reshape(-1)
    return points, ground_truth
