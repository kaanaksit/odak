"""
``odak.learn.wave.complex_gaussians``

Provides complex-valued 3D Gaussian splatting primitives for holographic rendering.
This module implements the core Gaussian representation and wave-based rendering
pipeline from "Complex Valued Holographic Radiance Fields" as a pure PyTorch
implementation without external dependencies (no pytorch3d or CUDA extensions).

References
----------
Zhan, Y etal. 2025. ACM TOG
Complex Valued Holographic Radiance Fields.
"""

import math
from argparse import Namespace
from typing import Optional, Tuple

import odak
import torch
import torch.nn.functional as F
from odak.learn.tools import circular_binary_mask, crop_center, zero_pad


def quaternion_to_rotation_matrix(quaternions):
    """
    Convert rotations given as unit quaternions to rotation matrices.

    This is a pure PyTorch replacement for ``pytorch3d.transforms.quaternion_to_matrix``.

    Parameters
    ----------
    quaternions : torch.Tensor
                  Quaternions with real part first, shape ``(*, 4)``
                  in ``(w, x, y, z)`` convention.

    Returns
    -------
    rotation_matrices : torch.Tensor
                        Rotation matrices, shape ``(*, 3, 3)``.
    """
    quaternions = F.normalize(quaternions, dim=-1)
    w, x, y, z = quaternions.unbind(-1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    rotation_matrices = torch.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - w * z),
            two_s * (x * z + w * y),
            two_s * (x * y + w * z),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - w * x),
            two_s * (x * z - w * y),
            two_s * (y * z + w * x),
            1 - two_s * (x * x + y * y),
        ],
        dim=-1,
    )

    return rotation_matrices.reshape(quaternions.shape[:-1] + (3, 3))


class PerspectiveCamera:
    """
    A lightweight perspective camera model.

    Replacement for ``pytorch3d.renderer.cameras.PerspectiveCameras``
    that stores camera intrinsics and extrinsics and provides
    coordinate-transform utilities.

    Parameters
    ----------
    R              : torch.Tensor
                     Rotation matrix, shape ``(3, 3)`` or ``(1, 3, 3)``.
    T              : torch.Tensor
                     Translation vector, shape ``(3,)`` or ``(1, 3)``.
    focal_length   : torch.Tensor
                     Focal lengths ``(fx, fy)``, shape ``(2,)`` or ``(1, 2)``.
    principal_point: torch.Tensor
                     Principal point ``(px, py)``, shape ``(2,)`` or ``(1, 2)``.
    device         : torch.device or str, optional
                     Device for all tensors (default: ``"cpu"``).
    """

    def __init__(self, R, T, focal_length, principal_point, device="cpu"):
        self.device = torch.device(device)
        self.R = (
            R.to(self.device)
            if isinstance(R, torch.Tensor)
            else torch.tensor(R, dtype=torch.float32, device=self.device)
        )
        self.T = (
            T.to(self.device)
            if isinstance(T, torch.Tensor)
            else torch.tensor(T, dtype=torch.float32, device=self.device)
        )
        self.focal_length = (
            focal_length.to(self.device)
            if isinstance(focal_length, torch.Tensor)
            else torch.tensor(focal_length, dtype=torch.float32, device=self.device)
        )
        self.principal_point = (
            principal_point.to(self.device)
            if isinstance(principal_point, torch.Tensor)
            else torch.tensor(principal_point, dtype=torch.float32, device=self.device)
        )

    def transform_world_to_camera_space(self, points):
        """
        Transform world-space points into camera space.

        Follows the pytorch3d convention: ``X_cam = X_world @ R + T``.

        Parameters
        ----------
        points : torch.Tensor
                 World-space points, shape ``(N, 3)``.

        Returns
        -------
        cam_points : torch.Tensor
                     Camera-space points, shape ``(N, 3)``.
        """
        R = self.R[0] if self.R.dim() == 3 else self.R
        T = self.T[0] if self.T.dim() == 2 else self.T
        return points @ R + T

    def get_camera_center(self):
        """
        Compute the camera centre in world coordinates.

        Returns
        -------
        center : torch.Tensor
                 Camera centre, shape ``(1, 3)``.
        """
        R = self.R[0] if self.R.dim() == 3 else self.R
        T = self.T[0] if self.T.dim() == 2 else self.T
        center = -T @ R.transpose(0, 1)
        return center.unsqueeze(0)


def bandlimited_angular_spectrum_propagation(
    field,
    wavelength,
    pixel_pitch,
    distance,
    size,
    aperture_size=-1,
):
    """
    Band-limited angular spectrum propagation (pure PyTorch).

    Propagates a 2-D complex field by a given ``distance`` using the
    band-limited angular-spectrum method with zero-padding.

    Parameters
    ----------
    field        : torch.Tensor
                   Input complex field, shape ``(H, W)``.
    wavelength   : float
                   Wavelength of light in metres.
    pixel_pitch  : float
                   Pixel pitch in metres.
    distance     : float
                   Propagation distance in metres (can be negative).
    size         : tuple of int
                   ``(H, W)`` of the *original* (unpadded) field.
    aperture_size: float, optional
                   Aperture radius in pixels for a circular mask.
                   ``-1`` disables the aperture (default).

    Returns
    -------
    field_propagated : torch.Tensor
                       Propagated complex field, cropped back to
                       ``size``.
    """
    padded_size = [i * 2 for i in size]
    original_shape = field.shape
    field = zero_pad(field, padded_size)

    aperture = (
        circular_binary_mask(
            padded_size[0],
            padded_size[1],
            aperture_size,
        ).to(field.device)
        * 1.0
    )

    field_f = torch.fft.fftshift(torch.fft.fft2(field))

    Nx, Ny = padded_size
    fx = torch.fft.fftshift(torch.fft.fftfreq(Nx, d=pixel_pitch)).to(field.device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(Ny, d=pixel_pitch)).to(field.device)
    FX, FY = torch.meshgrid(fx, fy, indexing="ij")

    x = torch.tensor(pixel_pitch * float(Nx), device=field.device)
    y = torch.tensor(pixel_pitch * float(Ny), device=field.device)
    distance_t = torch.tensor(distance, device=field.device)
    wavelength_t = torch.tensor(wavelength, device=field.device)

    fx_max = 1 / torch.sqrt((2 * distance_t * (1 / x)) ** 2 + 1) / wavelength_t
    fy_max = 1 / torch.sqrt((2 * distance_t * (1 / y)) ** 2 + 1) / wavelength_t
    bandlimit_mask = (torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)

    k = 2 * torch.pi / wavelength_t
    kz = torch.sqrt(k**2 - (2 * torch.pi * FX) ** 2 - (2 * torch.pi * FY) ** 2)
    kz = torch.where(torch.isnan(kz), torch.zeros_like(kz), kz)

    H = torch.exp(1j * distance_t * kz) * bandlimit_mask

    field_propagated_f = field_f * H * aperture
    field_propagated = torch.fft.ifft2(torch.fft.ifftshift(field_propagated_f))
    field_propagated = crop_center(field_propagated)
    # Ensure output matches input dimensionality (2D field in, 2D field out)
    while field_propagated.dim() > len(original_shape):
        field_propagated = field_propagated.squeeze(0)
    return field_propagated


class STEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for discrete (argmax) selection.

    In the forward pass the input logits are discretised via argmax +
    one-hot encoding.  In the backward pass a temperature-scaled
    softmax is used so that gradients flow through.
    """

    @staticmethod
    def forward(ctx, input, temperature=0.001):
        ctx.save_for_backward(input)
        ctx.temperature = temperature
        indices = torch.argmax(input, dim=1)
        return F.one_hot(indices, num_classes=input.size(1)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        temperature = ctx.temperature
        soft_probs = F.softmax(input / temperature, dim=1)
        return grad_output * soft_probs, None


class StraightThroughEstimator(torch.nn.Module):
    """
    Module wrapper around :class:`STEFunction`.

    Parameters
    ----------
    temperature : float
                  Softmax temperature used during the backward pass
                  (default: ``0.001``).
    """

    def __init__(self, temperature=0.001):
        super(StraightThroughEstimator, self).__init__()
        self.temperature = temperature

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input logits, shape ``(N, C)``.

        Returns
        -------
        y : torch.Tensor
            One-hot encoded output, shape ``(N, C)``.
        """
        return STEFunction.apply(x, self.temperature)


class Gaussians(torch.nn.Module):
    """
    Complex-valued 3-D Gaussian primitives for holographic rendering.

    Each Gaussian is parameterised by a 3-D mean, a rotation quaternion,
    log-scales, per-channel colour amplitudes, per-channel phases,
    opacity, and a discrete plane-assignment vector.

    Parameters
    ----------
    init_type        : str
                       One of ``"gaussians"`` (load from checkpoint),
                       ``"random"`` (random initialisation), or
                       ``"point"`` (from a point cloud).
    device           : str
                       Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    load_path        : str or None, optional
                       Path to a ``.pth`` checkpoint (required when
                       ``init_type="gaussians"``).
    num_points       : int or None, optional
                       Number of Gaussians (required when
                       ``init_type="random"``).
    args_prop        : argparse.Namespace
                       Must contain at least ``num_planes``.
    pointcloud_data  : dict or None, optional
                       ``{"positions": Tensor, "colors": Tensor}``
                       (required when ``init_type="point"``).
    generate_dense_point : int, optional
                       Number of densification rounds (default: ``0``).
    densepoint_scatter   : float, optional
                       Standard deviation of the densification noise
                       (default: ``0.01``).
    img_size         : tuple or None, optional
                       Image size for random init hints.
    """

    def __init__(
        self,
        init_type: str,
        device: str,
        load_path: Optional[str] = None,
        num_points: Optional[int] = None,
        args_prop: Namespace = None,
        pointcloud_data: Optional[dict] = None,
        generate_dense_point=False,
        densepoint_scatter=0.01,
        img_size=None,
    ):
        super(Gaussians, self).__init__()

        self.device = device
        self.num_planes = args_prop.num_planes
        self.generate_dense_point = generate_dense_point
        self.densepoint_scatter = densepoint_scatter
        self.NEAR_PLANE = 1.0
        self.FAR_PLANE = 1000.0

        if init_type == "gaussians":
            if load_path is None:
                raise ValueError("load_path is required for init_type='gaussians'")
            data = self._load_gaussians(load_path)

        elif init_type == "random":
            if num_points is None:
                raise ValueError("num_points is required for init_type='random'")
            data = self._load_random(num_points, img_size)

        elif init_type == "point":
            if pointcloud_data is None:
                raise ValueError("pointcloud_data is required for init_type='point'")
            self.is_outdoor = args_prop.is_outdoor
            data = self._load_point(pointcloud_data)

        else:
            raise ValueError(f"Invalid init_type: {init_type}")

        self.register_parameter(
            "pre_act_quats",
            torch.nn.Parameter(data["pre_act_quats"], requires_grad=False),
        )
        self.register_parameter(
            "means", torch.nn.Parameter(data["means"], requires_grad=False)
        )
        self.register_parameter(
            "pre_act_scales",
            torch.nn.Parameter(data["pre_act_scales"], requires_grad=False),
        )
        self.register_parameter(
            "colours", torch.nn.Parameter(data["colours"], requires_grad=False)
        )
        self.register_parameter(
            "pre_act_phase",
            torch.nn.Parameter(data["pre_act_phase"], requires_grad=False),
        )
        self.register_parameter(
            "pre_act_opacities",
            torch.nn.Parameter(data["pre_act_opacities"], requires_grad=False),
        )
        self.register_parameter(
            "pre_act_plane_assignment",
            torch.nn.Parameter(data["pre_act_plane_assignment"], requires_grad=False),
        )
        self.to(self.device)

    def __len__(self):
        return len(self.means)

    def _load_gaussians(self, ply_path: str):
        if ply_path.endswith(".pth"):
            checkpoint = torch.load(ply_path, map_location="cpu", weights_only=False)
            data = {
                "pre_act_quats": checkpoint["pre_act_quats"]
                .clone()
                .detach()
                .to(torch.float32)
                .contiguous(),
                "means": checkpoint["means"]
                .clone()
                .detach()
                .to(torch.float32)
                .contiguous(),
                "pre_act_scales": checkpoint["pre_act_scales"]
                .clone()
                .detach()
                .to(torch.float32)
                .contiguous(),
                "colours": checkpoint["colours"]
                .clone()
                .detach()
                .to(torch.float32)
                .contiguous(),
                "pre_act_phase": checkpoint["pre_act_phase"]
                .clone()
                .detach()
                .to(torch.float32)
                .contiguous(),
                "pre_act_opacities": checkpoint["pre_act_opacities"]
                .clone()
                .detach()
                .to(torch.float32)
                .contiguous(),
                "pre_act_plane_assignment": checkpoint["pre_act_plane_assignment"]
                .clone()
                .detach()
                .to(torch.float32)
                .contiguous(),
            }
            num = len(data["means"])
            print(f"Loaded Gaussians {num} from checkpoint: {ply_path}")
            return data

    def _load_random(self, num_points: int, image_size=None):
        data = dict()
        means = (torch.rand((num_points, 3)) * 2 - 1).to(torch.float32) * 15.7
        data["means"] = means.to(torch.float32)
        data["colours"] = torch.rand((num_points, 3), dtype=torch.float32)
        quats_norm = torch.randn((num_points, 4), dtype=torch.float32)
        quats_norm = F.normalize(quats_norm, dim=1)
        quats = torch.zeros((num_points, 4), dtype=torch.float32)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats + quats_norm * 0.01
        data["pre_act_scales"] = torch.log(
            (torch.rand((num_points, 1), dtype=torch.float32) + 1e-6) * 0.01
        )
        data["pre_act_scales"] = data["pre_act_scales"].repeat(1, 3)
        data["pre_act_phase"] = torch.randn((num_points, 3), dtype=torch.float32)
        data["pre_act_opacities"] = torch.ones((num_points,), dtype=torch.float32)
        data["pre_act_plane_assignment"] = (
            torch.randn((num_points, self.num_planes), dtype=torch.float32) * 10.0
        )

        print(
            f"Loaded Randomly {num_points} gaussians with image size {image_size if image_size else 'default'}"
        )
        return data

    def _load_point(self, pointcloud_data: dict) -> dict:
        positions = pointcloud_data["positions"]
        colors = pointcloud_data["colors"]
        data = {}

        centre = positions.mean(dim=0, keepdim=True)
        distances = torch.norm(positions - centre, dim=1)

        if self.is_outdoor:
            num_points = positions.shape[0]
            num_points_to_keep = int(num_points * 0.98)
            sorted_indices = torch.argsort(distances)
            keep_indices = sorted_indices[:num_points_to_keep]
            print(f"Keeping {num_points_to_keep} points from {num_points} points")
            positions = positions[keep_indices]
            colors = colors[keep_indices]

        if self.generate_dense_point > 0:
            orig_positions = positions
            orig_colors = colors
            for _ in range(self.generate_dense_point):
                offset = torch.randn_like(orig_positions) * self.densepoint_scatter
                positions = torch.cat([positions, orig_positions + offset], dim=0)
                colors = torch.cat([colors, orig_colors], dim=0)

        if self.is_outdoor:
            divide = self.generate_dense_point if self.generate_dense_point > 0 else 1
            divide = 1
            bg_count = int(positions.shape[0] * (0.8 / divide))
            print(f"randomize {bg_count} points for outdoor scene")
            centre = positions.mean(dim=0, keepdim=True)
            centred = positions - centre
            max_dist = torch.norm(centred, dim=1).max().item()

            cov = torch.matmul(centred.T, centred) / centred.shape[0]
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, sorted_indices]

            main_axis = eigenvectors[:, 0]
            up_direction = eigenvectors[:, 1]
            side_direction = eigenvectors[:, 2]

            pole_threshold = 0.35
            valid_directions = []
            batch_size = bg_count * 3

            while len(valid_directions) < bg_count:
                directions_batch = F.normalize(
                    torch.randn(batch_size, 3, device=positions.device), dim=1
                )
                valid_mask = torch.abs(directions_batch[:, 2]) <= pole_threshold
                valid_batch = directions_batch[valid_mask]
                valid_directions.append(valid_batch)
                all_valid = torch.cat(valid_directions, dim=0)
                if all_valid.size(0) >= bg_count:
                    directions_standard = all_valid[:bg_count]
                    break

            rotation_matrix = torch.stack(
                [main_axis, up_direction, side_direction], dim=1
            )
            directions = torch.matmul(directions_standard, rotation_matrix.T)
            radii = torch.empty(bg_count, device=positions.device).uniform_(
                max_dist * 0.5, max_dist * 0.8
            )
            bg_positions = directions * radii.unsqueeze(1) + centre
            bg_colors = torch.rand((bg_count, 3), dtype=torch.float32).to(
                positions.device
            )
            positions = torch.cat([positions, bg_positions.to(positions.device)], dim=0)
            colors = torch.cat([colors, bg_colors], dim=0)

        total_points = positions.shape[0]
        print(f"Total points in original point cloud: {total_points}")

        data["means"] = positions.to(torch.float32).contiguous()
        data["colours"] = colors.to(torch.float32).contiguous()

        quats_norm = F.normalize(
            torch.randn((total_points, 4), dtype=torch.float32), dim=1
        )
        quats = torch.zeros((total_points, 4), dtype=torch.float32)
        quats[:, 0] = 1.0
        data["pre_act_quats"] = quats + quats_norm * 0.01

        scales = torch.log(
            (torch.rand((total_points, 1), dtype=torch.float32) + 1e-6) * 0.01
        )
        data["pre_act_scales"] = scales.repeat(1, 3)
        data["pre_act_phase"] = torch.randn((total_points, 3), dtype=torch.float32)
        data["pre_act_opacities"] = torch.ones(total_points, dtype=torch.float32)
        data["pre_act_plane_assignment"] = (
            torch.randn((total_points, self.num_planes), dtype=torch.float32) * 10.0
        )

        print(f"Initialized {total_points} Gaussians from point cloud data")
        return data

    def check_if_trainable(self):
        """Raise an exception if any learnable parameter has ``requires_grad=False``."""
        attrs = [
            "means",
            "pre_act_scales",
            "colours",
            "pre_act_phase",
            "pre_act_opacities",
            "pre_act_plane_assignment",
            "pre_act_quats",
        ]
        for attr in attrs:
            param = getattr(self, attr)
            if not getattr(param, "requires_grad", False):
                raise Exception(
                    "Please use function make_trainable to make parameters trainable"
                )

    def compute_cov_3D(self, quats: torch.Tensor, scales: torch.Tensor):
        """
        Compute 3-D covariance matrices from quaternions and scales.

        Parameters
        ----------
        quats  : torch.Tensor
                 Unit quaternions ``(N, 4)`` in ``(w, x, y, z)`` convention.
        scales : torch.Tensor
                 Scale vectors ``(N, 3)``.

        Returns
        -------
        cov_3D : torch.Tensor
                 Covariance matrices ``(N, 3, 3)``.
        """
        Is = torch.eye(scales.size(1), device=scales.device)
        scale_mats = (scales.unsqueeze(2).expand(*scales.size(), scales.size(1))) * Is
        rots = quaternion_to_rotation_matrix(quats)
        cov_3D = torch.matmul(rots, scale_mats)
        cov_3D = torch.matmul(cov_3D, torch.transpose(scale_mats, 1, 2))
        cov_3D = torch.matmul(cov_3D, torch.transpose(rots, 1, 2))
        return cov_3D

    def _compute_jacobian(self, cam_means_3D: torch.Tensor, fx, fy, img_size: Tuple):
        """
        Compute the Jacobian matrix for the perspective projection.

        Parameters
        ----------
        cam_means_3D : torch.Tensor
                       Camera-space 3-D means ``(N, 3)``.
        fx, fy       : float or torch.Tensor
                       Focal lengths in pixels.
        img_size     : tuple of int
                       ``(W, H)`` image dimensions.

        Returns
        -------
        J : torch.Tensor
            Jacobian matrices ``(N, 2, 3)``.
        """
        W, H = img_size
        half_tan_fov_x = 0.5 * W / fx
        half_tan_fov_y = 0.5 * H / fy

        tx = cam_means_3D[:, 0]
        ty = cam_means_3D[:, 1]
        tz = cam_means_3D[:, 2]
        tz2 = tz * tz

        clipping_mask = (tz > self.NEAR_PLANE) & (tz < self.FAR_PLANE)

        lim_x = 1.3 * half_tan_fov_x
        lim_y = 1.3 * half_tan_fov_y

        tx = torch.clamp(tx / tz, -lim_x, lim_x) * tz
        ty = torch.clamp(ty / tz, -lim_y, lim_y) * tz

        J = torch.zeros((len(tx), 2, 3), device=cam_means_3D.device)
        J[:, 0, 0] = fx / tz
        J[:, 1, 1] = fy / tz
        J[:, 0, 2] = -(fx * tx) / tz2
        J[:, 1, 2] = -(fy * ty) / tz2

        clipping_mask = clipping_mask.to(torch.float32).view(-1, 1, 1)
        J = J * clipping_mask

        return J

    def compute_cov_2D(
        self,
        cam_means_3D: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        fx,
        fy,
        R,
        img_size: Tuple,
    ):
        """
        Compute 2-D projected covariance matrices (Eq. 5 of 3DGS paper).

        Parameters
        ----------
        cam_means_3D : torch.Tensor
                       Camera-space means ``(N, 3)``.
        quats        : torch.Tensor
                       Quaternions ``(N, 4)``.
        scales       : torch.Tensor
                       Scales ``(N, 3)``.
        fx, fy       : float or torch.Tensor
                       Focal lengths.
        R            : torch.Tensor
                       View rotation matrix.
        img_size     : tuple of int
                       ``(W, H)``.

        Returns
        -------
        cov_2D : torch.Tensor
                 2-D covariance matrices ``(N, 2, 2)``.

        References
        ----------
        Kerbl, B. et al. "3D Gaussian Splatting for Real-Time Radiance Field
        Rendering." *SIGGRAPH 2023*.
        """
        J = self._compute_jacobian(cam_means_3D, fx, fy, img_size)
        N = J.shape[0]

        W = R.repeat(N, 1, 1)
        cov_3D = self.compute_cov_3D(quats, scales)

        cov_2D = torch.matmul(J, W)
        cov_2D = torch.matmul(cov_2D, cov_3D)
        cov_2D = torch.matmul(cov_2D, torch.transpose(W, 1, 2))
        cov_2D = torch.matmul(cov_2D, torch.transpose(J, 1, 2))

        cov_2D[:, 0, 0] += 0.3
        cov_2D[:, 1, 1] += 0.3

        return cov_2D

    def compute_means_2D(self, cam_means_3D: torch.Tensor, fx, fy, px, py):
        """
        Project 3-D camera-space points to 2-D pixel coordinates.

        Parameters
        ----------
        cam_means_3D : torch.Tensor
                       Camera-space means ``(N, 3)``.
        fx, fy       : float or torch.Tensor
                       Focal lengths.
        px, py       : float or torch.Tensor
                       Principal-point offsets.

        Returns
        -------
        means_2D : torch.Tensor
                   2-D pixel coordinates ``(N, 2)``.
        """
        clipping_mask = (cam_means_3D[:, 2] > self.NEAR_PLANE) & (
            cam_means_3D[:, 2] < self.FAR_PLANE
        )

        inv_z = 1.0 / cam_means_3D[:, 2].unsqueeze(1)
        cam_means_3D_xy = -cam_means_3D[:, :2] * inv_z

        means_2D = torch.empty((cam_means_3D.shape[0], 2), device=cam_means_3D.device)
        means_2D[:, 0] = fx * cam_means_3D_xy[:, 0] + px
        means_2D[:, 1] = fy * cam_means_3D_xy[:, 1] + py

        large_value = 1e6
        means_2D[~clipping_mask] = large_value

        return means_2D

    @staticmethod
    def invert_cov_2D(cov_2D: torch.Tensor):
        """
        Invert 2×2 covariance matrices.

        Parameters
        ----------
        cov_2D : torch.Tensor
                 Covariance matrices ``(N, 2, 2)``.

        Returns
        -------
        cov_2D_inverse : torch.Tensor
                         Inverse covariance matrices ``(N, 2, 2)``.
        """
        determinants = (
            cov_2D[:, 0, 0] * cov_2D[:, 1, 1] - cov_2D[:, 1, 0] * cov_2D[:, 0, 1]
        )
        determinants = determinants[:, None, None]

        cov_2D_inverse = torch.zeros_like(cov_2D)
        cov_2D_inverse[:, 0, 0] = cov_2D[:, 1, 1]
        cov_2D_inverse[:, 1, 1] = cov_2D[:, 0, 0]
        cov_2D_inverse[:, 0, 1] = -1.0 * cov_2D[:, 0, 1]
        cov_2D_inverse[:, 1, 0] = -1.0 * cov_2D[:, 1, 0]

        cov_2D_inverse = (1.0 / determinants) * cov_2D_inverse
        return cov_2D_inverse

    @staticmethod
    def calculate_gaussian_bounds(means_2D, cov_2D, img_size, confidence=3.0):
        """
        Compute axis-aligned bounding boxes from 2-D covariance.

        Parameters
        ----------
        means_2D   : torch.Tensor
                     2-D positions ``(N, 2)``.
        cov_2D     : torch.Tensor
                     Covariance matrices ``(N, 2, 2)``.
        img_size   : tuple of int
                     ``(W, H)``.
        confidence : float, optional
                     Number of standard deviations (default: ``3.0``).

        Returns
        -------
        bounds : torch.Tensor
                 ``(N, 4)`` with ``[min_x, min_y, max_x, max_y]``.
        """
        var_x = cov_2D[:, 0, 0]
        var_y = cov_2D[:, 1, 1]

        std_x = torch.sqrt(var_x)
        std_y = torch.sqrt(var_y)

        radius_x = confidence * std_x
        radius_y = confidence * std_y

        min_x = means_2D[:, 0] - radius_x
        min_y = means_2D[:, 1] - radius_y
        max_x = means_2D[:, 0] + radius_x
        max_y = means_2D[:, 1] + radius_y

        W, H = img_size
        min_x = torch.clamp(min_x, 0, W - 1)
        min_y = torch.clamp(min_y, 0, H - 1)
        max_x = torch.clamp(max_x, 0, W - 1)
        max_y = torch.clamp(max_y, 0, H - 1)

        bounds = torch.stack([min_x, min_y, max_x, max_y], dim=1)
        return bounds

    @staticmethod
    def apply_activations(
        pre_act_quats,
        pre_act_scales,
        pre_act_phase=None,
        pre_act_opacities=None,
        pre_act_plane_assignment=None,
        step=None,
        max_step=None,
    ):
        """
        Apply non-linear activations to raw Gaussian parameters.

        Parameters
        ----------
        pre_act_quats            : torch.Tensor
        pre_act_scales           : torch.Tensor
        pre_act_phase            : torch.Tensor or None
        pre_act_opacities        : torch.Tensor or None
        pre_act_plane_assignment : torch.Tensor or None
        step, max_step           : int or None

        Returns
        -------
        quats, scales, phase, opacities, plane_probs : torch.Tensor
        """
        scales = torch.exp(pre_act_scales)
        quats = F.normalize(pre_act_quats)
        phase = pre_act_phase % (2.0 * odak.pi)
        opacities = torch.sigmoid(pre_act_opacities)

        ste = StraightThroughEstimator()
        plane_probs = ste(pre_act_plane_assignment)

        return quats, scales, phase, opacities, plane_probs

    def save_gaussians(self, save_path: str):
        """
        Save Gaussian parameters to a ``.pth`` checkpoint.

        Parameters
        ----------
        save_path : str
                    Destination file path.
        """
        state_dict = {
            "pre_act_quats": self.pre_act_quats.cpu(),
            "means": self.means.cpu(),
            "pre_act_scales": self.pre_act_scales.cpu(),
            "colours": self.colours.cpu(),
            "pre_act_phase": self.pre_act_phase.cpu(),
            "pre_act_opacities": self.pre_act_opacities.cpu(),
            "pre_act_plane_assignment": self.pre_act_plane_assignment.cpu(),
        }
        torch.save(state_dict, save_path)
        print(f"Gaussians saved to {save_path}")


class Scene:
    """
    Wave-based rendering scene for complex-valued Gaussian splatting.

    Combines a set of :class:`Gaussians` with a camera model to produce
    holographic fields via tile-based splatting and band-limited
    angular-spectrum propagation.

    Parameters
    ----------
    gaussians : Gaussians
                The Gaussian primitives.
    args_prop : argparse.Namespace
                Must contain ``wavelengths``, ``pixel_pitch``,
                ``distances``, ``pad_size``, and ``aperture_size``.
    """

    def __init__(self, gaussians: Gaussians, args_prop):
        self.gaussians = gaussians
        self.args_prop = args_prop
        self.device = self.gaussians.device
        self.wavelengths = torch.tensor(
            args_prop.wavelengths, dtype=torch.float32, device=self.device
        )
        self.mean_2D_for_planeprob = None

    def __repr__(self):
        return f"<Scene with {len(self.gaussians)} Gaussians>"

    def compute_transmittance(self, alphas: torch.Tensor):
        """
        Compute transmittance from per-Gaussian alpha values.

        Parameters
        ----------
        alphas : torch.Tensor
                 Alpha (opacity × Gaussian) values ``(N, H, W)``.

        Returns
        -------
        transmittance : torch.Tensor
                        Cumulative transmittance ``(N, H, W)``.
        """
        _, H, W = alphas.shape
        S = torch.ones((1, H, W), device=alphas.device, dtype=alphas.dtype)
        one_minus_alphas = 1.0 - alphas
        one_minus_alphas = torch.cat((S, one_minus_alphas), dim=0)
        transmittance = torch.cumprod(one_minus_alphas, dim=0)[:-1]
        transmittance = torch.where(transmittance < 1e-4, 0.0, transmittance)
        return transmittance

    def compute_depth_values(self, camera: PerspectiveCamera):
        """
        Compute per-Gaussian depth values in camera space.

        Parameters
        ----------
        camera : PerspectiveCamera

        Returns
        -------
        z_vals : torch.Tensor
                 Depth values ``(N,)``.
        """
        means_3D = self.gaussians.means
        R = camera.R[0] if camera.R.dim() == 3 else camera.R
        T = camera.T[0] if camera.T.dim() == 2 else camera.T
        means_cam = means_3D @ R + T
        z_vals = means_cam[:, -1]
        return z_vals

    def calculate_gaussian_directions(self, means_3D, camera):
        """
        Compute unit direction vectors from camera centre to each Gaussian.

        Parameters
        ----------
        means_3D : torch.Tensor
                   3-D positions ``(N, 3)``.
        camera   : PerspectiveCamera

        Returns
        -------
        gaussian_dirs : torch.Tensor
                        Unit direction vectors ``(N, 3)``.
        """
        N = means_3D.shape[0]
        camera_centers = camera.get_camera_center().repeat(N, 1)
        gaussian_dirs = means_3D - camera_centers
        gaussian_dirs = F.normalize(gaussian_dirs)
        return gaussian_dirs

    def get_idxs_to_filter_and_sort(self, z_vals: torch.Tensor):
        """
        Sort Gaussians by depth and filter those behind the camera.

        Parameters
        ----------
        z_vals : torch.Tensor
                 Depth values ``(N,)``.

        Returns
        -------
        idxs : torch.Tensor
               Sorted indices with ``z >= 0``.
        """
        sorted_vals, indices = torch.sort(z_vals)
        mask = sorted_vals >= 0
        idxs = torch.masked_select(indices, mask).to(torch.int64)
        return idxs

    def splat(
        self,
        camera: PerspectiveCamera,
        means_3D: torch.Tensor,
        z_vals: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colours: torch.Tensor,
        phase: torch.Tensor,
        opacities: torch.Tensor,
        plane_probs: torch.Tensor,
        wavelengths: torch.Tensor,
        img_size: Tuple = (256, 256),
        tile_size: Tuple = (64, 64),
    ):
        """
        Multi-channel wave-based tile splatting and propagation.

        Parameters
        ----------
        camera       : PerspectiveCamera
        means_3D     : torch.Tensor ``(N, 3)``
        z_vals       : torch.Tensor ``(N,)``
        quats        : torch.Tensor ``(N, 4)``
        scales       : torch.Tensor ``(N, 3)``
        colours      : torch.Tensor ``(N, 3)``
        phase        : torch.Tensor ``(N, 3)``
        opacities    : torch.Tensor ``(N,)``
        plane_probs  : torch.Tensor ``(N, num_planes)``
        wavelengths  : torch.Tensor ``(C,)``
        img_size     : tuple of int
        tile_size    : tuple of int

        Returns
        -------
        hologram_complex : torch.Tensor
                           Complex hologram ``(C, H, W)``.
        plane_fields     : torch.Tensor
                           Per-plane fields ``(P, C, H, W)``.
        """
        W, H = img_size
        device = means_3D.device
        num_planes = plane_probs.shape[1]

        if isinstance(wavelengths, list):
            wavelengths = torch.tensor(wavelengths, device=device, dtype=torch.float32)

        R = camera.R
        fx, fy = camera.focal_length.flatten()
        px, py = camera.principal_point.flatten()

        if tile_size[0] <= 0 or tile_size[1] <= 0:
            tile_size = (64, 64)

        num_channels = len(wavelengths)

        cam_means_3D = camera.transform_world_to_camera_space(means_3D)

        means_2D = self.gaussians.compute_means_2D(cam_means_3D, fx, fy, px, py)
        self.mean_2D_for_planeprob = means_2D
        cov_2D = self.gaussians.compute_cov_2D(
            cam_means_3D, quats, scales, fx, fy, R, img_size
        )
        gaussian_bounds = self.gaussians.calculate_gaussian_bounds(
            means_2D, cov_2D, img_size
        )
        plane_fields = torch.zeros(
            (num_planes, num_channels, H, W), dtype=torch.complex64, device=device
        )

        tile_w, tile_h = tile_size
        x_tiles = math.ceil(W / tile_w)
        y_tiles = math.ceil(H / tile_h)

        for y_idx in range(y_tiles):
            for x_idx in range(x_tiles):
                x = x_idx * tile_w
                y = y_idx * tile_h
                actual_tile_w = min(tile_w, W - x)
                actual_tile_h = min(tile_h, H - y)
                x_min, y_min = x, y
                x_max = x + actual_tile_w - 1
                y_max = y + actual_tile_h - 1

                in_x_range = (gaussian_bounds[:, 0] <= x_max) & (
                    gaussian_bounds[:, 2] >= x_min
                )
                in_y_range = (gaussian_bounds[:, 1] <= y_max) & (
                    gaussian_bounds[:, 3] >= y_min
                )
                gaussian_indices = torch.where(in_x_range & in_y_range)[0]

                tile_plane_fields = self.splat_tile(
                    R,
                    fx,
                    fy,
                    px,
                    py,
                    cam_means_3D,
                    z_vals,
                    quats,
                    scales,
                    colours,
                    phase,
                    opacities,
                    plane_probs,
                    x,
                    y,
                    (actual_tile_w, actual_tile_h),
                    gaussian_indices,
                    img_size,
                    wavelengths,
                )
                plane_fields[
                    :, :, y : y + actual_tile_h, x : x + actual_tile_w
                ] += tile_plane_fields

        hologram_complex_planes = []
        for p in range(num_planes):
            plane_hologram = []
            for c, plane_field_c in enumerate(plane_fields[p]):
                wavelength_val = float(wavelengths[c].cpu().item())
                hologram_complex_c = bandlimited_angular_spectrum_propagation(
                    plane_field_c,
                    wavelength=wavelength_val,
                    pixel_pitch=self.args_prop.pixel_pitch,
                    distance=-self.args_prop.distances[p],
                    size=self.args_prop.pad_size,
                    aperture_size=self.args_prop.aperture_size,
                )
                plane_hologram.append(hologram_complex_c)
            hologram_complex_planes.append(torch.stack(plane_hologram, dim=0))

        hologram_complex = sum(hologram_complex_planes)
        return hologram_complex, plane_fields

    def splat_tile(
        self,
        R,
        fx,
        fy,
        px,
        py,
        cam_means_3D,
        z_vals,
        quats,
        scales,
        colours,
        phase,
        opacities,
        plane_probs,
        tile_x,
        tile_y,
        tile_size,
        gaussian_indices,
        img_size,
        wavelengths,
    ):
        """
        Render a single tile for all planes (pure PyTorch).

        Parameters
        ----------
        R              : torch.Tensor
                         Rotation matrix.
        fx, fy, px, py : float or torch.Tensor
                         Camera intrinsics.
        cam_means_3D   : torch.Tensor ``(N, 3)``
        z_vals         : torch.Tensor ``(N,)``
        quats, scales, colours, phase, opacities : torch.Tensor
        plane_probs    : torch.Tensor ``(N, P)``
        tile_x, tile_y : int
        tile_size      : tuple of int
                         ``(tile_w, tile_h)`` for this tile.
        gaussian_indices : torch.Tensor
                         Indices of Gaussians overlapping this tile.
        img_size       : tuple of int
                         Full image ``(W, H)``.
        wavelengths    : torch.Tensor ``(C,)``

        Returns
        -------
        result : torch.Tensor
                 ``(P, C, tile_h, tile_w)`` complex field for this tile.
        """
        device = cam_means_3D.device
        W, H = img_size
        tile_w, tile_h = tile_size
        num_planes = plane_probs.shape[1]

        tile_plane_fields = []
        for _ in range(num_planes):
            tile_plane_fields.append(
                torch.zeros(
                    (len(wavelengths), tile_h, tile_w),
                    device=device,
                    dtype=torch.complex64,
                )
            )

        if gaussian_indices.numel() == 0:
            return torch.stack(tile_plane_fields, dim=0)

        xs, ys = torch.meshgrid(
            torch.arange(tile_x, tile_x + tile_w, device=device),
            torch.arange(tile_y, tile_y + tile_h, device=device),
            indexing="xy",
        )
        points_2D = torch.stack([xs.flatten(), ys.flatten()], dim=1)

        tile_means_3D = cam_means_3D[gaussian_indices]
        valid_mask = (tile_means_3D[:, 2] > self.gaussians.NEAR_PLANE) & (
            tile_means_3D[:, 2] < self.gaussians.FAR_PLANE
        )
        if not valid_mask.any():
            return torch.stack(tile_plane_fields, dim=0)

        tile_means_3D = tile_means_3D[valid_mask]
        valid_gaussian_indices = gaussian_indices[valid_mask]

        tile_means_2D = self.gaussians.compute_means_2D(tile_means_3D, fx, fy, px, py)

        tile_plane_probs = plane_probs[valid_gaussian_indices]

        tile_means_2D = tile_means_2D.unsqueeze(1)
        diff = points_2D.unsqueeze(0) - tile_means_2D

        tile_cov_2D = self.gaussians.compute_cov_2D(
            tile_means_3D,
            quats[valid_gaussian_indices],
            scales[valid_gaussian_indices],
            fx,
            fy,
            R,
            img_size,
        )
        cov_inv = self.gaussians.invert_cov_2D(tile_cov_2D)

        term = torch.bmm(diff, cov_inv)
        term = (term * diff).sum(dim=-1)
        term = term.view(-1, tile_h, tile_w)

        gauss_exp = torch.exp(-0.5 * term)
        tile_opacities = opacities[valid_gaussian_indices].view(-1, 1, 1)
        base_alphas = tile_opacities * gauss_exp

        for plane_idx in range(num_planes):
            plane_mask = tile_plane_probs[:, plane_idx].view(-1, 1, 1)
            plane_alphas = base_alphas * plane_mask
            plane_alphas_reshaped = plane_alphas.reshape(-1, tile_h, tile_w)
            transmittance = self.compute_transmittance(plane_alphas_reshaped)

            for c in range(len(wavelengths)):
                colours_c = colours[valid_gaussian_indices, c].view(-1, 1, 1)
                phase_c = phase[valid_gaussian_indices, c].view(-1, 1, 1)
                tile_plane_fields[plane_idx][c] = torch.sum(
                    colours_c * plane_alphas * transmittance * torch.exp(1j * phase_c),
                    dim=0,
                )

        result = torch.stack(tile_plane_fields, dim=0)
        return result

    def render(
        self,
        camera: PerspectiveCamera,
        img_size: Tuple = (-1, -1),
        bg_colour: Tuple = (0.0, 0.0, 0.0),
        tile_size: Tuple = (64, 64),
        step=-1,
        max_step=-1,
    ):
        """
        Render a complex hologram from the current Gaussians.

        Parameters
        ----------
        camera    : PerspectiveCamera
        img_size  : tuple of int
                    ``(W, H)``.
        bg_colour : tuple of float
                    Background colour (unused in wave rendering).
        tile_size : tuple of int
                    Tile dimensions for splatting.
        step      : int
                    Current training step (for scheduled activations).
        max_step  : int
                    Maximum training step.

        Returns
        -------
        hologram_complex : torch.Tensor
                           Complex hologram ``(C, H, W)``.
        plane_field      : torch.Tensor
                           Per-plane complex fields
                           ``(P, C, H, W)``.
        """
        z_vals = self.compute_depth_values(camera)

        cam_means_3D = camera.transform_world_to_camera_space(self.gaussians.means)
        visible_mask = (cam_means_3D[:, 2] > self.gaussians.NEAR_PLANE) & (
            cam_means_3D[:, 2] < self.gaussians.FAR_PLANE
        )
        valid_indices = torch.where(visible_mask)[0]

        idxs = self.get_idxs_to_filter_and_sort(z_vals[valid_indices])
        idxs = valid_indices[idxs]

        pre_act_quats = self.gaussians.pre_act_quats[idxs]
        pre_act_scales = self.gaussians.pre_act_scales[idxs]
        pre_act_phase = self.gaussians.pre_act_phase[idxs]
        pre_act_opacities = self.gaussians.pre_act_opacities[idxs]
        pre_act_plane_assignment = self.gaussians.pre_act_plane_assignment[idxs]

        z_vals = z_vals[idxs]
        means_3D = self.gaussians.means[idxs]
        colours = self.gaussians.colours[idxs]

        quats, scales, phase_val, opacities, plane_probs = (
            self.gaussians.apply_activations(
                pre_act_quats,
                pre_act_scales,
                pre_act_phase,
                pre_act_opacities,
                pre_act_plane_assignment,
                step,
                max_step,
            )
        )

        hologram_complex, plane_field = self.splat(
            camera,
            means_3D,
            z_vals,
            quats,
            scales,
            colours,
            phase_val,
            opacities,
            plane_probs,
            self.wavelengths,
            img_size,
            tile_size,
        )

        return hologram_complex, plane_field
