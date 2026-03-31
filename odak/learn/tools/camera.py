import torch


class PerspectiveCamera:
    """
    A lightweight perspective camera model.

    Stores camera intrinsics and extrinsics and provides
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

        Follows the convention: ``X_cam = X_world @ R + T``.

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
