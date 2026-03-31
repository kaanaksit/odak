import sys
from argparse import Namespace

import torch


def test_quaternion_to_rotation_matrix():
    """Test that quaternion_to_rotation_matrix produces valid rotation matrices."""
    from odak.learn.wave.complex_gaussians import quaternion_to_rotation_matrix

    # Identity quaternion (w=1, x=0, y=0, z=0) should give identity matrix
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    R = quaternion_to_rotation_matrix(identity_quat)
    assert torch.allclose(
        R.squeeze(0), torch.eye(3), atol=1e-6
    ), "Identity quaternion did not produce identity matrix"

    # 180-degree rotation around z-axis: (w=0, x=0, y=0, z=1)
    z180_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    R = quaternion_to_rotation_matrix(z180_quat)
    expected = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(
        R.squeeze(0), expected, atol=1e-5
    ), "180-degree z rotation incorrect"

    # Batch test: rotation matrices should be orthogonal (R @ R^T = I)
    quats = torch.randn(16, 4)
    Rs = quaternion_to_rotation_matrix(quats)
    identity = torch.eye(3).unsqueeze(0).expand(16, -1, -1)
    product = torch.bmm(Rs, Rs.transpose(1, 2))
    assert torch.allclose(
        product, identity, atol=1e-5
    ), "Rotation matrices are not orthogonal"

    # Determinant should be +1
    dets = torch.det(Rs)
    assert torch.allclose(
        dets, torch.ones(16), atol=1e-5
    ), "Rotation matrix determinants are not +1"

    print("  quaternion_to_rotation_matrix: PASSED")


def test_perspective_camera():
    """Test PerspectiveCamera world-to-camera transform and camera center."""
    from odak.learn.wave.complex_gaussians import PerspectiveCamera

    R = torch.eye(3).unsqueeze(0)
    T = torch.tensor([[1.0, 2.0, 3.0]])
    focal_length = torch.tensor([500.0, 500.0])
    principal_point = torch.tensor([128.0, 128.0])

    camera = PerspectiveCamera(R, T, focal_length, principal_point)

    # With identity R: X_cam = X_world @ I + T = X_world + T
    points = torch.tensor([[0.0, 0.0, 0.0]])
    cam_points = camera.transform_world_to_camera_space(points)
    assert torch.allclose(
        cam_points, T, atol=1e-6
    ), "Camera transform with identity R incorrect"

    # Camera center: -T @ R^T = -T for identity R
    center = camera.get_camera_center()
    assert torch.allclose(
        center, -T, atol=1e-6
    ), "Camera center with identity R incorrect"

    # Non-identity rotation
    angle = torch.tensor(torch.pi / 4)
    R_rot = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    camera2 = PerspectiveCamera(R_rot, T, focal_length, principal_point)
    center2 = camera2.get_camera_center()
    # Camera center should satisfy: center @ R + T = 0
    check = center2 @ R_rot.squeeze(0) + T
    assert torch.allclose(
        check, torch.zeros(1, 3), atol=1e-5
    ), "Camera center does not satisfy center @ R + T = 0"

    print("  PerspectiveCamera: PASSED")


def test_gaussians_random_init():
    """Test Gaussians random initialization and basic operations."""
    from odak.learn.wave.complex_gaussians import Gaussians

    num_points = 64
    num_planes = 2
    args = Namespace(num_planes=num_planes)

    gaussians = Gaussians(
        init_type="random",
        device="cpu",
        num_points=num_points,
        args_prop=args,
    )

    assert (
        len(gaussians) == num_points
    ), f"Expected {num_points} gaussians, got {len(gaussians)}"
    assert gaussians.means.shape == (num_points, 3)
    assert gaussians.pre_act_quats.shape == (num_points, 4)
    assert gaussians.pre_act_scales.shape == (num_points, 3)
    assert gaussians.colours.shape == (num_points, 3)
    assert gaussians.pre_act_phase.shape == (num_points, 3)
    assert gaussians.pre_act_opacities.shape == (num_points,)
    assert gaussians.pre_act_plane_assignment.shape == (num_points, num_planes)

    print("  Gaussians random init: PASSED")


def test_gaussians_activations():
    """Test that apply_activations produces valid outputs."""
    from odak.learn.wave.complex_gaussians import Gaussians

    num_points = 32
    num_planes = 3
    args = Namespace(num_planes=num_planes)

    gaussians = Gaussians(
        init_type="random",
        device="cpu",
        num_points=num_points,
        args_prop=args,
    )

    quats, scales, phase, opacities, plane_probs = Gaussians.apply_activations(
        gaussians.pre_act_quats,
        gaussians.pre_act_scales,
        gaussians.pre_act_phase,
        gaussians.pre_act_opacities,
        gaussians.pre_act_plane_assignment,
    )

    # Quaternions should be unit-norm
    quat_norms = quats.norm(dim=1)
    assert torch.allclose(
        quat_norms, torch.ones(num_points), atol=1e-5
    ), "Activated quaternions are not unit-norm"

    # Scales should be positive (exp of pre_act)
    assert (scales > 0).all(), "Scales should be positive"

    # Phase should be in [0, 2*pi)
    assert (phase >= 0).all() and (
        phase < 2 * torch.pi + 1e-5
    ).all(), "Phase should be in [0, 2*pi)"

    # Opacities should be in (0, 1) (sigmoid output)
    assert (opacities > 0).all() and (
        opacities < 1
    ).all(), "Opacities should be in (0, 1)"

    # Plane probs should be one-hot (STE)
    row_sums = plane_probs.sum(dim=1)
    assert torch.allclose(
        row_sums, torch.ones(num_points), atol=1e-6
    ), "Plane probs rows should sum to 1 (one-hot)"

    print("  Gaussians activations: PASSED")


def test_covariance_computation():
    """Test 3D and 2D covariance computation shapes and symmetry."""
    from odak.learn.wave.complex_gaussians import Gaussians

    num_points = 16
    args = Namespace(num_planes=1)

    gaussians = Gaussians(
        init_type="random",
        device="cpu",
        num_points=num_points,
        args_prop=args,
    )

    quats = torch.nn.functional.normalize(gaussians.pre_act_quats, dim=1)
    scales = torch.exp(gaussians.pre_act_scales)

    # 3D covariance
    cov_3d = gaussians.compute_cov_3D(quats, scales)
    assert cov_3d.shape == (
        num_points,
        3,
        3,
    ), f"Expected cov_3D shape ({num_points}, 3, 3), got {cov_3d.shape}"

    # Covariance should be symmetric
    assert torch.allclose(
        cov_3d, cov_3d.transpose(1, 2), atol=1e-5
    ), "3D covariance matrices should be symmetric"

    # 2D covariance
    R = torch.eye(3).unsqueeze(0)
    cam_means = gaussians.means.clone()
    cam_means[:, 2] = cam_means[:, 2].abs() + 2.0  # ensure positive z

    fx, fy = 500.0, 500.0
    img_size = (256, 256)

    cov_2d = gaussians.compute_cov_2D(cam_means, quats, scales, fx, fy, R, img_size)
    assert cov_2d.shape == (
        num_points,
        2,
        2,
    ), f"Expected cov_2D shape ({num_points}, 2, 2), got {cov_2d.shape}"

    # 2D covariance should be symmetric
    assert torch.allclose(
        cov_2d, cov_2d.transpose(1, 2), atol=1e-4
    ), "2D covariance matrices should be symmetric"

    print("  Covariance computation: PASSED")


def test_projection():
    """Test 3D to 2D projection and inversion."""
    from odak.learn.wave.complex_gaussians import Gaussians

    args = Namespace(num_planes=1)
    gaussians = Gaussians(
        init_type="random",
        device="cpu",
        num_points=8,
        args_prop=args,
    )

    # Points in front of camera
    cam_means = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [1.0, 1.0, 10.0],
            [-1.0, 2.0, 3.0],
        ]
    )

    fx, fy = 500.0, 500.0
    px, py = 128.0, 128.0

    means_2d = gaussians.compute_means_2D(cam_means, fx, fy, px, py)
    assert means_2d.shape == (3, 2), f"Expected shape (3, 2), got {means_2d.shape}"

    # Point at optical axis (0,0,z) should project near principal point
    assert torch.allclose(
        means_2d[0], torch.tensor([px, py]), atol=1.0
    ), "On-axis point should project near principal point"

    # Test cov_2D inversion roundtrip
    cov = torch.tensor(
        [
            [[2.0, 0.5], [0.5, 3.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    cov_inv = Gaussians.invert_cov_2D(cov)
    product = torch.bmm(cov, cov_inv)
    identity = torch.eye(2).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(
        product, identity, atol=1e-4
    ), "cov @ cov_inv should be identity"

    print("  Projection and inversion: PASSED")


def test_scene_render_shapes():
    """Test that Scene.render produces correct output shapes."""
    from odak.learn.wave.complex_gaussians import Gaussians, PerspectiveCamera, Scene

    num_points = 32
    num_planes = 2
    img_size = (64, 64)
    wavelengths = [633e-9, 532e-9, 450e-9]

    args = Namespace(
        num_planes=num_planes,
        wavelengths=wavelengths,
        pixel_pitch=8e-6,
        distances=[0.01, 0.02],
        pad_size=list(img_size),
        aperture_size=-1,
    )

    gaussians = Gaussians(
        init_type="random",
        device="cpu",
        num_points=num_points,
        args_prop=args,
    )

    scene = Scene(gaussians, args)

    R = torch.eye(3).unsqueeze(0)
    T = torch.tensor([[0.0, 0.0, 0.0]])
    focal_length = torch.tensor([500.0, 500.0])
    principal_point = torch.tensor([32.0, 32.0])
    camera = PerspectiveCamera(R, T, focal_length, principal_point)

    hologram, plane_field = scene.render(
        camera=camera,
        img_size=img_size,
        tile_size=(32, 32),
    )

    W, H = img_size
    num_channels = len(wavelengths)

    assert (
        hologram.dtype == torch.complex64 or hologram.dtype == torch.complex128
    ), f"Hologram should be complex, got {hologram.dtype}"
    assert hologram.shape == (
        num_channels,
        H,
        W,
    ), f"Expected hologram shape ({num_channels}, {H}, {W}), got {hologram.shape}"
    assert plane_field.shape == (
        num_planes,
        num_channels,
        H,
        W,
    ), f"Expected plane_field shape ({num_planes}, {num_channels}, {H}, {W}), got {plane_field.shape}"

    print("  Scene render shapes: PASSED")


def test_save_load_roundtrip(tmp_path=None):
    """Test that save_gaussians and _load_gaussians roundtrip correctly."""
    import os
    import tempfile

    from odak.learn.wave.complex_gaussians import Gaussians

    args = Namespace(num_planes=2)
    gaussians = Gaussians(
        init_type="random",
        device="cpu",
        num_points=16,
        args_prop=args,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_gaussians.pth")
        gaussians.save_gaussians(save_path)

        loaded = Gaussians(
            init_type="gaussians",
            device="cpu",
            load_path=save_path,
            args_prop=args,
        )

        assert torch.allclose(
            gaussians.means, loaded.means
        ), "Means mismatch after save/load"
        assert torch.allclose(
            gaussians.colours, loaded.colours
        ), "Colours mismatch after save/load"
        assert torch.allclose(
            gaussians.pre_act_quats, loaded.pre_act_quats
        ), "Quaternions mismatch after save/load"
        assert torch.allclose(
            gaussians.pre_act_scales, loaded.pre_act_scales
        ), "Scales mismatch after save/load"
        assert torch.allclose(
            gaussians.pre_act_phase, loaded.pre_act_phase
        ), "Phase mismatch after save/load"

    print("  Save/load roundtrip: PASSED")


def main():
    print("Running complex_gaussians unit tests...")
    test_quaternion_to_rotation_matrix()
    test_perspective_camera()
    test_gaussians_random_init()
    test_gaussians_activations()
    test_covariance_computation()
    test_projection()
    test_scene_render_shapes()
    test_save_load_roundtrip()
    print("All tests PASSED.")
    assert True


if __name__ == "__main__":
    sys.exit(main())
