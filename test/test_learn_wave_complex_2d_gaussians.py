import sys
from argparse import Namespace

import torch


def test_gaussians_2d_init():
    """Test complex_2d_gaussians initialization and parameter shapes."""
    from odak.learn.wave.complex_gaussians import complex_2d_gaussians

    num_points = 64
    img_size = (128, 96)

    gaussians = complex_2d_gaussians(
        num_points=num_points,
        img_size=img_size,
        device="cpu",
    )

    assert (
        len(gaussians) == num_points
    ), f"Expected {num_points} gaussians, got {len(gaussians)}"
    assert gaussians.means_2d.shape == (num_points, 2)
    assert gaussians.pre_act_scales.shape == (num_points, 2)
    assert gaussians.pre_act_rotation.shape == (num_points,)
    assert gaussians.colours.shape == (num_points, 3)
    assert gaussians.pre_act_phase.shape == (num_points, 3)
    assert gaussians.pre_act_opacities.shape == (num_points,)

    print("  complex_2d_gaussians init: PASSED")


def test_gaussians_2d_activations():
    """Test that apply_activations produces valid outputs."""
    from odak.learn.wave.complex_gaussians import complex_2d_gaussians

    num_points = 32
    img_size = (64, 64)
    gaussians = complex_2d_gaussians(
        num_points=num_points,
        img_size=img_size,
        device="cpu",
    )

    scales, rotation, phase, opacities, means_2d = gaussians.apply_activations()

    # Scales should be positive.
    assert (scales > 0).all(), "Scales should be positive"

    # Phase should be in [0, 2*pi).
    assert (phase >= 0).all() and (
        phase < 2 * torch.pi + 1e-5
    ).all(), "Phase should be in [0, 2*pi)"

    # Opacities should be in (0, 1) (sigmoid output).
    assert (opacities > 0).all() and (
        opacities < 1
    ).all(), "Opacities should be in (0, 1)"

    # Means should map into the image extent via the tanh activation.
    W, H = img_size
    assert (means_2d[:, 0] >= 0).all() and (means_2d[:, 0] <= W).all()
    assert (means_2d[:, 1] >= 0).all() and (means_2d[:, 1] <= H).all()

    # merge_opacity fixes opacities to one.
    merged = complex_2d_gaussians(
        num_points=num_points,
        img_size=img_size,
        device="cpu",
        merge_opacity=True,
    )
    _, _, _, merged_opacities, _ = merged.apply_activations()
    assert torch.allclose(
        merged_opacities, torch.ones(num_points)
    ), "merge_opacity should fix opacities to one"

    print("  complex_2d_gaussians activations: PASSED")


def test_covariance_2d():
    """Test 2D covariance element computation and inversion roundtrip."""
    from odak.learn.wave.complex_gaussians import complex_2d_gaussians

    num_points = 16
    gaussians = complex_2d_gaussians(
        num_points=num_points,
        img_size=(64, 64),
        device="cpu",
    )

    scales, rotation, _, _, _ = gaussians.apply_activations()
    cov_00, cov_01, cov_11 = gaussians.compute_cov_2D(scales, rotation)

    assert cov_00.shape == (num_points,)
    # Covariance must be positive-definite: positive diagonal and determinant.
    det = cov_00 * cov_11 - cov_01 * cov_01
    assert (cov_00 > 0).all() and (cov_11 > 0).all(), "Diagonal must be positive"
    assert (det > 0).all(), "Covariance determinant must be positive"

    # Inversion roundtrip: C @ C^-1 == I.
    inv_00, inv_01, inv_11 = gaussians.invert_cov_2D(cov_00, cov_01, cov_11)
    prod_00 = cov_00 * inv_00 + cov_01 * inv_01
    prod_11 = cov_01 * inv_01 + cov_11 * inv_11
    prod_01 = cov_00 * inv_01 + cov_01 * inv_11
    assert torch.allclose(prod_00, torch.ones(num_points), atol=1e-4)
    assert torch.allclose(prod_11, torch.ones(num_points), atol=1e-4)
    assert torch.allclose(prod_01, torch.zeros(num_points), atol=1e-4)

    print("  2D covariance computation: PASSED")


def test_scene_2d_render_shapes():
    """Test that Scene2D.render produces a correctly shaped complex hologram."""
    from odak.learn.wave.complex_gaussians import complex_2d_gaussians, Scene2D

    num_points = 64
    img_size = (64, 64)
    wavelengths = [633e-9, 532e-9, 450e-9]
    pad_size = [128, 128]

    args = Namespace(wavelengths=wavelengths, pad_size=pad_size)

    gaussians = complex_2d_gaussians(
        num_points=num_points,
        img_size=img_size,
        device="cpu",
    )
    scene = Scene2D(gaussians, args)

    hologram = scene.render(img_size=img_size)

    assert (
        hologram.dtype == torch.complex64 or hologram.dtype == torch.complex128
    ), f"Hologram should be complex, got {hologram.dtype}"
    assert hologram.shape == (
        len(wavelengths),
        pad_size[0],
        pad_size[1],
    ), f"Unexpected hologram shape {hologram.shape}"

    print("  Scene2D render shapes: PASSED")


def test_render_2d_differentiable():
    """Test that the rendered hologram is differentiable w.r.t. the parameters."""
    from odak.learn.wave.complex_gaussians import complex_2d_gaussians, Scene2D

    img_size = (32, 32)
    args = Namespace(wavelengths=[633e-9], pad_size=[64, 64])

    gaussians = complex_2d_gaussians(
        num_points=16,
        img_size=img_size,
        device="cpu",
    )
    gaussians.make_trainable()
    scene = Scene2D(gaussians, args)

    hologram = scene.render(img_size=img_size)
    loss = hologram.abs().mean()
    loss.backward()

    assert gaussians.colours.grad is not None, "Colours should receive gradients"
    assert (
        gaussians.pre_act_scales.grad is not None
    ), "Scales should receive gradients"

    print("  Scene2D differentiability: PASSED")


def test_save_load_roundtrip_2d():
    """Test that save_gaussians and load_gaussians roundtrip correctly."""
    import os
    import tempfile

    from odak.learn.wave.complex_gaussians import complex_2d_gaussians

    img_size = (64, 64)
    gaussians = complex_2d_gaussians(
        num_points=16,
        img_size=img_size,
        device="cpu",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_gaussians_2d.pth")
        gaussians.save_gaussians(save_path)

        loaded = complex_2d_gaussians(
            num_points=16,
            img_size=img_size,
            device="cpu",
        )
        loaded.load_gaussians(save_path)

        assert torch.allclose(gaussians.means_2d, loaded.means_2d)
        assert torch.allclose(gaussians.colours, loaded.colours)
        assert torch.allclose(gaussians.pre_act_scales, loaded.pre_act_scales)
        assert torch.allclose(gaussians.pre_act_rotation, loaded.pre_act_rotation)
        assert torch.allclose(gaussians.pre_act_phase, loaded.pre_act_phase)

    print("  Save/load roundtrip: PASSED")


def main():
    print("Running complex_2d_gaussians unit tests...")
    test_gaussians_2d_init()
    test_gaussians_2d_activations()
    test_covariance_2d()
    test_scene_2d_render_shapes()
    test_render_2d_differentiable()
    test_save_load_roundtrip_2d()
    print("All tests PASSED.")
    assert True


if __name__ == "__main__":
    sys.exit(main())
