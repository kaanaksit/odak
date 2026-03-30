"""
Unit tests for cvd_loss function.

Tests the combined CVD loss computation including local contrast and color information losses.
"""

import torch
import sys
import pytest


def test_cvd_loss_basic():
    """Test basic functionality with simple images."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    # Create two images
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    
    # All losses should be positive
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(lc_loss, torch.Tensor)
    assert isinstance(ci_loss, torch.Tensor)
    assert total_loss.item() > 0.0
    assert lc_loss.item() > 0.0
    assert ci_loss.item() > 0.0


def test_cvd_loss_default_weights():
    """Test with default weights (alpha=15.0, beta=1.0)."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    
    # Total loss should be: alpha * lc_loss + beta * ci_loss
    expected_total = 15.0 * lc_loss.item() + 1.0 * ci_loss.item()
    
    assert abs(total_loss.item() - expected_total) < 1e-5


def test_cvd_loss_custom_weights():
    """Test with custom weights."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    total_loss, lc_loss, ci_loss = cvd_loss(
        image, simulated_image,
        alpha=10.0,
        beta=2.0
    )
    
    # Total loss should be: alpha * lc_loss + beta * ci_loss
    expected_total = 10.0 * lc_loss.item() + 2.0 * ci_loss.item()
    
    assert abs(total_loss.item() - expected_total) < 1e-5


def test_cvd_loss_alpha_zero():
    """Test with alpha=0 (only color information loss)."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image, alpha=0.0, beta=1.0)
    
    # Total loss should equal ci_loss when alpha=0
    assert abs(total_loss.item() - ci_loss.item()) < 1e-5


def test_cvd_loss_beta_zero():
    """Test with beta=0 (only local contrast loss)."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image, alpha=1.0, beta=0.0)
    
    # Total loss should equal lc_loss when beta=0
    assert abs(total_loss.item() - lc_loss.item()) < 1e-5


def test_cvd_loss_identical_images():
    """Test with identical images (loss should be zero)."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image.clone()
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    
    assert total_loss.item() == 0.0
    assert lc_loss.item() == 0.0
    assert ci_loss.item() == 0.0


def test_cvd_loss_gradient_flow():
    """Test that gradients flow through the combined loss."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64, requires_grad=True)
    simulated_image = image.detach() * 0.8
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    total_loss.backward()
    
    assert image.grad is not None
    assert image.grad.shape == image.shape


def test_cvd_loss_different_parameters():
    """Test with different parameter configurations."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    # Default parameters
    total_loss_default, _, _ = cvd_loss(image, simulated_image)
    
    # Custom patch sizes and kernel sizes
    total_loss_custom, _, _ = cvd_loss(
        image, simulated_image,
        lc_patch_size=8,
        lc_stride=4,
        ci_kernel_size=7,
        ci_sigma=2.0
    )
    
    # Both should be valid positive losses
    assert total_loss_default.item() > 0.0
    assert total_loss_custom.item() > 0.0


def test_cvd_loss_minimum_image_size():
    """Test with minimum allowed image size (4x4)."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(1, 3, 4, 4)
    simulated_image = image * 0.9
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    
    assert total_loss.item() >= 0.0
    assert lc_loss.item() >= 0.0
    assert ci_loss.item() >= 0.0


def test_cvd_loss_large_batch():
    """Test with larger batch size."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(8, 3, 64, 64)
    simulated_image = image * 0.75
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    
    assert total_loss.item() > 0.0
    assert lc_loss.item() > 0.0
    assert ci_loss.item() > 0.0


def test_cvd_loss_type_error_image():
    """Test that TypeError is raised for non-tensor image."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    with pytest.raises(TypeError):
        cvd_loss("not_a_tensor", torch.rand(2, 3, 64, 64))


def test_cvd_loss_type_error_simulated():
    """Test that TypeError is raised for non-tensor simulated_image."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    with pytest.raises(TypeError):
        cvd_loss(torch.rand(2, 3, 64, 64), "not_a_tensor")


def test_cvd_loss_shape_mismatch():
    """Test that ValueError is raised for shape mismatch."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = torch.rand(2, 3, 32, 32)
    
    with pytest.raises(ValueError):
        cvd_loss(image, simulated_image)


def test_cvd_loss_image_too_small():
    """Test that ValueError is raised for images smaller than 4x4."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(1, 3, 3, 3)
    simulated_image = image * 0.9
    
    with pytest.raises(ValueError):
        cvd_loss(image, simulated_image)


def test_cvd_loss_wrong_dimensions():
    """Test that ValueError is raised for wrong tensor dimensions."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(3, 64, 64)  # 3D instead of 4D
    simulated_image = image * 0.9
    
    with pytest.raises(ValueError):
        cvd_loss(image, simulated_image)


def test_cvd_loss_cuda():
    """Test compatibility with CUDA tensors if available."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    image = torch.rand(2, 3, 64, 64, device='cuda')
    simulated_image = image.detach() * 0.8
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    
    assert total_loss.device.type == 'cuda'
    assert lc_loss.device.type == 'cuda'
    assert ci_loss.device.type == 'cuda'
    assert total_loss.item() > 0.0


def test_cvd_loss_grayscale_images():
    """Test with grayscale images (1 channel)."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 1, 64, 64)
    simulated_image = image * 0.8
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    
    assert total_loss.item() > 0.0
    assert lc_loss.item() > 0.0
    assert ci_loss.item() > 0.0


def test_cvd_loss_extreme_weights():
    """Test with extreme weight combinations."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    # Very high alpha
    total_loss_high_alpha, lc_loss_alpha, _ = cvd_loss(image, simulated_image, alpha=100.0, beta=1.0)
    
    # Very high beta
    total_loss_high_beta, _, ci_loss_beta = cvd_loss(image, simulated_image, alpha=1.0, beta=100.0)
    
    # Both should be valid and positive
    assert total_loss_high_alpha.item() > 0.0
    assert total_loss_high_beta.item() > 0.0
    
    # High alpha should primarily scale lc_loss, high beta should scale ci_loss
    # The comparison depends on which base loss is larger, so just verify both work
    expected_alpha = 100.0 * lc_loss_alpha.item() + 1.0 * ci_loss_beta.item()  # approximate
    assert abs(total_loss_high_alpha.item() - 100.0 * lc_loss_alpha.item()) < 1.0 + ci_loss_beta.item()


def test_cvd_loss_different_sigma_values():
    """Test with different sigma values for color information loss."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    # Small sigma
    loss_small_sigma, _, _ = cvd_loss(
        image, simulated_image,
        ci_sigma=0.5
    )
    
    # Large sigma
    loss_large_sigma, _, _ = cvd_loss(
        image, simulated_image,
        ci_sigma=3.0
    )
    
    # Both should be valid positive losses
    assert loss_small_sigma.item() > 0.0
    assert loss_large_sigma.item() > 0.0


def test_cvd_loss_return_values_consistency():
    """Test that return values are always in correct order."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    # Run multiple times to ensure consistency
    for _ in range(5):
        total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
        
        # Verify relationship
        expected_total = 15.0 * lc_loss.item() + 1.0 * ci_loss.item()
        assert abs(total_loss.item() - expected_total) < 1e-5


def test_cvd_loss_inverted_colors():
    """Test with inverted colors (extreme case)."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = 1.0 - image  # Invert
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    
    # Total loss should be positive due to color information differences
    # Note: inverted colors preserve contrast, so lc_loss may be 0
    assert total_loss.item() > 0.0
    assert ci_loss.item() > 0.0  # Color information loss should be positive
    assert lc_loss.item() >= 0.0  # Contrast may be preserved


def test_cvd_loss_very_similar_images():
    """Test with very similar images."""
    from odak.learn.perception.cvd_loss_functions import cvd_loss
    
    image = torch.rand(2, 3, 64, 64)
    # Very slight difference
    simulated_image = image * 0.999
    
    total_loss, lc_loss, ci_loss = cvd_loss(image, simulated_image)
    
    # Loss should be very small but positive
    assert total_loss.item() > 0.0
    assert total_loss.item() < 0.1  # Should be very small
    assert lc_loss.item() < 0.1
    assert ci_loss.item() < 0.1


def run_all_tests():
    """Run all tests and return True if all pass."""
    try:
        test_cvd_loss_basic()
        test_cvd_loss_default_weights()
        test_cvd_loss_custom_weights()
        test_cvd_loss_alpha_zero()
        test_cvd_loss_beta_zero()
        test_cvd_loss_identical_images()
        test_cvd_loss_gradient_flow()
        test_cvd_loss_different_parameters()
        test_cvd_loss_minimum_image_size()
        test_cvd_loss_large_batch()
        test_cvd_loss_type_error_image()
        test_cvd_loss_type_error_simulated()
        test_cvd_loss_shape_mismatch()
        test_cvd_loss_image_too_small()
        test_cvd_loss_wrong_dimensions()
        test_cvd_loss_grayscale_images()
        test_cvd_loss_extreme_weights()
        test_cvd_loss_different_sigma_values()
        test_cvd_loss_return_values_consistency()
        test_cvd_loss_inverted_colors()
        test_cvd_loss_very_similar_images()
        
        # Test CUDA if available
        if torch.cuda.is_available():
            test_cvd_loss_cuda()
        
        print("All tests passed!")
        return True
    except AssertionError as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
