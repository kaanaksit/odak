"""
Unit tests for local_contrast_loss function.

Tests the local contrast loss computation between original and simulated images.
"""

import torch
import sys
import pytest


def test_local_contrast_loss_basic():
    """Test basic functionality with simple images."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    # Create two identical images
    image = torch.ones(2, 3, 64, 64)
    simulated_image = torch.ones(2, 3, 64, 64)
    
    loss = local_contrast_loss(image, simulated_image)
    
    # Identical images should have zero loss
    assert loss.shape == torch.Size([])
    assert loss.item() == 0.0


def test_local_contrast_loss_different_images():
    """Test with different images."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    # Create two different images
    image = torch.rand(2, 3, 64, 64)
    simulated_image = torch.rand(2, 3, 64, 64)
    
    loss = local_contrast_loss(image, simulated_image)
    
    # Loss should be positive
    assert loss.shape == torch.Size([])
    assert loss.item() > 0.0
    assert loss.item() <= 2.0  # Maximum would be 1 - 0 = 1 per patch, but with noise it's bounded


def test_local_contrast_loss_scaled_images():
    """Test with scaled images (simulating CVD effect)."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    # Create image and scaled version
    image = torch.rand(1, 3, 64, 64)
    simulated_image = image * 0.7  # Simulate CVD color reduction
    
    loss = local_contrast_loss(image, simulated_image)
    
    # Loss should be positive but not too large (contrast is preserved)
    assert loss.item() > 0.0
    assert loss.item() < 0.5  # Similar contrast should give small loss


def test_local_contrast_loss_different_patch_sizes():
    """Test with different patch sizes."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image * 0.8
    
    # Default patch size
    loss_default = local_contrast_loss(image, simulated_image, patch_size=4, stride=2)
    
    # Larger patch size
    loss_large = local_contrast_loss(image, simulated_image, patch_size=8, stride=4)
    
    # Both should be valid positive losses
    assert loss_default.item() > 0.0
    assert loss_large.item() > 0.0


def test_local_contrast_loss_different_strides():
    """Test with different stride values."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(1, 3, 64, 64)
    simulated_image = image * 0.9
    
    # Smaller stride (more overlap)
    loss_small_stride = local_contrast_loss(image, simulated_image, stride=1)
    
    # Larger stride (less overlap)
    loss_large_stride = local_contrast_loss(image, simulated_image, stride=4)
    
    # Both should be valid positive losses
    assert loss_small_stride.item() > 0.0
    assert loss_large_stride.item() > 0.0


def test_local_contrast_loss_minimum_image_size():
    """Test with minimum allowed image size (4x4)."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(1, 3, 4, 4)
    simulated_image = image * 0.9
    
    loss = local_contrast_loss(image, simulated_image)
    
    assert loss.item() >= 0.0


def test_local_contrast_loss_batch_size_1():
    """Test with batch size of 1."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(1, 3, 32, 32)
    simulated_image = image * 0.85
    
    loss = local_contrast_loss(image, simulated_image)
    
    assert loss.item() > 0.0


def test_local_contrast_loss_large_batch():
    """Test with larger batch size."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(8, 3, 64, 64)
    simulated_image = image * 0.8
    
    loss = local_contrast_loss(image, simulated_image)
    
    assert loss.item() > 0.0


def test_local_contrast_loss_type_error_image():
    """Test that TypeError is raised for non-tensor image."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    with pytest.raises(TypeError):
        local_contrast_loss("not_a_tensor", torch.rand(1, 3, 32, 32))


def test_local_contrast_loss_type_error_simulated():
    """Test that TypeError is raised for non-tensor simulated_image."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    with pytest.raises(TypeError):
        local_contrast_loss(torch.rand(1, 3, 32, 32), "not_a_tensor")


def test_local_contrast_loss_shape_mismatch():
    """Test that ValueError is raised for shape mismatch."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(2, 3, 64, 64)
    simulated_image = torch.rand(2, 3, 32, 32)  # Different size
    
    with pytest.raises(ValueError):
        local_contrast_loss(image, simulated_image)


def test_local_contrast_loss_image_too_small():
    """Test that ValueError is raised for images smaller than 4x4."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(1, 3, 3, 3)
    simulated_image = image * 0.9
    
    with pytest.raises(ValueError):
        local_contrast_loss(image, simulated_image)


def test_local_contrast_loss_wrong_dimensions():
    """Test that ValueError is raised for wrong tensor dimensions."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(3, 64, 64)  # 3D instead of 4D
    simulated_image = image * 0.9
    
    with pytest.raises(ValueError):
        local_contrast_loss(image, simulated_image)


def test_local_contrast_loss_gradient_flow():
    """Test that gradients flow through the loss function."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(2, 3, 64, 64, requires_grad=True)
    simulated_image = image.detach() * 0.8  # detached simulated
    
    loss = local_contrast_loss(image, simulated_image)
    loss.backward()
    
    assert image.grad is not None
    assert image.grad.shape == image.shape


def test_local_contrast_loss_perfect_preservation():
    """Test with perfect contrast preservation (identical images)."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    # Create random image and exact copy
    image = torch.rand(2, 3, 64, 64)
    simulated_image = image.clone()
    
    loss = local_contrast_loss(image, simulated_image)
    
    # Loss should be exactly 0
    assert loss.item() == 0.0


def test_local_contrast_loss_different_contrast_patterns():
    """Test with images having different contrast patterns."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    # Create image with high contrast (peaks and valleys)
    image = torch.rand(1, 3, 64, 64)
    image = (image - 0.5) * 2.0  # Center around 0, then * 2 = range [-1, 1]
    image = torch.clamp(image, 0, 1)  # Clamp to [0, 1]
    
    # Create simulated with reduced contrast (compressed range)
    simulated_image = image * 0.3  # Compress to [0, 0.3]
    
    loss = local_contrast_loss(image, simulated_image)
    
    # Loss should be significantly higher than with mild scaling
    loss_mild = local_contrast_loss(image, image * 0.9)
    assert loss.item() > loss_mild.item()
    assert loss.item() > 0.1  # Higher contrast reduction should give higher loss


def test_local_contrast_loss_cuda():
    """Test compatibility with CUDA tensors if available."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    image = torch.rand(2, 3, 64, 64, device='cuda')
    simulated_image = image.detach() * 0.8
    
    loss = local_contrast_loss(image, simulated_image)
    
    assert loss.device.type == 'cuda'
    assert loss.item() > 0.0


def test_local_contrast_loss_different_channel_counts():
    """Test with different channel counts (grayscale and RGB)."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    # Grayscale
    image_gray = torch.rand(2, 1, 64, 64)
    simulated_gray = image_gray * 0.8
    loss_gray = local_contrast_loss(image_gray, simulated_gray)
    
    # RGB
    image_rgb = torch.rand(2, 3, 64, 64)
    simulated_rgb = image_rgb * 0.8
    loss_rgb = local_contrast_loss(image_rgb, simulated_rgb)
    
    # Both should be valid positive losses
    assert loss_gray.item() > 0.0
    assert loss_rgb.item() > 0.0


def test_local_contrast_loss_epsilon_parameter():
    """Test with custom epsilon values."""
    from odak.learn.perception.cvd_loss_functions import local_contrast_loss
    
    image = torch.rand(1, 3, 64, 64)
    simulated_image = image * 0.85
    
    # Default epsilon
    loss_default = local_contrast_loss(image, simulated_image, epsilon=1e-8)
    
    # Larger epsilon
    loss_large_epsilon = local_contrast_loss(image, simulated_image, epsilon=1e-4)
    
    # Both should be valid and close to each other
    assert loss_default.item() > 0.0
    assert loss_large_epsilon.item() > 0.0
    # They should be reasonably close (within 10%)
    diff_percent = abs(loss_default.item() - loss_large_epsilon.item()) / loss_default.item()
    assert diff_percent < 0.1


def run_all_tests():
    """Run all tests and return True if all pass."""
    try:
        test_local_contrast_loss_basic()
        test_local_contrast_loss_different_images()
        test_local_contrast_loss_scaled_images()
        test_local_contrast_loss_different_patch_sizes()
        test_local_contrast_loss_different_strides()
        test_local_contrast_loss_minimum_image_size()
        test_local_contrast_loss_batch_size_1()
        test_local_contrast_loss_large_batch()
        test_local_contrast_loss_type_error_image()
        test_local_contrast_loss_type_error_simulated()
        test_local_contrast_loss_shape_mismatch()
        test_local_contrast_loss_image_too_small()
        test_local_contrast_loss_wrong_dimensions()
        test_local_contrast_loss_gradient_flow()
        test_local_contrast_loss_perfect_preservation()
        test_local_contrast_loss_different_contrast_patterns()
        test_local_contrast_loss_different_channel_counts()
        test_local_contrast_loss_epsilon_parameter()
        
        # Test CUDA if available
        if torch.cuda.is_available():
            test_local_contrast_loss_cuda()
        
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
