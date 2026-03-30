"""
Unit tests for compute_local_contrast_similarity function.

Tests the local contrast similarity computation between image patches.
"""

import torch
import sys
import pytest


def test_compute_local_contrast_similarity_basic():
    """Test basic functionality with simple patches."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    # Create two identical patches
    patch_x = torch.ones(16, 10)  # 16 channels (4x4), 10 patches
    patch_y = torch.ones(16, 10)
    
    similarity = compute_local_contrast_similarity(patch_x, patch_y)
    
    # Identical patches should have similarity close to 1.0
    assert similarity.shape == (10,)
    assert torch.allclose(similarity, torch.ones(10), atol=1e-5)


def test_compute_local_contrast_similarity_different_patches():
    """Test with different patches."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    # Create two different patches
    patch_x = torch.randn(16, 10)
    patch_y = torch.randn(16, 10)
    
    similarity = compute_local_contrast_similarity(patch_x, patch_y)
    
    # Similarity should be between 0 and 1
    assert similarity.shape == (10,)
    assert torch.all(similarity >= 0.0)
    assert torch.all(similarity <= 1.0)


def test_compute_local_contrast_similarity_batch():
    """Test with batches of patches with varying similarity."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    # Create patches with known relationship
    base = torch.randn(16, 5)
    patch_x = base
    patch_y = base * 0.8  # Scaled version
    
    similarity = compute_local_contrast_similarity(patch_x, patch_y)
    
    # Should have similarity less than 1.0 but greater than 0
    assert similarity.shape == (5,)
    assert torch.all(similarity < 1.0)
    assert torch.all(similarity > 0.0)


def test_compute_local_contrast_similarity_single_patch():
    """Test with single patch."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    patch_x = torch.randn(16, 1)
    patch_y = torch.randn(16, 1)
    
    similarity = compute_local_contrast_similarity(patch_x, patch_y)
    
    assert similarity.shape == (1,)
    assert similarity.item() >= 0.0
    assert similarity.item() <= 1.0


def test_compute_local_contrast_similarity_epsilon_handling():
    """Test epsilon prevents division issues."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    # Create patches with very small standard deviation
    patch_x = torch.ones(16, 10) * 0.001
    patch_y = torch.ones(16, 10) * 0.001
    
    similarity = compute_local_contrast_similarity(patch_x, patch_y, epsilon=1e-8)
    
    # Should not crash and should be close to 1.0
    assert similarity.shape == (10,)
    assert torch.allclose(similarity, torch.ones(10), atol=1e-4)


def test_compute_local_contrast_similarity_type_error():
    """Test that TypeError is raised for non-tensor inputs."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    with pytest.raises(TypeError):
        compute_local_contrast_similarity("not_a_tensor", torch.randn(16, 10))
    
    with pytest.raises(TypeError):
        compute_local_contrast_similarity(torch.randn(16, 10), "not_a_tensor")


def test_compute_local_contrast_similarity_shape_mismatch():
    """Test that ValueError is raised for mismatched shapes."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    patch_x = torch.randn(16, 10)
    patch_y = torch.randn(32, 10)  # Different channel count
    
    with pytest.raises(ValueError):
        compute_local_contrast_similarity(patch_x, patch_y)


def test_compute_local_contrast_similarity_cuda():
    """Test compatibility with CUDA tensors if available."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    patch_x = torch.randn(16, 10, device='cuda')
    patch_y = torch.randn(16, 10, device='cuda')
    
    similarity = compute_local_contrast_similarity(patch_x, patch_y)
    
    assert similarity.device.type == 'cuda'
    assert similarity.shape == (10,)
    assert torch.all(similarity >= 0.0)
    assert torch.all(similarity <= 1.0)


def test_compute_local_contrast_similarity_gradient():
    """Test that gradients flow through the function."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    patch_x = torch.randn(16, 10, requires_grad=True)
    patch_y = torch.randn(16, 10, requires_grad=False)
    
    similarity = compute_local_contrast_similarity(patch_x, patch_y)
    loss = similarity.sum()
    loss.backward()
    
    assert patch_x.grad is not None
    assert patch_x.grad.shape == patch_x.shape


def test_compute_local_contrast_similarity_custom_epsilon():
    """Test with custom epsilon values."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    patch_x = torch.randn(16, 10)
    patch_y = torch.randn(16, 10)
    
    # Test with different epsilon values
    sim_default = compute_local_contrast_similarity(patch_x, patch_y, epsilon=1e-8)
    sim_large = compute_local_contrast_similarity(patch_x, patch_y, epsilon=1e-4)
    
    # Both should be valid (0 to 1)
    assert torch.all(sim_default >= 0.0) and torch.all(sim_default <= 1.0)
    assert torch.all(sim_large >= 0.0) and torch.all(sim_large <= 1.0)


def test_compute_local_contrast_similarity_constant_patches():
    """Test with constant value patches."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    # All zeros
    patch_x = torch.zeros(16, 10)
    patch_y = torch.zeros(16, 10)
    similarity = compute_local_contrast_similarity(patch_x, patch_y)
    assert torch.allclose(similarity, torch.ones(10), atol=1e-5)
    
    # All same constant
    patch_x = torch.ones(16, 10) * 5.0
    patch_y = torch.ones(16, 10) * 5.0
    similarity = compute_local_contrast_similarity(patch_x, patch_y)
    assert torch.allclose(similarity, torch.ones(10), atol=1e-5)


def test_compute_local_contrast_similarity_large_patches():
    """Test with larger number of patches."""
    from odak.learn.perception.cvd_loss_functions import compute_local_contrast_similarity
    
    patch_x = torch.randn(64, 100)  # 8x8 patches, 100 patches
    patch_y = torch.randn(64, 100)
    
    similarity = compute_local_contrast_similarity(patch_x, patch_y)
    
    assert similarity.shape == (100,)
    assert torch.all(similarity >= 0.0)
    assert torch.all(similarity <= 1.0)


def run_all_tests():
    """Run all tests and return True if all pass."""
    try:
        test_compute_local_contrast_similarity_basic()
        test_compute_local_contrast_similarity_different_patches()
        test_compute_local_contrast_similarity_batch()
        test_compute_local_contrast_similarity_single_patch()
        test_compute_local_contrast_similarity_epsilon_handling()
        test_compute_local_contrast_similarity_type_error()
        test_compute_local_contrast_similarity_shape_mismatch()
        test_compute_local_contrast_similarity_gradient()
        test_compute_local_contrast_similarity_custom_epsilon()
        test_compute_local_contrast_similarity_constant_patches()
        test_compute_local_contrast_similarity_large_patches()
        
        # Test CUDA if available
        if torch.cuda.is_available():
            test_compute_local_contrast_similarity_cuda()
        
        print("All tests passed!")
        return True
    except AssertionError as e:
        print(f"Test failed: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
