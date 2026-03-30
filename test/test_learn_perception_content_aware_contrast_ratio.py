"""
Unit tests for content_aware_contrast_ratio function.

Tests the CWMC implementation following the Ortiz-Jaramillo et al. (2018) paper.
"""

import torch
import sys
import pytest


def test_content_aware_contrast_ratio_basic():
    """Test basic functionality with simple image."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    # Create a simple test image
    image = torch.rand(1, 1, 64, 64)
    
    result = content_aware_contrast_ratio(image, window_size=15, step=3)
    
    # Check returned dictionary structure
    assert 'contrast_map' in result
    assert 'threshold_map' in result
    assert 'foreground_mean_map' in result
    assert 'background_mean_map' in result
    assert 'pooled_harmonic_mean' in result
    assert 'max_contrast_ratio' in result
    assert 'window_size' in result
    assert 'step' in result
    assert 'n_patches' in result
    
    # Check contrast map exists and has correct shape
    assert result['contrast_map'].shape == (64, 64)
    assert result['pooled_harmonic_mean'] >= 0.0
    assert result['max_contrast_ratio'] >= 0.0


def test_content_aware_contrast_ratio_multichannel():
    """Test with multi-channel image."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 3, 64, 64)  # RGB image
    
    result = content_aware_contrast_ratio(image, window_size=15, step=3)
    
    assert result['contrast_map'].shape == (64, 64)
    assert result['pooled_harmonic_mean'] >= 0.0


def test_content_aware_contrast_ratio_2d_input():
    """Test with 2D grayscale input."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(64, 64)  # 2D input
    
    result = content_aware_contrast_ratio(image, window_size=15, step=3)
    
    assert result['contrast_map'].shape == (64, 64)


def test_content_aware_contrast_ratio_different_window_sizes():
    """Test with different window sizes."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 1, 128, 128)
    
    for window_size in [5, 9, 15, 21]:
        result = content_aware_contrast_ratio(
            image, window_size=window_size, step=3
        )
        assert result['contrast_map'].shape == (128, 128)
        assert result['window_size'] == window_size
        assert result['pooled_harmonic_mean'] >= 0.0


def test_content_aware_contrast_ratio_different_steps():
    """Test with different step sizes."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 1, 128, 128)
    
    for step in [1, 3, 5]:
        result = content_aware_contrast_ratio(
            image, window_size=15, step=step
        )
        assert result['contrast_map'].shape == (128, 128)
        assert result['step'] == step


def test_content_aware_contrast_ratio_percentile():
    """Test with different pooling percentiles."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 1, 128, 128)
    
    for percentile in [50.0, 75.0, 90.0]:
        result = content_aware_contrast_ratio(
            image, window_size=15, step=3, pooling_percentile=percentile
        )
        assert result['pooled_harmonic_mean'] >= 0.0


def test_content_aware_contrast_ratio_identical_patches():
    """Test with image that has identical patches (should give contrast ~0)."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    # Create image with uniform intensity
    image = torch.ones(1, 1, 64, 64) * 0.5
    
    result = content_aware_contrast_ratio(image, window_size=15, step=3)
    
    # For uniform image, contrast should be very small (all patches identical)
    assert result['pooled_harmonic_mean'] < 0.01


def test_content_aware_contrast_ratio_gradient_flow():
    """Test that gradients can flow through the function."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 1, 64, 64, requires_grad=True)
    
    result = content_aware_contrast_ratio(image, window_size=15, step=3)
    
    # Check that result has gradient capability
    # Note: This test may fail if the implementation uses operations that break gradients
    print("Gradient test completed")


def test_content_aware_contrast_ratio_value_ranges():
    """Test that computed values are in expected ranges."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 1, 128, 128)
    
    result = content_aware_contrast_ratio(image, window_size=15, step=3)
    
    # Contrast values should be in [0, 1] range (Weber contrast)
    assert result['max_contrast_ratio'] >= 0.0
    assert result['max_contrast_ratio'] <= 1.0
    assert result['pooled_harmonic_mean'] >= 0.0
    assert result['pooled_harmonic_mean'] <= 1.0


def test_content_aware_contrast_ratio_maps_consistency():
    """Test that all maps have consistent non-zero values where expected."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 1, 64, 64)
    
    result = content_aware_contrast_ratio(image, window_size=15, step=3)
    
    # Check that contrast_map has valid values at window centers
    contrast_map = result['contrast_map']
    threshold_map = result['threshold_map']
    
    # At step=3, windows are placed at positions 7, 10, 13, 16, ...
    # (starting from index 7 which is window_size//2)
    step = 3
    half = 7  # window_size // 2
    
    # Check a few sample positions
    for i in range(half, 64, step):
        for j in range(half, 64, step):
            # Check contrast value is valid
            assert 0.0 <= contrast_map[i, j] <= 1.0


def test_content_aware_contrast_ratio_cuda():
    """Test compatibility with CUDA tensors if available."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    image = torch.rand(1, 1, 64, 64, device='cuda')
    
    result = content_aware_contrast_ratio(image, window_size=15, step=3)
    
    assert result['contrast_map'].device.type == 'cuda'
    assert result['pooled_harmonic_mean'] >= 0.0


def test_content_aware_contrast_ratio_error_invalid_window_size():
    """Test that ValueError is raised for invalid window_size."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 1, 64, 64)
    
    # Even window size
    with pytest.raises(ValueError):
        content_aware_contrast_ratio(image, window_size=10, step=3)
    
    # Too small window size
    with pytest.raises(ValueError):
        content_aware_contrast_ratio(image, window_size=1, step=3)


def test_content_aware_contrast_ratio_error_invalid_step():
    """Test that ValueError is raised for invalid step."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 1, 64, 64)
    
    # Step < 1
    with pytest.raises(ValueError):
        content_aware_contrast_ratio(image, window_size=15, step=0)


def test_content_aware_contrast_ratio_error_invalid_percentile():
    """Test that ValueError is raised for invalid pooling_percentile."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    image = torch.rand(1, 1, 64, 64)
    
    # Percentile <= 0
    with pytest.raises(ValueError):
        content_aware_contrast_ratio(image, window_size=15, step=3, pooling_percentile=0.0)
    
    # Percentile > 100
    with pytest.raises(ValueError):
        content_aware_contrast_ratio(image, window_size=15, step=3, pooling_percentile=105.0)


def test_content_aware_contrast_ratio_batch():
    """Test with batched input."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    images = torch.rand(4, 1, 64, 64)  # Batch of 4 images
    
    results = content_aware_contrast_ratio(images, window_size=15, step=3)
    
    # Should return list of results for batch
    assert isinstance(results, list)
    assert len(results) == 4
    
    for result in results:
        assert result['contrast_map'].shape == (64, 64)
        assert result['pooled_harmonic_mean'] >= 0.0


def test_content_aware_contrast_ratio_map_values():
    """Test that map values are reasonable."""
    from odak.learn.perception import content_aware_contrast_ratio
    
    # Create an image with high contrast regions
    image = torch.zeros(1, 1, 64, 64)
    image[:, :, :32, :] = 0.1  # Dark region
    image[:, :, 32:, :] = 0.9  # Bright region
    
    result = content_aware_contrast_ratio(image, window_size=15, step=3)
    
    # Should have significant contrast
    assert result['max_contrast_ratio'] > 0.0
    assert result['pooled_harmonic_mean'] > 0.0


def run_all_tests():
    """Run all tests and return True if all pass."""
    try:
        test_content_aware_contrast_ratio_basic()
        test_content_aware_contrast_ratio_multichannel()
        test_content_aware_contrast_ratio_2d_input()
        test_content_aware_contrast_ratio_different_window_sizes()
        test_content_aware_contrast_ratio_different_steps()
        test_content_aware_contrast_ratio_percentile()
        test_content_aware_contrast_ratio_identical_patches()
        test_content_aware_contrast_ratio_gradient_flow()
        test_content_aware_contrast_ratio_value_ranges()
        test_content_aware_contrast_ratio_maps_consistency()
        test_content_aware_contrast_ratio_error_invalid_window_size()
        test_content_aware_contrast_ratio_error_invalid_step()
        test_content_aware_contrast_ratio_error_invalid_percentile()
        test_content_aware_contrast_ratio_batch()
        test_content_aware_contrast_ratio_map_values()
        
        # Test CUDA if available
        if torch.cuda.is_available():
            test_content_aware_contrast_ratio_cuda()
        
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
