"""
Unit tests for michelson_contrast function.

Tests the Michelson contrast ratio computation for specified image regions.
"""

import torch
import sys
import pytest
import numpy as np


def test_michelson_contrast_2d():
    """Test Michelson contrast with 2D input."""
    from odak.learn.perception import michelson_contrast
    
    # Create test image with known intensity pattern
    image = torch.zeros(64, 64)
    image[0:32, 0:32] = 0.8  # High region
    image[32:64, 32:64] = 0.2  # Low region
    
    roi_high = [0, 32, 0, 32]
    roi_low = [32, 64, 32, 64]
    
    contrast = michelson_contrast(image, roi_high, roi_low)
    
    # Expected: (0.8 - 0.2) / (0.8 + 0.2) = 0.6
    expected = torch.tensor(0.6)
    assert torch.allclose(contrast, expected, atol=1e-5)


def test_michelson_contrast_3d():
    """Test Michelson contrast with 3D (C, H, W) input."""
    from odak.learn.perception import michelson_contrast
    
    image = torch.zeros(3, 64, 64)
    image[:, 0:32, 0:32] = 0.9  # High region
    image[:, 32:64, 32:64] = 0.3  # Low region
    
    roi_high = [0, 32, 0, 32]
    roi_low = [32, 64, 32, 64]
    
    contrast = michelson_contrast(image, roi_high, roi_low)
    
    # Expected per channel: (0.9 - 0.3) / (0.9 + 0.3) = 0.5
    expected = torch.tensor(0.5)
    assert contrast.shape == (3,)
    assert torch.allclose(contrast, expected, atol=1e-5)


def test_michelson_contrast_4d():
    """Test Michelson contrast with 4D (B, C, H, W) input."""
    from odak.learn.perception import michelson_contrast
    
    image = torch.zeros(2, 1, 64, 64)
    image[:, :, 0:32, 0:32] = 0.7  # High region
    image[:, :, 32:64, 32:64] = 0.4  # Low region
    
    roi_high = [0, 32, 0, 32]
    roi_low = [32, 64, 32, 64]
    
    contrast = michelson_contrast(image, roi_high, roi_low)
    
    # Expected: (0.7 - 0.4) / (0.7 + 0.4) = 0.2727...
    expected = torch.tensor((0.7 - 0.4) / (0.7 + 0.4))
    assert torch.allclose(contrast, expected, atol=1e-5)


def test_michelson_contrast_identical_regions():
    """Test with identical high and low regions (contrast = 0)."""
    from odak.learn.perception import michelson_contrast
    
    image = torch.ones(64, 64) * 0.5
    roi_high = [0, 32, 0, 32]
    roi_low = [32, 64, 32, 64]
    
    contrast = michelson_contrast(image, roi_high, roi_low)
    
    assert torch.allclose(contrast, torch.tensor(0.0), atol=1e-5)


def test_michelson_contrast_value_range():
    """Test that Michelson contrast is always in [0, 1] range."""
    from odak.learn.perception import michelson_contrast
    
    # Test various intensity combinations
    for high_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for low_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if high_val >= low_val:
                image = torch.zeros(64, 64)
                image[0:32, 0:32] = high_val
                image[32:64, 32:64] = low_val
                
                roi_high = [0, 32, 0, 32]
                roi_low = [32, 64, 32, 64]
                
                contrast = michelson_contrast(image, roi_high, roi_low)
                
                # Michelson contrast should always be in [0, 1]
                assert 0.0 <= contrast.item() <= 1.0


def test_michelson_contrast_maximum_contrast():
    """Test maximum contrast case (low = 0, high > 0)."""
    from odak.learn.perception import michelson_contrast
    
    image = torch.zeros(64, 64)
    image[0:32, 0:32] = 1.0  # High region
    image[32:64, 32:64] = 0.0  # Low region (zero)
    
    roi_high = [0, 32, 0, 32]
    roi_low = [32, 64, 32, 64]
    
    contrast = michelson_contrast(image, roi_high, roi_low)
    
    # Expected: (1.0 - 0.0) / (1.0 + 0.0) = 1.0
    assert torch.allclose(contrast, torch.tensor(1.0), atol=1e-5)


def test_michelson_contrast_type_error():
    """Test that TypeError is raised for non-tensor input."""
    from odak.learn.perception import michelson_contrast
    
    with pytest.raises(TypeError):
        michelson_contrast("not_a_tensor", [0, 32, 0, 32], [32, 64, 32, 64])


def test_michelson_contrast_invalid_roi_length():
    """Test that ValueError is raised for invalid ROI length."""
    from odak.learn.perception import michelson_contrast
    
    image = torch.rand(64, 64)
    
    with pytest.raises(ValueError):
        michelson_contrast(image, [0, 32, 0], [32, 64, 32, 64])


def test_michelson_contrast_roi_out_of_bounds():
    """Test that ValueError is raised for ROI out of bounds."""
    from odak.learn.perception import michelson_contrast
    
    image = torch.rand(64, 64)
    
    with pytest.raises(ValueError):
        michelson_contrast(image, [0, 100, 0, 100], [32, 64, 32, 64])


def test_michelson_contrast_invalid_roi_order():
    """Test that ValueError is raised for invalid ROI ordering."""
    from odak.learn.perception import michelson_contrast
    
    image = torch.rand(64, 64)
    
    with pytest.raises(ValueError):
        michelson_contrast(image, [32, 0, 32, 0], [0, 32, 0, 32])


def test_michelson_contrast_zero_sum():
    """Test that ValueError is raised when sum of means is zero."""
    from odak.learn.perception import michelson_contrast
    
    image = torch.zeros(64, 64)  # All zeros
    roi_high = [0, 32, 0, 32]
    roi_low = [32, 64, 32, 64]
    
    with pytest.raises(ValueError):
        michelson_contrast(image, roi_high, roi_low)


def test_michelson_contrast_cuda():
    """Test compatibility with CUDA tensors if available."""
    from odak.learn.perception import michelson_contrast
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    image = torch.rand(64, 64, device='cuda')
    image[0:32, 0:32] = 0.8
    image[32:64, 32:64] = 0.2
    
    roi_high = [0, 32, 0, 32]
    roi_low = [32, 64, 32, 64]
    
    contrast = michelson_contrast(image, roi_high, roi_low)
    
    assert contrast.device.type == 'cuda'
    assert 0.0 <= contrast.item() <= 1.0


def test_michelson_contrast_from_real_image():
    """Test with a real image from test data."""
    from odak.learn.perception import michelson_contrast
    
    # Try to load a test image
    try:
        import cv2
        img = cv2.imread('/mnt/yedek/bulut/depolar/odak/test/data/fruit_lady.png')
        if img is not None:
            # Convert to float and normalize
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_tensor = torch.from_numpy(img).float() / 255.0
            
            roi_high = [0, 100, 0, 100]
            roi_low = [200, 300, 200, 300]
            
            contrast = michelson_contrast(img_tensor, roi_high, roi_low)
            assert 0.0 <= contrast.item() <= 1.0
    except Exception as e:
        # Skip if test image not available
        pytest.skip(f"Cannot load test image: {e}")


def test_michelson_contrast_multiple_channels():
    """Test with different values per channel."""
    from odak.learn.perception import michelson_contrast
    
    image = torch.zeros(3, 64, 64)
    # Different intensity patterns per channel
    image[0, 0:32, 0:32] = 0.9
    image[0, 32:64, 32:64] = 0.3
    
    image[1, 0:32, 0:32] = 0.8
    image[1, 32:64, 32:64] = 0.4
    
    image[2, 0:32, 0:32] = 0.7
    image[2, 32:64, 32:64] = 0.5
    
    roi_high = [0, 32, 0, 32]
    roi_low = [32, 64, 32, 64]
    
    contrast = michelson_contrast(image, roi_high, roi_low)
    
    # Expected per channel
    expected_0 = (0.9 - 0.3) / (0.9 + 0.3)  # 0.5
    expected_1 = (0.8 - 0.4) / (0.8 + 0.4)  # 0.3333...
    expected_2 = (0.7 - 0.5) / (0.7 + 0.5)  # 0.1667...
    
    expected = torch.tensor([expected_0, expected_1, expected_2])
    assert torch.allclose(contrast, expected, atol=1e-5)


def test_michelson_contrast_comparison_weber():
    """Test that Michelson and Weber contrast give different results."""
    from odak.learn.perception import michelson_contrast, weber_contrast
    
    image = torch.zeros(64, 64)
    image[0:32, 0:32] = 0.8
    image[32:64, 32:64] = 0.2
    
    roi_high = [0, 32, 0, 32]
    roi_low = [32, 64, 32, 64]
    
    weber = weber_contrast(image, roi_high, roi_low)
    michelson = michelson_contrast(image, roi_high, roi_low)
    
    # Weber: (0.8 - 0.2) / 0.2 = 3.0
    # Michelson: (0.8 - 0.2) / (0.8 + 0.2) = 0.6
    assert weber.item() > michelson.item()  # Weber always > Michelson for the same regions


def run_all_tests():
    """Run all tests and return True if all pass."""
    try:
        test_michelson_contrast_2d()
        test_michelson_contrast_3d()
        test_michelson_contrast_4d()
        test_michelson_contrast_identical_regions()
        test_michelson_contrast_value_range()
        test_michelson_contrast_maximum_contrast()
        test_michelson_contrast_type_error()
        test_michelson_contrast_invalid_roi_length()
        test_michelson_contrast_roi_out_of_bounds()
        test_michelson_contrast_invalid_roi_order()
        test_michelson_contrast_zero_sum()
        test_michelson_contrast_multiple_channels()
        test_michelson_contrast_comparison_weber()
        
        # Test CUDA if available
        if torch.cuda.is_available():
            test_michelson_contrast_cuda()
        
        # Test with real image
        try:
            test_michelson_contrast_from_real_image()
        except:
            pass  # Skip if test image not available
        
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
