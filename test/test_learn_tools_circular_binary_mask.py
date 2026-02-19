import torch
import unittest
from odak.learn.tools.mask import circular_binary_mask

class TestCircularBinaryMask(unittest.TestCase):
    def test_basic_functionality(self):
        """Test basic functionality of circular_binary_mask"""
        # Test with basic parameters
        mask = circular_binary_mask(10, 10, 5)
        self.assertEqual(mask.shape, (1, 1, 10, 10))
        self.assertTrue(torch.all(mask >= 0))
        self.assertTrue(torch.all(mask <= 1))
        
    def test_radius_zero(self):
        """Test with radius of 0"""
        mask = circular_binary_mask(5, 5, 0)
        # Should be all zeros
        self.assertTrue(torch.all(mask == 0))
        
    def test_radius_larger_than_dimensions(self):
        """Test with radius larger than image dimensions"""
        mask = circular_binary_mask(5, 5, 10)
        # Should be all ones (entire image covered)
        self.assertTrue(torch.all(mask == 1))
        
    def test_radius_equal_to_dimensions(self):
        """Test with radius equal to half of dimensions"""
        mask = circular_binary_mask(10, 10, 5)
        # Center should be 1, edges should be 0
        # Check the center point (should be 1)
        self.assertEqual(mask[0, 0, 4, 4], 1)
        # Check a point outside the circle (should be 0)
        self.assertEqual(mask[0, 0, 0, 0], 0)
        
    def test_different_radius_types(self):
        """Test with different radius types (int and float)"""
        mask1 = circular_binary_mask(10, 10, 3)
        mask2 = circular_binary_mask(10, 10, 3.5)
        self.assertEqual(mask1.shape, (1, 1, 10, 10))
        self.assertEqual(mask2.shape, (1, 1, 10, 10))

if __name__ == '__main__':
    unittest.main()