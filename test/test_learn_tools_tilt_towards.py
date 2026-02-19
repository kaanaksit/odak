import torch
import unittest
from odak.learn.tools.transformation import tilt_towards

class TestTiltTowards(unittest.TestCase):
    def test_tilt_towards_basic(self):
        """Test basic functionality of tilt_towards function"""
        location = [0, 0, 0]
        lookat = [1, 0, 0]
        result = tilt_towards(location, lookat)
        # For a point looking at [1, 0, 0] from [0, 0, 0]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        # Should return [0, 90.0, 180.0]
        self.assertAlmostEqual(result[0], 0.0, places=5)   # X rotation
        self.assertAlmostEqual(result[1], 90.0, places=5)  # Y rotation (theta)
        self.assertAlmostEqual(result[2], 180.0, places=5) # Z rotation (phi)

    def test_tilt_towards_upward(self):
        """Test tilt_towards with upward pointing"""
        location = [0, 0, 0]
        lookat = [0, 0, 1]
        result = tilt_towards(location, lookat)
        # For a point looking at [0, 0, 1] from [0, 0, 0]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        # Should return [0, 180.0, 0.0] or [0, 180.0, 180.0] 
        self.assertAlmostEqual(result[0], 0.0, places=5)   # X rotation
        self.assertAlmostEqual(result[1], 180.0, places=5)  # Y rotation (theta)
        # Check that phi is either 0 or 180 degrees
        self.assertIn(result[2], [0.0, 180.0], msg="Phi should be 0 or 180 degrees")

    def test_tilt_towards_diagonal(self):
        """Test tilt_towards with diagonal pointing"""
        location = [0, 0, 0]
        lookat = [1, 1, 0]
        result = tilt_towards(location, lookat)
        # For a point looking at [1, 1, 0] from [0, 0, 0]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        # Should return [0, 90.0, -135.0] (or equivalent angle)
        self.assertAlmostEqual(result[0], 0.0, places=5)   # X rotation
        self.assertAlmostEqual(result[1], 90.0, places=5)  # Y rotation (theta)
        # Phi should be -135.0 (which is equivalent to 225.0)
        self.assertAlmostEqual(result[2], -135.0, places=5) # Z rotation (phi)

    def test_tilt_towards_offset_location(self):
        """Test tilt_towards with offset location"""
        location = [1, 1, 1]
        lookat = [2, 1, 1]
        result = tilt_towards(location, lookat)
        # Should point in X direction only (return [0, 90.0, 180.0])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.0, places=5)   # X rotation
        self.assertAlmostEqual(result[1], 90.0, places=5)  # Y rotation (theta)
        self.assertAlmostEqual(result[2], 180.0, places=5) # Z rotation (phi)

if __name__ == "__main__":
    unittest.main()