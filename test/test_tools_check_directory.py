import unittest
import os
import tempfile
from odak.tools import check_directory


class TestCheckDirectory(unittest.TestCase):
    def test_directory_exists(self):
        """Test that check_directory returns True when directory exists"""
        temp_dir = tempfile.mkdtemp()
        result = check_directory(temp_dir)
        self.assertTrue(result)

    def test_directory_does_not_exist(self):
        """Test that check_directory creates directory and returns False when directory does not exist"""
        # Create a temporary directory path that doesn't exist
        temp_dir = tempfile.mkdtemp()
        nonexistent_dir = os.path.join(temp_dir, "nonexistent_directory")

        # Ensure it doesn't exist
        if os.path.exists(nonexistent_dir):
            os.rmdir(nonexistent_dir)

        result = check_directory(nonexistent_dir)
        self.assertFalse(result)

        # Verify that the directory was created
        self.assertTrue(os.path.exists(nonexistent_dir))

    def test_directory_with_tilde_expansion(self):
        """Test that check_directory works with tilde expansion"""
        # Create a temporary directory manually
        temp_dir = tempfile.mkdtemp()
        # Create a subdirectory inside the temp dir
        subdir = os.path.join(temp_dir, "test_dir")
        os.makedirs(subdir)

        # Test with tilde expansion
        tilde_path = subdir.replace(os.path.expanduser("~"), "~")
        result = check_directory(tilde_path)

        # Should return True since the directory exists
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
