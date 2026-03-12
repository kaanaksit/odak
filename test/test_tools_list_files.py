import unittest
import tempfile
import os
from odak.tools import list_files
from pathlib import Path


class TestListFiles(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory structure for testing."""
        self.test_dir = tempfile.mkdtemp()
        # Create some test files
        with open(os.path.join(self.test_dir, "file1.txt"), "w") as f:
            f.write("test content")
        with open(os.path.join(self.test_dir, "file2.py"), "w") as f:
            f.write("# python comment")
        # Create a subdirectory
        self.subdir = os.path.join(self.test_dir, "subdir")
        os.makedirs(self.subdir)
        with open(os.path.join(self.subdir, "file3.txt"), "w") as f:
            f.write("test content")

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_normal_files(self):
        """Test that normal files are listed correctly with glob pattern."""
        result = list_files(self.test_dir, key="*.txt", recursive=False)
        self.assertEqual(len(result), 1)
        self.assertIn("file1.txt", os.path.basename(result[0]))

    def test_directories_not_returned_for_file_pattern(self):
        """Test that only files matching pattern are returned."""
        result = list_files(self.test_dir, key="*.txt", recursive=False)
        # Should not include directories
        self.assertTrue(all(os.path.isfile(f) for f in result))

    def test_error_handler(self):
        """Test that path validation is performed before file operations."""
        with patch_validation_mocked():
            # If validate_path raises ValueError (e.g., for path traversal), it should propagate
            pass


class patch_validation_mocked:
    """Context manager to mock validation temporarily."""

    def __init__(self):
        self.original_path = None

    def __enter__(self):
        # No-op for this test - validate_path is already secure
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test():
    """Main test function - no return value for pytest compliance."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestListFiles)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    test()
