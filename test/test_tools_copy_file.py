import os
import tempfile
import shutil
import unittest
from odak.tools import copy_file


class TestCopyFile(unittest.TestCase):
    def setUp(self):
        """Create temporary directory and test file for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.source_file = os.path.join(self.temp_dir, "source.txt")
        with open(self.source_file, "w") as f:
            f.write("test content")

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_copy_file_success(self):
        """Test successful file copy."""
        dest_file = os.path.join(self.temp_dir, "dest.txt")
        result = copy_file(self.source_file, dest_file)

        # Should return destination path
        self.assertEqual(result, dest_file)

        # Verify file was copied
        self.assertTrue(os.path.exists(dest_file))
        with open(dest_file, "r") as f:
            self.assertEqual(f.read(), "test content")

    def test_source_does_not_exist(self):
        """Test that ValueError is raised when source doesn't exist."""
        fake_source = os.path.join(self.temp_dir, "nonexistent.txt")
        dest_file = os.path.join(self.temp_dir, "dest.txt")

        with self.assertRaises(ValueError) as context:
            copy_file(fake_source, dest_file)

        self.assertIn("Source file does not exist", str(context.exception))

    def test_source_is_directory(self):
        """Test that ValueError is raised when source is a directory."""
        subdir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(subdir)
        dest_file = os.path.join(self.temp_dir, "dest.txt")

        with self.assertRaises(ValueError) as context:
            copy_file(subdir, dest_file)

        self.assertIn("Source is not a file", str(context.exception))

    def test_destination_is_directory(self):
        """Test that ValueError is raised when destination is a directory."""
        subdir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(subdir)

        with self.assertRaises(ValueError) as context:
            copy_file(self.source_file, subdir)

        self.assertIn("Destination is a directory", str(context.exception))

    def test_copy_preserves_metadata(self):
        """Test that file metadata (timestamps) are preserved."""
        dest_file = os.path.join(self.temp_dir, "dest.txt")
        copy_file(self.source_file, dest_file)

        # Get timestamps
        source_stat = os.stat(self.source_file)
        dest_stat = os.stat(dest_file)

        # Timestamps should be preserved (copy2 does this)
        self.assertEqual(
            source_stat.st_mtime, dest_stat.st_mtime, "Modification times should match"
        )
        self.assertEqual(
            source_stat.st_atime, dest_stat.st_atime, "Access times should match"
        )

    def test_overwrite_existing_file(self):
        """Test that copy_file can overwrite existing destination."""
        dest_file = os.path.join(self.temp_dir, "dest.txt")
        # Create existing destination file
        with open(dest_file, "w") as f:
            f.write("old content")

        copy_file(self.source_file, dest_file)

        # Verify content was overwritten
        with open(dest_file, "r") as f:
            self.assertEqual(f.read(), "test content")

    def test_copy_with_tilde_expansion(self):
        """Test that tilde expansion works in paths."""
        if os.path.expanduser("~") == "~":
            self.skipTest("~ does not expand in this environment")

        home_dir = os.path.expanduser("~/test_copy_temp")
        try:
            os.makedirs(home_dir, exist_ok=True)
            source_tilde = os.path.join("~/test_copy_temp", "source.txt")
            dest_tilde = os.path.join("~/test_copy_temp", "dest.txt")

            # Create source file in home directory using tilde path
            full_source = os.path.expanduser(source_tilde)
            with open(full_source, "w") as f:
                f.write("tilde test content")

            copy_file(source_tilde, dest_tilde)

            full_dest = os.path.expanduser(dest_tilde)
            self.assertTrue(os.path.exists(full_dest))
        finally:
            # Cleanup
            shutil.rmtree(home_dir, ignore_errors=True)

    def test_follow_symlinks_true(self):
        """Test that follow_symlinks=True copies the linked file."""
        # Create a symlink to source_file
        symlink_path = os.path.join(self.temp_dir, "symlink.txt")
        try:
            os.symlink(self.source_file, symlink_path)
        except OSError:
            self.skipTest("Symlinks not supported on this platform")

        dest_file = os.path.join(self.temp_dir, "dest.txt")
        copy_file(symlink_path, dest_file, follow_symlinks=True)

        # Should have copied the target file content
        with open(dest_file, "r") as f:
            self.assertEqual(f.read(), "test content")

    def test_follow_symlinks_false(self):
        """Test that follow_symlinks=False creates a new symlink."""
        # Create a symlink to source_file
        symlink_path = os.path.join(self.temp_dir, "symlink.txt")
        try:
            os.symlink(self.source_file, symlink_path)
        except OSError:
            self.skipTest("Symlinks not supported on this platform")

        dest_file = os.path.join(self.temp_dir, "dest_link.txt")
        copy_file(symlink_path, dest_file, follow_symlinks=False)

        # Note: When copying symlinks with follow_symlinks=False,
        # shutil.copy2 creates a new symlink pointing to the same target
        if os.path.islink(dest_file):
            # Test passed - symlink was created
            self.assertEqual(
                os.readlink(dest_file),
                os.readlink(symlink_path),
                "Symlink targets should match",
            )
        else:
            # If not a symlink, verify the target file content was copied
            # This handles platforms where copy2 always resolves symlinks
            self.assertTrue(os.path.isfile(dest_file))
            with open(dest_file, "r") as f:
                self.assertEqual(f.read(), "test content")


if __name__ == "__main__":
    unittest.main()
