import unittest
from odak.tools import get_base_filename

class TestGetBaseFilename(unittest.TestCase):
    def test_normal_file(self):
        self.assertEqual(get_base_filename("/path/to/file.txt"), ('file', '.txt'))

    def test_no_extension(self):
        self.assertEqual(get_base_filename("/path/to/file"), ('file', ''))

    def test_directory(self):
        self.assertEqual(get_base_filename("/path/to/dir"), ('dir', ''))

    def test_special_chars(self):
        self.assertEqual(get_base_filename("/path/to/file@#%.txt"), ('file@#%', '.txt'))

if __name__ == '__main__':
    unittest.main()