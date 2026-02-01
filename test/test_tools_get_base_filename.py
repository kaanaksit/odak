import os
import sys
import unittest

from odak.tools import get_base_filename


class TestGetBaseFilename(unittest.TestCase):

    def test_get_base_filename_with_extension(self):
        self.assertEqual(get_base_filename("hello.txt"), ("hello", ".txt"))

    def test_get_base_filename_no_extension(self):
        self.assertEqual(get_base_filename("hello"), ("hello", ""))

    def test_get_base_filename_with_multiple_dots(self):
        self.assertEqual(get_base_filename("file.name.tar.gz"), ("file.name.tar", ".gz"))

    def test_get_base_filename_with_path(self):
        self.assertEqual(get_base_filename("//to/my/file.txt"), ("file", ".txt"))


if __name__ == '__main__':
    sys.exit(unittest.main())
