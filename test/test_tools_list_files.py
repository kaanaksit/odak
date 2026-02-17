import unittest
from odak.tools import list_files
from unittest.mock import patch


class TestListFiles(unittest.TestCase):
    def test_normal_files(self):
        with patch("os.listdir", return_value=["overrides/main.html"]):
            self.assertEqual(list_files("overrides/"), ["overrides/main.html"])

    def test_directories(self):
        with patch("os.listdir", return_value=["dir1", "dir2"]):
            self.assertEqual(list_files("/path"), [])

    def test_error(self):
        with patch("os.listdir", side_effect=Exception("Permission denied")):
            self.assertEqual(list_files("/path"), [])


if __name__ == "__main__":
    unittest.main()
