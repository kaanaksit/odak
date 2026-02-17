import unittest
import odak
import shutil


class TestListDirectories(unittest.TestCase):

    def test_list_directories_non_recursive(
        self,
        output_directory="test_output",
    ):
        odak.tools.check_directory(output_directory)
        odak.tools.check_directory("{}/temp_test_dir".format(output_directory))
        odak.tools.check_directory("{}/temp_test_dir/subdir1".format(output_directory))
        odak.tools.check_directory("{}/temp_test_dir/subdir2".format(output_directory))
        open("{}/temp_test_dir/file1.txt".format(output_directory), "w").close()

        expected_directories = ["subdir1", "subdir2"]
        actual_directories = odak.tools.list_directories(
            path="{}/temp_test_dir".format(output_directory), recursive=False
        )
        self.assertEqual(actual_directories, expected_directories)
        shutil.rmtree("{}/temp_test_dir".format(output_directory))

    def test_list_directories_recursive(
        self,
        output_directory="test_output",
    ):
        odak.tools.check_directory(output_directory)
        odak.tools.check_directory("{}/temp_test_dir".format(output_directory))
        odak.tools.check_directory("{}/temp_test_dir/subdir1".format(output_directory))
        odak.tools.check_directory("{}/temp_test_dir/subdir2".format(output_directory))
        odak.tools.check_directory(
            "{}/temp_test_dir/subdir2/subsubdir1".format(output_directory)
        )
        open("{}/temp_test_dir/file1.txt".format(output_directory), "w").close()

        expected_directories = ["subdir1", "subdir2", "subsubdir1"]
        actual_directories = odak.tools.list_directories(
            path="{}/temp_test_dir".format(output_directory), recursive=True
        )
        self.assertEqual(actual_directories, expected_directories)
        shutil.rmtree("{}/temp_test_dir".format(output_directory))


if __name__ == "__main__":
    unittest.main()
