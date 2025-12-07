import sys
import odak


def test():


    input_dict = {"a": 1, "b": 2, "c": 3}
    keys_to_keep = ["a", "b"]
    expected_output = {"a": 1, "b": 2}
    result = odak.tools.copy_dict_with_keys(d = input_dict, keys = keys_to_keep)
    assert result == expected_output


    input_dict = {"x": 5, "y": 6}
    keys_to_keep = ["z", "w"]
    expected_output = {}
    result = odak.tools.copy_dict_with_keys(d = input_dict, keys = keys_to_keep)
    assert result == expected_output


    input_dict = {}
    keys_to_keep = ["a", "b"]
    expected_output = {}
    result = odak.tools.copy_dict_with_keys(d = input_dict, keys = keys_to_keep)
    assert result == expected_output


    input_dict = {"name": "test", 123: [4,5,6], True: False}
    keys_to_keep = ["name", 123]
    expected_output = {"name": "test", 123: [4,5,6]}
    result = odak.tools.copy_dict_with_keys(d = input_dict, keys = keys_to_keep)
    assert result == expected_output



if __name__ == '__main__':
    sys.exit(test())
