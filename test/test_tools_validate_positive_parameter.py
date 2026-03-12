import sys
import numpy as np
import torch
import odak


def test():
    # Test with positive float value
    result = odak.tools.validate_positive_parameter(5.0, "test_value")
    assert result == 5.0

    # Test with positive int value
    result = odak.tools.validate_positive_parameter(10, "test_value")
    assert result == 10

    # Test with zero value (should raise error when allow_zero=False)
    try:
        odak.tools.validate_positive_parameter(0, "test_value", allow_zero=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be positive" in str(e)

    # Test with negative value (should raise error)
    try:
        odak.tools.validate_positive_parameter(-5.0, "test_value", allow_zero=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be positive" in str(e)

    # Test with zero value and allow_zero=True (should pass)
    result = odak.tools.validate_positive_parameter(0, "test_value", allow_zero=True)
    assert result == 0

    # Test with negative value and allow_zero=True (should raise error)
    try:
        odak.tools.validate_positive_parameter(-5.0, "test_value", allow_zero=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be non-negative" in str(e)

    # Test with numpy array containing positive values
    arr = np.array([1.0, 2.0, 3.0])
    result = odak.tools.validate_positive_parameter(arr, "array_value")
    assert np.array_equal(result, arr)

    # Test with numpy array containing zero (should raise error when allow_zero=False)
    arr_with_zero = np.array([1.0, 0.0, 3.0])
    try:
        odak.tools.validate_positive_parameter(
            arr_with_zero, "array_value", allow_zero=False
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "contains negative or zero values" in str(e)

    # Test with numpy array containing negative value (should raise error)
    arr_with_neg = np.array([1.0, -2.0, 3.0])
    try:
        odak.tools.validate_positive_parameter(
            arr_with_neg, "array_value", allow_zero=False
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "contains negative or zero values" in str(e)

    # Test with numpy array containing only positive values and allow_zero=True (should pass)
    arr_positive = np.array([1.0, 2.0, 3.0])
    result = odak.tools.validate_positive_parameter(
        arr_positive, "array_value", allow_zero=True
    )
    assert np.array_equal(result, arr_positive)

    # Test with numpy array containing zero and allow_zero=True (should pass)
    arr_with_zero = np.array([1.0, 0.0, 3.0])
    result = odak.tools.validate_positive_parameter(
        arr_with_zero, "array_value", allow_zero=True
    )
    assert np.array_equal(result, arr_with_zero)

    # Test with torch Tensor containing positive values
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = odak.tools.validate_positive_parameter(tensor, "tensor_value")
    assert torch.equal(result, tensor)

    # Test with torch Tensor containing zero (should raise error when allow_zero=False)
    tensor_with_zero = torch.tensor([1.0, 0.0, 3.0])
    try:
        odak.tools.validate_positive_parameter(
            tensor_with_zero, "tensor_value", allow_zero=False
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "contains negative or zero values" in str(e)

    # Test with torch Tensor containing negative value (should raise error)
    tensor_with_neg = torch.tensor([1.0, -2.0, 3.0])
    try:
        odak.tools.validate_positive_parameter(
            tensor_with_neg, "tensor_value", allow_zero=False
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "contains negative or zero values" in str(e)

    # Test with torch Tensor containing only positive values and allow_zero=True (should pass)
    tensor_positive = torch.tensor([1.0, 2.0, 3.0])
    result = odak.tools.validate_positive_parameter(
        tensor_positive, "tensor_value", allow_zero=True
    )
    assert torch.equal(result, tensor_positive)

    # Test with torch Tensor containing zero and allow_zero=True (should pass)
    tensor_with_zero = torch.tensor([1.0, 0.0, 3.0])
    result = odak.tools.validate_positive_parameter(
        tensor_with_zero, "tensor_value", allow_zero=True
    )
    assert torch.equal(result, tensor_with_zero)

    # Test with non-numeric value (should raise TypeError)
    try:
        odak.tools.validate_positive_parameter("not a number", "string_value")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "must be numeric" in str(e)

    print("All tests passed!")


if __name__ == "__main__":
    test()
