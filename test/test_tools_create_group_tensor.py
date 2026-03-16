import sys
import torch
import odak


def test():
    # Test with basic percentages that sum to 1.0
    result = odak.tools.create_group_tensor(10, [0.5, 0.3, 0.2])
    assert len(result) == 10
    assert result.dtype == torch.long
    assert result[0] == 0  # First element in group 0
    assert (result == 0).sum() == 5  # 5 elements in group 0
    assert (result == 1).sum() == 3  # 3 elements in group 1
    assert (result == 2).sum() == 2  # 2 elements in group 2

    # Test with equal distribution
    result = odak.tools.create_group_tensor(100, [0.25, 0.25, 0.25, 0.25])
    assert len(result) == 100
    assert (result == 0).sum() == 25
    assert (result == 1).sum() == 25
    assert (result == 2).sum() == 25
    assert (result == 3).sum() == 25

    # Test with single group
    result = odak.tools.create_group_tensor(10, [1.0])
    assert len(result) == 10
    assert all(r == 0 for r in result.tolist())

    # Test with small number of elements
    result = odak.tools.create_group_tensor(5, [0.4, 0.6])
    assert len(result) == 5
    group_counts = [(result == i).sum() for i in range(2)]
    assert sum(group_counts) == 5  # Total should be 5

    # Test with zero percentage group (should not appear in result)
    result = odak.tools.create_group_tensor(10, [0.5, 0.0, 0.5])
    assert len(result) == 10
    assert all((result != 1).tolist())  # No elements in group 1

    # Test with uneven distribution due to rounding
    result = odak.tools.create_group_tensor(10, [0.34, 0.33, 0.33])
    assert len(result) == 10
    group_counts = [(result == i).sum() for i in range(3)]
    assert sum(group_counts) == 10  # Total should be exactly 10

    # Test with many groups
    result = odak.tools.create_group_tensor(100, [0.1] * 10)
    assert len(result) == 100
    for i in range(10):
        assert (result == i).sum() == 10

    # Test with percentages at boundaries
    result = odak.tools.create_group_tensor(10, [0.0, 0.0, 1.0])
    assert len(result) == 10
    assert all((result == 2).tolist())  # All elements in last group

    # Test with percentages at boundaries (reverse)
    result = odak.tools.create_group_tensor(10, [1.0, 0.0, 0.0])
    assert len(result) == 10
    assert all((result == 0).tolist())  # All elements in first group

    # Test with percentages at boundaries (middle group only)
    result = odak.tools.create_group_tensor(10, [0.0, 1.0, 0.0])
    assert len(result) == 10
    assert all((result == 1).tolist())  # All elements in middle group

    # Test with percentages that don't sum to 1.0 (should raise error)
    try:
        odak.tools.create_group_tensor(10, [0.5, 0.3, 0.1])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must sum to 1.0" in str(e)

    # Test with percentages that sum to more than 1.0 (should raise error)
    try:
        odak.tools.create_group_tensor(10, [0.5, 0.5, 0.5])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must sum to 1.0" in str(e)

    # Test with negative percentage (should raise error)
    try:
        odak.tools.create_group_tensor(10, [0.5, -0.1, 0.6])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "between 0.0 and 1.0" in str(e)

    # Test with percentage greater than 1.0 (should raise error)
    try:
        odak.tools.create_group_tensor(10, [1.5, -0.5])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "between 0.0 and 1.0" in str(e)

    # Test with non-integer number_of_elements (should raise TypeError)
    try:
        odak.tools.create_group_tensor(10.5, [0.5, 0.5])
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "must be an integer" in str(e)

    # Test with float number_of_elements (should raise TypeError)
    try:
        odak.tools.create_group_tensor(10.0, [0.5, 0.5])
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "must be an integer" in str(e)

    # Test with very small percentages (edge case for rounding)
    result = odak.tools.create_group_tensor(1000, [0.001, 0.999])
    assert len(result) == 1000
    group_counts = [(result == i).sum() for i in range(2)]
    assert sum(group_counts) == 1000

    # Test with zero elements (should return empty tensor)
    result = odak.tools.create_group_tensor(0, [0.5, 0.5])
    assert len(result) == 0
    assert result.dtype == torch.long

    # Test with very large number of elements
    result = odak.tools.create_group_tensor(10000, [0.3, 0.7])
    assert len(result) == 10000
    assert (result == 0).sum() == 3000
    assert (result == 1).sum() == 7000

    # Test with percentages that require exact distribution via last group fix
    result = odak.tools.create_group_tensor(7, [0.35, 0.65])
    assert len(result) == 7
    group_counts = [(result == i).sum() for i in range(2)]
    assert sum(group_counts) == 7  # Should be exactly 7

    # Test with number_of_groups parameter (equal distribution)
    result = odak.tools.create_group_tensor(10, number_of_groups=4)
    assert len(result) == 10
    assert result.dtype == torch.long
    group_counts = [(result == i).sum() for i in range(4)]
    assert sum(group_counts) == 10  # Total should be 10

    # Test with number_of_groups parameter - exactly divisible case
    result = odak.tools.create_group_tensor(12, number_of_groups=3)
    assert len(result) == 12
    assert (result == 0).sum() == 4
    assert (result == 1).sum() == 4
    assert (result == 2).sum() == 4

    # Test with number_of_groups parameter - single group
    result = odak.tools.create_group_tensor(10, number_of_groups=1)
    assert len(result) == 10
    assert all(r == 0 for r in result.tolist())

    # Test with number_of_groups parameter - many groups
    result = odak.tools.create_group_tensor(20, number_of_groups=5)
    assert len(result) == 20
    for i in range(5):
        assert (result == i).sum() == 4

    # Test with number_of_groups parameter - unequal distribution due to remainder
    result = odak.tools.create_group_tensor(10, number_of_groups=3)
    assert len(result) == 10
    group_counts = [(result == i).sum() for i in range(3)]
    assert sum(group_counts) == 10  # Total should be exactly 10

    # Test with group_percentages=None without number_of_groups (should raise error)
    try:
        odak.tools.create_group_tensor(10, group_percentages=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "number_of_groups must be provided" in str(e)

    print("All tests passed!")


if __name__ == "__main__":
    test()
