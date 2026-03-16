import numpy as np
import torch
from ..log import logger


def copy_dict_with_keys(d: dict, keys=None):
    """
    Copy elements from a dictionary based on specified keys.

    This function creates and returns a new dictionary that includes only the
    key-value pairs from the input dictionary `d` whose keys are listed in
    `keys`. If no list of keys is provided (i.e., `keys` is None), all key-value
    pairs from `d` will be included.


    Parameters
    ----------
    d        : dict
               The original dictionary to copy elements from.
    keys     : list-like or None, optional
               A list of keys to include in the new dictionary. If not provided,
               all keys are copied. Defaults to None.


    Returns
    -------
    new_dict : dict
               A new dictionary containing only the specified key-value pairs.

    Examples
    --------
    >>> original_dict = {'a': 1, 'b': 2, 'c': 3}
    >>> selected_keys = ['a', 'c']
    >>> result = copy_dict_with_keys(original_dict, keys=selected_keys)
    >>> print(result)  # Output: {'a': 1, 'c': 3}
    """

    if not isinstance(d, dict):
        return None

    new_dict = {}
    if keys is None:
        new_dict.update(d)
    else:
        for key in keys:
            if key in d:
                new_dict[key] = d[key]

    return new_dict


def validate_positive_parameter(value, name="value", allow_zero=False):
    """
    Validate that a parameter is positive (or non-negative if allow_zero).

    Parameters
    ----------
    value             : float or np.ndarray or torch.Tensor
                        Value to validate.
    name              : str
                        Parameter name for error messages.
    allow_zero        : bool
                        If True, zero values are allowed. Default: False.

    Returns
    -------
    validated_value   : same type as input
                        The validated value (unchanged if valid).

    Raises
    ------
    ValueError        : If value is negative or zero (when not allowed).
    TypeError         : If value is not a number.
    """
    if isinstance(value, (int, float)):
        if not allow_zero and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        elif allow_zero and value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    elif isinstance(value, np.ndarray):
        if not allow_zero and np.any(value <= 0):
            raise ValueError(f"{name} contains negative or zero values")
        elif allow_zero and np.any(value < 0):
            raise ValueError(f"{name} contains negative values")

    elif isinstance(value, torch.Tensor):
        if not allow_zero and torch.any(value <= 0):
            raise ValueError(f"{name} contains negative or zero values")
        elif allow_zero and torch.any(value < 0):
            raise ValueError(f"{name} contains negative values")

    else:
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")

    logger.debug(
        f"Validated {name}: {value if isinstance(value, (int, float)) else f'{value.shape}'}"
    )
    return value


def create_group_tensor(
    number_of_elements: int,
    group_percentages: list,
) -> torch.Tensor:
    """
    Create a tensor assigning elements to groups based on percentages.

    This function distributes the given number of elements into groups according
    to the specified percentages. Each element is assigned a group ID based on
    the percentage distribution provided.

    Parameters
    ----------
    number_of_elements : int
        Total number of elements to distribute among groups.
    group_percentages : list of float
        List of percentages (0.0 to 1.0) specifying the proportion of elements
        in each group. Must sum to 1.0.

    Returns
    -------
    torch.Tensor
        A 1D tensor of shape (number_of_elements,) where each element contains
        its assigned group ID (0-indexed). Dtype is torch.long.

    Raises
    ------
    ValueError
        If group_percentages does not sum to 1.0.
        If any percentage is outside the range [0.0, 1.0].
    TypeError
        If number_of_elements is not an integer.

    Examples
    --------
    >>> # Create a tensor with 10 elements distributed into 3 groups
    >>> percentages = [0.5, 0.3, 0.2]
    >>> result = create_group_tensor(10, percentages)
    >>> print(result)
    tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])

    >>> # Create a tensor with 100 elements equally distributed
    >>> percentages = [0.34, 0.33, 0.33]
    >>> result = create_group_tensor(100, percentages)
    >>> print(len(result))
    100
    """
    # Validate number_of_elements type
    if not isinstance(number_of_elements, int):
        raise TypeError(
            f"number_of_elements must be an integer, got {type(number_of_elements).__name__}"
        )

    # Validate input percentages sum to 1.0
    total_percentage = sum(group_percentages)
    if abs(total_percentage - 1.0) > 1e-6:
        raise ValueError(
            f"group_percentages must sum to 1.0, got {total_percentage}"
        )

    # Validate all percentages are within valid range
    if any(p < 0 or p > 1 for p in group_percentages):
        raise ValueError(
            "All percentages must be between 0.0 and 1.0"
        )

    # Calculate number of elements per group
    group_sizes = []
    remaining_elements = number_of_elements
    cumulative_percentage = 0.0  # Reserved for potential future use

    for i, pct in enumerate(group_percentages):
        if i == len(group_percentages) - 1:
            # Last group gets remaining elements to ensure exact total
            group_sizes.append(remaining_elements)
        else:
            size = int(round(number_of_elements * pct))
            group_sizes.append(size)
            remaining_elements -= size

    # Create tensor with group assignments
    groups = []
    for group_id, size in enumerate(group_sizes):
        if size > 0:
            groups.extend([group_id] * size)

    return torch.tensor(groups, dtype=torch.long)
