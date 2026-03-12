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
