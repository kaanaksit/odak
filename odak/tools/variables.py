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
