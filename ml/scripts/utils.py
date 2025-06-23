import collections.abc

def recursive_dict_update(original_dict, update_dict):
    """
    Recursively updates a dictionary with values from another dictionary.
    Handles nested dictionaries by merging them.
    """
    for key, value in update_dict.items():
        if key in original_dict and isinstance(original_dict[key], dict) and isinstance(value, dict):
            # If both values are dictionaries, recurse
            original_dict[key] = recursive_dict_update(original_dict[key], value)
        else:
            # Otherwise, update or add the key-value pair
            original_dict[key] = value
    return original_dict


def dict_update(d, u):
    """
    Recursively update a dict with another dict.
    This is a deep update, meaning that if a key in the first dict
    has a dict as its value, and the second dict has a key with
    the same name, the value in the first dict will be updated
    with the value from the second dict.
    Keys in the second dict are the ones iterated over. 
    If the value in the second dict is not a dict, it will
    overwrite the value in the first dict.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d