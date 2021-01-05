import operator
from functools import reduce
from typing import List, Dict, Union


def get_dict_val(root:Dict, keys:Union[str, List[str]]):
    """
    Access a nested object in root by item sequence.

    Args:
        root: Dict
            target object for accessing the value
        keys: Union[str, List[str]]
            a key or a list of key (for nested structure objecy) name
            to traverse through the Dict object 

    Examples::

        >>> obj = {"a": {"b": 100, "c": 0}, "d": 1}
        >>> get_dict_val(obj, "d")
        1

        >>> obj = {"a": {"b": 100, "c": 0}, "d": 1}
        >>> get_dict_val(obj, "a")
        {"b": 100, "c": 0}

        >>> obj = {"a": {"b": 100, "c": 0}, "d": 1}
        >>> get_dict_val(obj, ["a","b"])
        100
    """
    return reduce(operator.getitem, keys, root)
