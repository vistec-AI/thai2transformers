import operator
from functools import reduce
from typing import Tuple, Dict, Union, List


def get_dict_val(root:Dict, keys:Union[str, List[str]]):
    """
    Access a nested object in root by item sequence.

    Args:
        root: Dict
            target object for accessing the value
        keys: Union[str, Tuple[str, str]]
            a key or a list of key (for nested structure objecy) name
            to traverse through the Dict object 

    Examples::

        >>> obj = {"a": [1,2,3]}
        >>> get_dict_val(obj, "a")
        [1,2,3]

        >>> obj = {"a": [ {"aa": 100, "bb": 0}, {"aa": 2, "bb": 5 } ] }
        >>> get_dict_val(obj, ("a", "aa"))
        [100, 2]

    """
    if type(keys) == str:
        return root[keys]
    elif type(keys) == list:
        return root[keys[0]][keys[1]]
    
    return None
