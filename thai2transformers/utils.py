import os
import operator
from functools import reduce
from typing import Tuple, Dict, Union, List
from thai2transformers import __file__ as thai2transformers_file


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
        _results = []
        for item in root[keys[0]]:
            _results.append(item[keys[1]])
        return _results
    
    return None

def get_thai2transformers_path() -> str:
    """
    This function returns full path of thai2transformers code; copied from pythainlp
    :return: full path of :mod:`thai2transformers` code
    :rtype: str
    :Example:
    ::
        from thai2transformers.utils import get_thai2transformers_path
        get_thai2transformers_path()
        # output: '/usr/local/lib/python3.6/dist-packages/thai2transformers'
    """
    return os.path.dirname(thai2transformers_file)
