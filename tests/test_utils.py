import unittest
import pytest
import os
import sys
import shutil
from typing import Collection, Iterable, List, Union, Dict, Callable

from thai2transformers.utils import (
    get_dict_val
)


class UtilsTest(unittest.TestCase):

    def test_get_dict_val_1(self):
        
        dataset = {
            'translation': [ 'th text rating: 0', 'th text rating: 4'],
            'rating': [0,4]
        }
        text_col_name =  'translation'
        label_col_name = 'rating'

        result = get_dict_val(dataset, text_col_name)

        self.assertEqual(
            result,
            ['th text rating: 0', 'th text rating: 4']
        )
        
        result = get_dict_val(dataset, label_col_name)
        self.assertEqual(
            result,
            [0, 4]
        )

    def test_get_dict_val_2(self):
        
        dataset = {
            'translation': [
                {'th': 'th text rating: 0', 'en':'en text rating: 0'},
                {'th': 'th text rating: 4', 'en':'en text rating: 4'}
            ],
            'rating': [0,4]
        }
        text_col_name =  'translation.th'
        label_col_name = 'rating'

        result = get_dict_val(dataset, text_col_name)

        self.assertEqual(
            result,
            ['th text rating: 0', 'th text rating: 4']
        )

        result = get_dict_val(dataset, label_col_name)
        self.assertEqual(
            result,
            [0, 4]
        )