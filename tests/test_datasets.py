import unittest
import pytest
import os
import sys
import shutil
from typing import Collection, Iterable, List, Union, Dict, Callable
from datasets import load_dataset
from sklearn import preprocessing
from transformers import (
    AutoTokenizer
)
from transformers.tokenization_utils import (
    PreTrainedTokenizer,
)

from thai2transformers.datasets import (
    SequenceClassificationDataset
)

from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer
)

from thai2transformers.conf import (
    Task
)
from thai2transformers.utils import (
    get_dict_val
)

class TestSequenceClassificationDataset(unittest.TestCase):

    def test_init_default_1(self):

        def preprocessor(x: str) -> str:
            return x.replace('.', 'DOT')

        seq_cls_dataset = SequenceClassificationDataset(
            tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
            data_dir='./tests/mockup/sequence_classification_datsaets/a',
            preprocessor=preprocessor
        )
        self.assertIsNotNone(seq_cls_dataset)

    def test_init_default_2(self):
        
        def preprocessor_1(x: str) -> str:
            return x.replace('.', 'DOT')
        def preprocessor_2(x: str) -> str:
            return x.replace('DOT', 'dot')

        seq_cls_dataset = SequenceClassificationDataset(
            tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
            data_dir='./tests/mockup/sequence_classification_datsaets/a',
            preprocessor=lambda x: list(map(lambda f: f(x), [preprocessor_1, preprocessor_2]))[0]
        )
        self.assertIsNotNone(seq_cls_dataset)

    def test_init_default_3(self):
        
        seq_cls_dataset = SequenceClassificationDataset(
            tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
            data_dir='./tests/mockup/sequence_classification_datsaets/a',
            preprocessor=None
        )
        self.assertIsNotNone(seq_cls_dataset)
    

    def test_init_default_with_error_1(self):

        # preprocess should return object with type string

        def preprocessor(x: str) -> int:
            return 100

        with self.assertRaises(AssertionError) as context:
            seq_cls_dataset = SequenceClassificationDataset(
                tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
                data_dir='./tests/mockup/sequence_classification_datsaets/a',
                preprocessor=preprocessor
            )
        
        self.assertEqual(type(AssertionError()), type(context.exception))

    def test_init_default_with_error_2(self):

        # preprocess should not return a list of strings

        def preprocessor(x: str) -> List[str]:
            return x.split('.')

        with self.assertRaises(ValueError) as context:
            seq_cls_dataset = SequenceClassificationDataset(
                tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
                data_dir='./tests/mockup/sequence_classification_datsaets/a',
                preprocessor=preprocessor
            )
        
        self.assertEqual(type(ValueError()), type(context.exception))


    def test_init_from_dataset(self):

        daataset = load_dataset('wongnai_reviews')
        daataset['train'] = daataset['train'][:100]
        text_column_name = 'review_body'
        label_column_name = 'star_rating'
        max_length = 10

        def preprocessor(x: str) -> str:
            return x.replace(' ', '<_>')

        seq_cls_dataset = SequenceClassificationDataset.from_dataset(
            task=Task.MULTICLASS_CLS,
            tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
            dataset=daataset['train'],
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            max_length=max_length,
            preprocessor=preprocessor
        )
        self.assertIsNotNone(seq_cls_dataset)
        self.assertEqual(len(seq_cls_dataset[0]['input_ids']), max_length)

    # Init with LabelEncoder instance
    def test_init_from_dataset_2(self):


        dataset = load_dataset('wongnai_reviews')
        dataset['train'] = dataset['train'][:100]
        text_column_name = 'review_body'
        label_column_name = 'star_rating'
        max_length = 10

        def preprocessor(x: str) -> str:
            return x.replace(' ', '<_>')

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(dataset['train'][label_column_name])
        print(label_encoder)

        seq_cls_dataset = SequenceClassificationDataset.from_dataset(
            task=Task.MULTICLASS_CLS,
            tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
            dataset=dataset['train'],
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            max_length=max_length,
            preprocessor=preprocessor,
            label_encoder=label_encoder
        )
        self.assertIsNotNone(seq_cls_dataset)
        self.assertEqual(len(seq_cls_dataset[0]['input_ids']), max_length)

    def test_init_from_dataset_with_error_1(self):

        daataset = load_dataset('wongnai_reviews')
        daataset['train'] = daataset['train'][:100]
        text_column_name = 'review_body'
        label_column_name = 'star_rating'
        def preprocessor(x: str) -> List[str]:
            return x.split(' ')


        with self.assertRaises(ValueError) as context:
            seq_cls_dataset = SequenceClassificationDataset.from_dataset(
                task=Task.MULTICLASS_CLS,
                tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
                dataset=daataset['train'],
                text_column_name=text_column_name,
                label_column_name=label_column_name,
                max_length=10,
                preprocessor=preprocessor
            )

        self.assertEqual(type(ValueError()), type(context.exception))

        