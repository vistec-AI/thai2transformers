import unittest
import pytest
import os
import sys
import shutil
from typing import Collection, Iterable, List, Union, Dict, Callable
from functools import partial

import torch
from datasets import load_dataset
from sklearn import preprocessing

from transformers import (
    CamembertTokenizer
)
from thai2transformers.conf import (
    Task
)
from thai2transformers.pipelines.finetuning import (
    TokenClassificationFinetuningPipeline
)

class TokenClassificationFinetuningPipelineTest(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp/seq_cls_finetuning_pipeline'):
            shutil.rmtree('./tmp/seq_cls_finetuning_pipeline')

    def test_token_cls_finetuning_pipeline_init(self):

        token_cls_finetuner = TokenClassificationFinetuningPipeline(
            task='chunk_level_classification'
        )
        self.assertIsNotNone(token_cls_finetuner)
        
        token_cls_finetuner = TokenClassificationFinetuningPipeline(
            task='token_level_classification'
        )
        self.assertIsNotNone(token_cls_finetuner)

        token_cls_finetuner = TokenClassificationFinetuningPipeline(
            task=Task.CHUNK_LEVEL_CLS
        )
        self.assertIsNotNone(token_cls_finetuner)
        token_cls_finetuner = TokenClassificationFinetuningPipeline(
            task=Task.TOKEN_LEVEL_CLS
        )
        self.assertIsNotNone(token_cls_finetuner)


    def test_token_cls_finetuning_pipeline_wangchanberta_spm_camembert_on_thainer_ner(self):

        token_cls_finetuner = TokenClassificationFinetuningPipeline(
            task='chunk_level_classification'
        )
        dataset_name = 'thainer'
        text_col_name = 'tokens'
        label_col_name = 'ner_tags'
        dataset = load_dataset(dataset_name)
        num_labels = dataset['train'].features['ner_tags'].feature.num_classes

        token_cls_finetuner.load_dataset(dataset_name_or_path=dataset_name,
                                        text_column_name=text_col_name,
                                        label_column_name=label_col_name)

        self.assertIsNotNone(token_cls_finetuner._dataset)
        
        token_cls_finetuner.load_tokenizer(tokenizer_cls=CamembertTokenizer,
                                         name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(token_cls_finetuner.finetuner.tokenizer)

        token_cls_finetuner.process_dataset(
                        space_token='<_>',
                        train_dataset_name='train',
                        val_dataset_name='val',
                        test_dataset_name='test',
                        num_train_examples=100,
                        num_val_examples=100,
                        num_test_examples=100,
                        max_length=416)
        self.assertIsNotNone(token_cls_finetuner.train_dataset)                
        self.assertIsNone(token_cls_finetuner.val_dataset)                
        self.assertIsNone(token_cls_finetuner.test_dataset)                

        self.assertIsNotNone(token_cls_finetuner._dataset)

        token_cls_finetuner.load_model(name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(token_cls_finetuner.finetuner.model)
        self.assertEqual(token_cls_finetuner.num_labels, num_labels)
        self.assertEqual(token_cls_finetuner.finetuner.model.num_labels, num_labels)

        training_args = {
            'max_steps': 10,
            'warmup_steps': 2,
            'logging_steps': 1,
            'run_name': None,
            'no_cuda': not torch.cuda.is_available(),
        }
        
        output_dir = './tmp/token_cls_finetuning_pipeline/wangchanbert-base-att-spm-uncased/thainer-ner'

        token_cls_finetuner.finetune(
            output_dir=output_dir,
            eval_on_test_set=False,
            **training_args
        )

        self.assertTrue(os.path.exists(
            os.path.join(output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ))
    
   
    def test_token_cls_finetuning_pipeline_wangchanberta_spm_camembert_on_thainer_pos(self):

        token_cls_finetuner = TokenClassificationFinetuningPipeline(
            task='token_level_classification'
        )
        dataset_name = 'thainer'
        text_col_name = 'tokens'
        label_col_name = 'pos_tags'
        dataset = load_dataset(dataset_name)
        num_labels = dataset['train'].features['pos_tags'].feature.num_classes

        token_cls_finetuner.load_dataset(dataset_name_or_path=dataset_name,
                                        text_column_name=text_col_name,
                                        label_column_name=label_col_name)

        self.assertIsNotNone(token_cls_finetuner._dataset)
        
        token_cls_finetuner.load_tokenizer(tokenizer_cls=CamembertTokenizer,
                                         name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(token_cls_finetuner.finetuner.tokenizer)

        token_cls_finetuner.process_dataset(
                        space_token='<_>',
                        train_dataset_name='train',
                        val_dataset_name='val',
                        test_dataset_name='test',
                        num_train_examples=100,
                        num_val_examples=100,
                        num_test_examples=100,
                        max_length=416)
        self.assertIsNotNone(token_cls_finetuner.train_dataset)                
        self.assertIsNone(token_cls_finetuner.val_dataset)                
        self.assertIsNone(token_cls_finetuner.test_dataset)                

        self.assertIsNotNone(token_cls_finetuner._dataset)

        token_cls_finetuner.load_model(name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(token_cls_finetuner.finetuner.model)
        self.assertEqual(token_cls_finetuner.num_labels, num_labels)
        self.assertEqual(token_cls_finetuner.finetuner.model.num_labels, num_labels)

        training_args = {
            'max_steps': 10,
            'warmup_steps': 2,
            'logging_steps': 1,
            'run_name': None,
            'no_cuda': not torch.cuda.is_available(),
        }
        
        output_dir = './tmp/token_cls_finetuning_pipeline/wangchanbert-base-att-spm-uncased/thainer-pos'

        token_cls_finetuner.finetune(
            output_dir=output_dir,
            eval_on_test_set=False,
            **training_args
        )

        self.assertTrue(os.path.exists(
            os.path.join(output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ))
    
   