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
    SequenceClassificationFinetuningPipeline
)

class SequenceClassificationFinetuningPipelineTest(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp/seq_cls_finetuning_pipeline'):
            shutil.rmtree('./tmp/seq_cls_finetuning_pipeline')

    def test_seq_cls_finetuning_pipeline_init(self):

        seq_cls_finetuner = SequenceClassificationFinetuningPipeline(
            task='multiclass_classification'
        )
        self.assertIsNotNone(seq_cls_finetuner)
        
        seq_cls_finetuner = SequenceClassificationFinetuningPipeline(
            task='multilabel_classification'
        )
        self.assertIsNotNone(seq_cls_finetuner)

        seq_cls_finetuner = SequenceClassificationFinetuningPipeline(
            task=Task.MULTICLASS_CLS
        )
        self.assertIsNotNone(seq_cls_finetuner)
        seq_cls_finetuner = SequenceClassificationFinetuningPipeline(
            task=Task.MULTILABEL_CLS
        )
        self.assertIsNotNone(seq_cls_finetuner)

    # @pytest.mark.skip(reason="already done")
    def test_seq_cls_finetuning_pipeline_wanchanberta_spm_camembert_on_wongnai(self):

        seq_cls_finetuner = SequenceClassificationFinetuningPipeline(
            task='multiclass_classification'
        )
        wongnai_dataset_name = 'wongnai_reviews'
        wongnai_text_col_name = 'review_body'
        wongnai_label_col_name = 'star_rating'
        wongnai_num_labels = 5

        seq_cls_finetuner.load_dataset(dataset_name_or_path=wongnai_dataset_name,
                                       text_column_name=wongnai_text_col_name,
                                       label_column_name=wongnai_label_col_name)

        self.assertIsNotNone(seq_cls_finetuner._dataset)
        
        seq_cls_finetuner.load_tokenizer(tokenizer_cls=CamembertTokenizer,
                                         name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(seq_cls_finetuner.finetuner.tokenizer)

        seq_cls_finetuner.process_dataset(
                        space_token='<_>',
                        train_dataset_name='train',
                        val_dataset_name='val',
                        test_dataset_name='test',
                        num_train_examples=100,
                        num_val_examples=100,
                        num_test_examples=100,
                        max_length=416)
        self.assertIsNotNone(seq_cls_finetuner.train_dataset)                
        self.assertIsNone(seq_cls_finetuner.val_dataset)                
        self.assertIsNotNone(seq_cls_finetuner.test_dataset)                

        self.assertIsNotNone(seq_cls_finetuner._dataset)

        seq_cls_finetuner.load_model(name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(seq_cls_finetuner.finetuner.model)
        self.assertEqual(seq_cls_finetuner.num_labels, wongnai_num_labels)
        self.assertEqual(seq_cls_finetuner.finetuner.model.num_labels, wongnai_num_labels)

        training_args = {
            'max_steps': 10,
            'warmup_steps': 1,
            'no_cuda': not torch.cuda.is_available(),
        }
        
        output_dir = './tmp/seq_cls_finetuning_pipeline/wangchanbert-base-att-spm-uncased/wongnai_reviews'

        eval_result = seq_cls_finetuner.finetune(
            output_dir=output_dir,
            eval_on_test_set=True,
            **training_args
        )

        self.assertIsNotNone(eval_result)
        print(eval_result)

        self.assertTrue(os.path.exists(
            os.path.join(output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ))
    
    def test_seq_cls_finetuning_pipeline_wanchanberta_spm_camembert_on_generated_reviews_enth(self):

        seq_cls_finetuner = SequenceClassificationFinetuningPipeline(
            task='multiclass_classification'
        )
        dataset_name = 'generated_reviews_enth'
        text_col_name =  'translation.th'
        label_col_name = 'review_star'
        num_labels = 5

        seq_cls_finetuner.load_dataset(dataset_name_or_path=dataset_name,
                                       text_column_name=text_col_name,
                                       label_column_name=label_col_name)

        self.assertIsNotNone(seq_cls_finetuner._dataset)
        
        seq_cls_finetuner.load_tokenizer(tokenizer_cls=CamembertTokenizer,
                                         name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(seq_cls_finetuner.finetuner.tokenizer)

        seq_cls_finetuner.process_dataset(
                        space_token='<_>',
                        train_dataset_name='train',
                        val_dataset_name='val',
                        test_dataset_name='test',
                        num_train_examples=100,
                        num_val_examples=100,
                        num_test_examples=100,
                        max_length=416)
        self.assertIsNotNone(seq_cls_finetuner.train_dataset)                
        self.assertIsNone(seq_cls_finetuner.val_dataset)                
        self.assertIsNotNone(seq_cls_finetuner.test_dataset)                

        self.assertIsNotNone(seq_cls_finetuner._dataset)

        seq_cls_finetuner.load_model(name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(seq_cls_finetuner.finetuner.model)
        self.assertEqual(seq_cls_finetuner.num_labels, num_labels)
        self.assertEqual(seq_cls_finetuner.finetuner.model.num_labels, num_labels)

        training_args = {
            'max_steps': 10,
            'warmup_steps': 1,
            'no_cuda': not torch.cuda.is_available(),
        }
        
        output_dir = './tmp/seq_cls_finetuning_pipeline/wangchanbert-base-att-spm-uncased/generated_reviews_enth'

        eval_result = seq_cls_finetuner.finetune(
            output_dir=output_dir,
            eval_on_test_set=True,
            **training_args
        )

        self.assertIsNotNone(eval_result)
        print(eval_result)

        self.assertTrue(os.path.exists(
            os.path.join(output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ))


    # @pytest.mark.skip(reason="already done")
    def test_multilabel_seq_cls_finetuning_pipeline_wanchanberta_spm_camembert_on_prachathai(self):

        seq_cls_finetuner = SequenceClassificationFinetuningPipeline(
            task='multilabel_classification'
        )
        prachathai_dataset_name = 'prachathai67k'
        prachathai_text_col_name = 'body_text'
        prachathai_label_col_name =  ['politics', 'human_rights', 'quality_of_life',
                                      'international', 'social', 'environment',
                                      'economics', 'culture', 'labor',
                                      'national_security', 'ict', 'education']
        prachathai_num_labels = 12

        seq_cls_finetuner.load_dataset(dataset_name_or_path=prachathai_dataset_name,
                                       text_column_name=prachathai_text_col_name,
                                       label_column_name=prachathai_label_col_name)

        self.assertIsNotNone(seq_cls_finetuner._dataset)
        
        seq_cls_finetuner.load_tokenizer(tokenizer_cls=CamembertTokenizer,
                                         name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(seq_cls_finetuner.finetuner.tokenizer)

        seq_cls_finetuner.process_dataset(
                        space_token='<_>',
                        train_dataset_name='train',
                        val_dataset_name='validattion',
                        test_dataset_name='test',
                        num_train_examples=100,
                        num_val_examples=100,
                        num_test_examples=100,
                        max_length=416)
        self.assertIsNotNone(seq_cls_finetuner.train_dataset)                
        self.assertIsNone(seq_cls_finetuner.val_dataset)                
        self.assertIsNotNone(seq_cls_finetuner.test_dataset)                

        self.assertIsNotNone(seq_cls_finetuner._dataset)

        seq_cls_finetuner.load_model(name_or_path='airesearch/wangchanberta-base-att-spm-uncased')

        self.assertIsNotNone(seq_cls_finetuner.finetuner.model)
        self.assertEqual(seq_cls_finetuner.num_labels, prachathai_num_labels)
        self.assertEqual(seq_cls_finetuner.finetuner.num_labels, prachathai_num_labels)
        self.assertEqual(seq_cls_finetuner.finetuner.model.num_labels, prachathai_num_labels)

        training_args = {
            'max_steps': 10,
            'warmup_steps': 1,
            'no_cuda': not torch.cuda.is_available(),
        }
        
        output_dir = './tmp/seq_cls_finetuning_pipeline/wangchanbert-base-att-spm-uncased/prachathai67k'

        eval_result = seq_cls_finetuner.finetune(
            output_dir=output_dir,
            eval_on_test_set=True,
            **training_args
        )

        self.assertIsNotNone(eval_result)
        print(eval_result)

        self.assertTrue(os.path.exists(
            os.path.join(output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ))
