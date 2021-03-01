import unittest
import pytest
import os
import sys
import shutil
from typing import Collection, Iterable, List, Union, Dict, Callable
from functools import partial
from datasets import load_dataset
from sklearn import preprocessing

from transformers.testing_utils import require_torch
from thai2transformers.finetuner import (
    BaseFinetuner,
    TokenClassificationFinetuner,
)

from thai2transformers.datasets import (
    TokenClassificationDataset
)
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer
)
from thai2transformers import preprocess
from thai2transformers.utils import (
    get_dict_val
)
from thai2transformers.conf import Task

from transformers import (
    CamembertTokenizer,
    XLMRobertaTokenizer,
    BertTokenizer,
    HfArgumentParser,
    TrainingArguments
)

TOKENIZER_CLS_MAPPING = {
    'spm_camembert': CamembertTokenizer,
    'spm': ThaiRobertaTokenizer,
    'newmm': ThaiWordsNewmmTokenizer,
    'syllable': ThaiWordsSyllableTokenizer,
    'sefr_cut': FakeSefrCutTokenizer,
    'xlmr': XLMRobertaTokenizer,
    'mbert': BertTokenizer,
}

def _process_transformers(
    text: str,
    pre_rules: Collection[Callable] = [
        preprocess.fix_html,
        preprocess.rm_brackets,
        preprocess.replace_newlines,
        preprocess.rm_useless_spaces,
        preprocess.replace_spaces,
        preprocess.replace_rep_after,
    ],
    tok_func: Callable = preprocess.word_tokenize,
    post_rules: Collection[Callable] = [preprocess.ungroup_emoji, preprocess.replace_wrep_post],
    lowercase: bool = False
) -> str:
    if lowercase:
        text = text.lower()
    for rule in pre_rules:
        text = rule(text)
    toks = tok_func(text)
    for rule in post_rules:
        toks = rule(toks)
    return "".join(toks)


class TestTokenClassificationFinetuner(unittest.TestCase):

    def test_init(self):
        token_cls_finetuner = TokenClassificationFinetuner()
        self.assertIsNotNone(token_cls_finetuner)

    
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_spm_camembert(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['spm_camembert']
        
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(token_cls_finetuner.tokenizer.__class__.__name__,'CamembertTokenizer')
        self.assertEqual(token_cls_finetuner.tokenizer.additional_special_tokens,
                         ['<s>NOTUSED', '</s>NOTUSED', '<_>'])

    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_spm(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-spm'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['spm']
        
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(token_cls_finetuner.tokenizer.__class__.__name__,'ThaiRobertaTokenizer')
    
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_newmm(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-newmm'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['newmm']
        
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(token_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsNewmmTokenizer')
    
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_syllable(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-ssg'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['syllable']
        
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(token_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsSyllableTokenizer')

    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_sefr(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-sefr'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['sefr_cut']
        
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(token_cls_finetuner.tokenizer.__class__.__name__,'FakeSefrCutTokenizer')
    
    @require_torch
    def test_load_pretrained_tokenizer_xlmr(self):
        
        pretrained_tokenizer_name = 'xlm-roberta-base'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['xlmr']
        
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(token_cls_finetuner.tokenizer.__class__.__name__,'XLMRobertaTokenizer')
    
    #@pytest.mark.skip(reason="change api")
    @require_torch
    def test_load_pretrained_tokenizer_mbert(self):
        
        pretrained_tokenizer_name = 'bert-base-multilingual-cased'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['mbert']
        
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(token_cls_finetuner.tokenizer.__class__.__name__,'BertTokenizer')
    
    @require_torch
    def test_load_pretrained_model_for_seq_cls_incorrect_task(self):
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate RobertaModelForSequenceClassification without `num_labels`
        token_cls_finetuner = TokenClassificationFinetuner()
        task = 'multilabel_classification'
        with self.assertRaises(NotImplementedError) as context:
            token_cls_finetuner.load_pretrained_model(
                task=task,
                name_or_path=pretrained_model_name
            )

        self.assertEqual(
            f'The task specified `{task}` is incorrect or not available for TokenClassificationFinetuner',
            str(context.exception))

    #@pytest.mark.skip(reason="skip")
    @require_torch
    def test_load_pretrained_model_for_chunk_level_cls_wangchanberta_spm_camembert_1(self):

        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate RobertaForTokenClassification without `num_labels`
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task='chunk_level_classification',
            name_or_path=pretrained_model_name

        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'RobertaForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'chunk_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
    
    @require_torch
    def test_load_pretrained_model_for_chunk_level_cls_wangchanberta_spm_camembert_2(self):
        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'

        # instantiate RobertaForTokenClassification with `num_labels` specified
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task='chunk_level_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'RobertaForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'chunk_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, 10)

    @require_torch
    def test_load_pretrained_model_for_chunk_level_cls_wangchanberta_spm_camembert_3(self):
        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate RobertaForTokenClassification with `num_labels` specified
        # instantiate RobertaForTokenClassification with `num_labels` specified and using Task enum variable
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task=Task.CHUNK_LEVEL_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'RobertaForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'chunk_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, 10)

    @require_torch
    def test_load_pretrained_model_for_token_level_cls_wangchanberta_spm_camembert_1(self):

        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate RobertaForTokenClassification without `num_labels`
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task='token_level_classification',
            name_or_path=pretrained_model_name

        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'RobertaForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'token_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
    
    @require_torch
    def test_load_pretrained_model_for_token_level_cls_wangchanberta_spm_camembert_2(self):

        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'

        # instantiate RobertaForTokenClassification with `num_labels` specified
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task='token_level_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'RobertaForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'token_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, 10)

    @require_torch
    def test_load_pretrained_model_for_token_level_cls_wangchanberta_spm_camembert_3(self):

        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'

        # instantiate RobertaForTokenClassification with `num_labels` specified and using Task enum variable
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task=Task.TOKEN_LEVEL_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'RobertaForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'token_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, 10)


class TestTokenClassificationFinetunerIntegration(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp/token_cls_finetuner'):
            shutil.rmtree('./tmp/token_cls_finetuner')

    @require_torch
    def test_finetune_wanchanbert_spm_camembert_on_thainer_ner(self):

        # 1. Initiate Token classification finetuner
        
        ner_token_cls_finetuner = TokenClassificationFinetuner()
        ner_token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=CamembertTokenizer,
            name_or_path='airesearch/wangchanberta-base-att-spm-uncased'
        )
        ner_token_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path='airesearch/wangchanberta-base-att-spm-uncased',
            num_labels=5
        )

        # print(f'\n\n[INFO] For Wongnai reviews dataset, perform train-val set splitting (0.9,0.1)')
        dataset = load_dataset('thainer')
        # print(f'\n\n[INFO] Perform dataset splitting')
        # train_val_split = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=2020)
        dataset['train'] = dataset['train'][:100]
        dataset['validation'] = dataset['validation'][:100]
        dataset['test'] =  dataset['test'][:100]

        print(f'\n\n[INFO] Done')
        print(f'# train examples: {len(dataset["train"])}')
        print(f'# val examples: {len(dataset["validation"])}')
        print(f'# test examples: {len(dataset["test"])}')

        # label_encoder = preprocessing.LabelEncoder()
        # label_encoder.fit(get_dict_val(dataset['train'], keys='star_rating'))

        # dataset_preprocessed = { split_name: SequenceClassificationDataset.from_dataset(
        #                 task=Task.MULTICLASS_CLS,
        #                 tokenizer=seq_cls_finetuner.tokenizer,
        #                 dataset=dataset[split_name],
        #                 text_column_name='review_body',
        #                 label_column_name='star_rating',
        #                 max_length=416,
        #                 space_token='<_>',
        #                 preprocessor=partial(_process_transformers, 
        #                     pre_rules = [
        #                     preprocess.fix_html,
        #                     preprocess.rm_brackets,
        #                     preprocess.replace_newlines,
        #                     preprocess.rm_useless_spaces,
        #                     partial(preprocess.replace_spaces, space_token='<_>') if '<_>' != ' ' else lambda x: x,
        #                     preprocess.replace_rep_after],
        #                     lowercase=True
        #                 ),
        #                 label_encoder=label_encoder) for split_name in ['train', 'validation', 'test']
        #             }


        # # define training args
        # output_dir = './tmp/seq_cls_finetuner/wangchanbert-base-att-spm-uncased/wongnai_reviews'
        # training_args = TrainingArguments(output_dir=output_dir)

        # print('training_args', training_args)
        # training_args.max_steps = 5
        # training_args.warmup_steps = 1
        # training_args.no_cuda = True
        # training_args.run_name = None # Set wandb run name to None
        
        # # with test dataset

        # eval_result = seq_cls_finetuner.finetune(training_args, 
        #                            train_dataset=dataset_preprocessed['train'],
        #                            val_dataset=None,
        #                            test_dataset=dataset_preprocessed['test']
        # )

        # self.assertIsNotNone(eval_result)
        # print(eval_result)

        # self.assertTrue(os.path.exists(
        #     os.path.join(training_args.output_dir, 'checkpoint-final', 'pytorch_model.bin')
        # ))