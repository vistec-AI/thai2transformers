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
    SequenceClassificationFinetuner,
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


class TestSequenceClassificationFinetuner(unittest.TestCase):

    def test_init(self):
        seq_cls_finetuner = SequenceClassificationFinetuner()
        self.assertIsNotNone(seq_cls_finetuner)
    
    def test_base_finetuner(self):
        base_finetuner = BaseFinetuner()
        self.assertIsNotNone(base_finetuner)
        base_finetuner.load_pretrained_tokenizer()
        base_finetuner.load_pretrained_model()
        base_finetuner.finetune()
    
    #@pytest.mark.skip(reason="change api")
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_spm_camembert(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['spm_camembert']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'CamembertTokenizer')
        self.assertEqual(seq_cls_finetuner.tokenizer.additional_special_tokens,
                         ['<s>NOTUSED', '</s>NOTUSED', '<_>'])

    #@pytest.mark.skip(reason="change api")
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_spm(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-spm'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['spm']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiRobertaTokenizer')
    
    #@pytest.mark.skip(reason="change api")
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_newmm(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-newmm'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['newmm']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsNewmmTokenizer')
    
    #@pytest.mark.skip(reason="change api")
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_syllable(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-ssg'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['syllable']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsSyllableTokenizer')

    #@pytest.mark.skip(reason="change api")
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_sefr(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-sefr'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['sefr_cut']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'FakeSefrCutTokenizer')
    
    #@pytest.mark.skip(reason="change api")
    @require_torch
    def test_load_pretrained_tokenizer_xlmr(self):
        
        pretrained_tokenizer_name = 'xlm-roberta-base'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['xlmr']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'XLMRobertaTokenizer')
    
    #@pytest.mark.skip(reason="change api")
    @require_torch
    def test_load_pretrained_tokenizer_mbert(self):
        
        pretrained_tokenizer_name = 'bert-base-multilingual-cased'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['mbert']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'BertTokenizer')
    
    @require_torch
    def test_load_pretrained_model_for_seq_cls_incorrect_task(self):
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate RobertaModelForSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        task = 'token_classification'
        with self.assertRaises(NotImplementedError) as context:
            seq_cls_finetuner.load_pretrained_model(
                task=task,
                name_or_path=pretrained_model_name
            )

        self.assertEqual(
            f'The task specified `{task}` is incorrect or not available for SequenceClassificationFinetuner',
            str(context.exception))

    @require_torch
    def test_load_pretrained_model_for_seq_cls_wangchanberta_spm_camembert(self):
        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate RobertaForSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name

        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate RobertaForSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate RobertaForSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTICLASS_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate RobertaForMultiLabelSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate RobertaForMultiLabelSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)


        # instantiate RobertaForMultiLabelSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTILABEL_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

    @require_torch
    def test_load_pretrained_model_for_seq_cls_mbert_spm_camembert(self):
        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/bert-base-multilingual-cased-finetuned'
        
        # instantiate BertForSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name

        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'BertForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate BertForSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'BertForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate BertForMultilabelSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTICLASS_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'BertForMultilabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate RobertaForMBertForMultilabelSequenceClassificationultiLabelSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'BertForMultilabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate BertForMultilabelSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'BertForMultilabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)


        # instantiate BertForMultilabelSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTILABEL_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'BertForMultilabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

    @require_torch
    def test_load_pretrained_model_for_seq_cls_xlmr_spm_camembert(self):
        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/xlm-roberta-base-finetuned'
        
        # instantiate XLMRobertaForSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name

        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'XLMRobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate XLMRobertaForSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'XLMRobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate XLMRobertaForSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTICLASS_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'XLMRobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate XLMRobertaForMultilabelSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'XLMRobertaForMultilabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate XLMRobertaForMultilabelSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'XLMRobertaForMultilabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)


        # instantiate XLMRobertaForMultilabelSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTILABEL_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'XLMRobertaForMultilabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)


class TestSequenceClassificationFinetunerIntegration(unittest.TestCase):

    def setUp(self):
        if os.path.exists('./tmp/seq_cls_finetuner'):
            shutil.rmtree('./tmp/seq_cls_finetuner')
    
    list_of_pretrained_model_name_or_paths = [
        'airesearch/wangchanberta-base-att-spm-uncased',
        'airesearch/wangchanberta-base-wiki-spm',
        'airesearch/wangchanberta-base-wiki-newmm.',
        'airesearch/wangchanberta-base-wiki-ssg',
        'airesearch/wangchanberta-base-wiki-sefr',
        'airesearch/bert-base-multilingual-cased-finetuned',
        'airesearch/xlm-roberta-base-finetuned',
    ]
    
    @require_torch
    @pytest.mark.parametrize("pretrained_model_name_or_path", list_of_pretrained_model_name_or_paths)
    def test_finetune_models_on_wongnai(self, pretrained_model_name_or_path):

        # 1. Initiate Sequence classification finetuner
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=CamembertTokenizer,
            name_or_path=pretrained_model_name_or_path
        )
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name_or_path,
            num_labels=5
        )

        print(f'\n\n[INFO] For Wongnai reviews dataset, perform train-val set splitting (0.9,0.1)')
        dataset = load_dataset('wongnai_reviews')
        print(f'\n\n[INFO] Perform dataset splitting')
        train_val_split = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=2020)
        dataset['train'] = train_val_split['train'][:100]
        dataset['validation'] = train_val_split['test'][:100]
        dataset['test'] = train_val_split['test'][:100]
        print(f'\n\n[INFO] Done')
        print(f'# train examples: {len(dataset["train"])}')
        print(f'# val examples: {len(dataset["validation"])}')
        print(f'# test examples: {len(dataset["test"])}')

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(get_dict_val(dataset['train'], keys='star_rating'))

        dataset_preprocessed = { split_name: SequenceClassificationDataset.from_dataset(
                        task=Task.MULTICLASS_CLS,
                        tokenizer=seq_cls_finetuner.tokenizer,
                        dataset=dataset[split_name],
                        text_column_name='review_body',
                        label_column_name='star_rating',
                        max_length=416,
                        space_token='<_>',
                        preprocessor=partial(_process_transformers, 
                            pre_rules = [
                            preprocess.fix_html,
                            preprocess.rm_brackets,
                            preprocess.replace_newlines,
                            preprocess.rm_useless_spaces,
                            partial(preprocess.replace_spaces, space_token='<_>') if '<_>' != ' ' else lambda x: x,
                            preprocess.replace_rep_after],
                            lowercase=True
                        ),
                        label_encoder=label_encoder) for split_name in ['train', 'validation', 'test']
                    }


        # define training args
        output_dir = f'./tmp/seq_cls_finetuner/{pretrained_model_name_or_path.spilt('/')[-1]}/wongnai_reviews'
        training_args = TrainingArguments(output_dir=output_dir)

        print('training_args', training_args)
        training_args.max_steps = 5
        training_args.warmup_steps = 1
        training_args.no_cuda = True
        training_args.run_name = None # Set wandb run name to None
        
        # with test dataset

        eval_result = seq_cls_finetuner.finetune(training_args, 
                                   train_dataset=dataset_preprocessed['train'],
                                   val_dataset=None,
                                   test_dataset=dataset_preprocessed['test']
        )

        self.assertIsNotNone(eval_result)
        print(eval_result)

        self.assertTrue(os.path.exists(
            os.path.join(training_args.output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ))