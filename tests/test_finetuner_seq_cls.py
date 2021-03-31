import unittest
import pytest
import os
import sys
import shutil
from typing import Collection, Iterable, List, Union, Dict, Callable
from functools import partial
from collections import defaultdict
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
    FakeSefrCutTokenizer,
    sefr_cut_tokenize,
    SEFR_SPLIT_TOKEN,
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
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-syllable'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['syllable']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsSyllableTokenizer')

    #@pytest.mark.skip(reason="change api")
    @pytest.mark.sefr
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
        
        # instantiate CamembertForSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name

        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'CamembertForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate CamembertForSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'CamembertForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate CamembertForSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTICLASS_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'CamembertForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate CamembertForMultiLabelSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'CamembertForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate CamembertForMultiLabelSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'CamembertForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)


        # instantiate CamembertForMultiLabelSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTILABEL_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'CamembertForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

    @require_torch
    def test_load_pretrained_model_for_seq_cls_mbert(self):
        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/bert-base-multilingual-cased-finetuned'
        
        # instantiate BertForSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name,
            revision='finetuned@wongnai_reviews',

        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'BertForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        # num_labels = 5 is the default value for wongnai_reviews
        self.assertEqual(seq_cls_finetuner.config.num_labels, 5) 
        
        with self.assertRaises(RuntimeError) as context:
            # expect RuntimeError due to num_labels mismatch
            # instantiate BertForSequenceClassification with `num_labels` specified
            seq_cls_finetuner = SequenceClassificationFinetuner()
            seq_cls_finetuner.load_pretrained_model(
                task='multiclass_classification',
                name_or_path=pretrained_model_name,
                revision='finetuned@wongnai_reviews',
                num_labels=10
            )
            print('context.exception', context.exception)

        self.assertEqual(type(RuntimeError()), type(context.exception))

        # instantiate BertForMultiLabelSequenceClassificationultiLabelSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name,
            revision='finetuned@prachathai67k',
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'BertForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        # num_labels = 12 is the default value for prachathai67k
        self.assertEqual(seq_cls_finetuner.config.num_labels, 12)
        
        with self.assertRaises(RuntimeError) as context:
            # expect RuntimeError due to num_labels mismatch
            # instantiate BertForMultiLabelSequenceClassification with `num_labels` specified
            seq_cls_finetuner = SequenceClassificationFinetuner()
            seq_cls_finetuner.load_pretrained_model(
                task='multilabel_classification',
                name_or_path=pretrained_model_name,
                revision='finetuned@prachathai67k',
                num_labels=10
            )
        self.assertEqual(type(RuntimeError()), type(context.exception))


    @require_torch
    def test_load_pretrained_model_for_seq_cls_xlmr(self):
        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/xlm-roberta-base-finetuned'
        
        # instantiate XLMRobertaForSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name,
            revision='finetuned@wongnai_reviews',

        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'XLMRobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        # num_labels = 5 is the default value for wongnai_reviews
        self.assertEqual(seq_cls_finetuner.config.num_labels, 5)
        
        with self.assertRaises(RuntimeError) as context:
            # expect RuntimeError due to num_labels mismatch
            # instantiate XLMRobertaForSequenceClassification with `num_labels` specified
            seq_cls_finetuner = SequenceClassificationFinetuner()
            seq_cls_finetuner.load_pretrained_model(
                task='multiclass_classification',
                name_or_path=pretrained_model_name,
                revision='finetuned@wongnai_reviews',
                num_labels=10
            )
        self.assertEqual(type(RuntimeError()), type(context.exception))

        # instantiate XLMRobertaForMultiLabelSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name,
            revision='finetuned@prachathai67k',
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'XLMRobertaForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        # num_labels = 2 is the default value for prachathai67k
        self.assertEqual(seq_cls_finetuner.config.num_labels, 12) 
        
        with self.assertRaises(RuntimeError) as context:
            # expect RuntimeError due to num_labels mismatch
            # instantiate XLMRobertaForMultiLabelSequenceClassification with `num_labels` specified
            seq_cls_finetuner = SequenceClassificationFinetuner()
            seq_cls_finetuner.load_pretrained_model(
                task='multilabel_classification',
                name_or_path=pretrained_model_name,
                revision='finetuned@prachathai67k',
                num_labels=10
            )
        self.assertEqual(type(RuntimeError()), type(context.exception))

@pytest.fixture()
def skip_sefr(pytestconfig):
    return pytestconfig.getoption("skip_sefr")

class TestSequenceClassificationFinetunerIntegration:

    def setUp(self):
        if os.path.exists('./tmp/seq_cls_finetuner'):
            shutil.rmtree('./tmp/seq_cls_finetuner')

    @pytest.mark.parametrize("model_name_or_path,tokenizer_name_or_path,tokenizer_cls,skip_sefr", [
        ('airesearch/wangchanberta-base-att-spm-uncased', 'airesearch/wangchanberta-base-att-spm-uncased', CamembertTokenizer, False),
        ('airesearch/wangchanberta-base-wiki-spm', 'airesearch/wangchanberta-base-wiki-spm', ThaiRobertaTokenizer, False),
        ('airesearch/wangchanberta-base-wiki-newmm', 'airesearch/wangchanberta-base-wiki-newmm', ThaiWordsNewmmTokenizer, False),
        ('airesearch/wangchanberta-base-wiki-syllable', 'airesearch/wangchanberta-base-wiki-syllable', ThaiWordsSyllableTokenizer, False),
        ('airesearch/wangchanberta-base-wiki-sefr', 'airesearch/wangchanberta-base-wiki-sefr', FakeSefrCutTokenizer, skip_sefr),
        ('bert-base-multilingual-cased', 'bert-base-multilingual-cased', BertTokenizer, False),
        ('xlm-roberta-base', 'xlm-roberta-base', XLMRobertaTokenizer, False),
    ])
    @require_torch
    def test_finetune_models_on_wongnai(self, model_name_or_path, tokenizer_name_or_path, tokenizer_cls, skip_sefr):

        if skip_sefr:
            pytest.skip('Skip tests requiring SEFR tokenizer')
        os.environ['WANDB_DISABLED'] = 'true'
        # 1. Initiate Sequence classification finetuner
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=tokenizer_name_or_path,
        )
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=model_name_or_path,
            num_labels=5
        )
        assert seq_cls_finetuner.tokenizer != None
        assert seq_cls_finetuner.tokenizer.__class__.__name__ == tokenizer_cls.__name__
        assert ('ForSequenceClassification' in seq_cls_finetuner.model.__class__.__name__) == True

        print(f'\n\n[INFO] For Wongnai reviews dataset, perform train-val set splitting (0.9,0.1)')
        dataset = defaultdict()
        dataset['train'] = load_dataset('wongnai_reviews', split='train[:100]')
        dataset['validation'] = load_dataset('wongnai_reviews', split='train[100:200]')
        dataset['test'] = load_dataset('wongnai_reviews', split='train[200:300]')
        

        print(f'\n\n[INFO] Done')
        print(f'# train examples: {len(dataset["train"])}')
        print(f'# val examples: {len(dataset["validation"])}')
        print(f'# test examples: {len(dataset["test"])}')
        
        text_column_name = 'review_body'
        label_encoder = preprocessing.LabelEncoder()
        dataset['train_full'] = load_dataset('wongnai_reviews', split='train')
        label_encoder.fit(get_dict_val(dataset['train_full'], keys='star_rating'))

        if tokenizer_cls.__name__ == 'FakeSefrCutTokenizer':
            
            space_token = '<_>'        
            def tokenize_fn(batch):
                results = []
                for tokens in sefr_cut_tokenize(get_dict_val(batch, text_column_name), n_jobs=1):
                    results.append(SEFR_SPLIT_TOKEN.join([ SEFR_SPLIT_TOKEN.join([token] + [space_token]) for token in tokens ] ))
                return results

            for split_name in dataset.keys():
                if split_name == 'train_full':
                    continue
                dataset[split_name] = dataset[split_name].map(lambda batch: {
                                                 text_column_name: tokenize_fn(batch) 
                                            }, batched=True, batch_size=1)

                print(f'[DEBUG] examples from {split_name} , {dataset[split_name][text_column_name][:3]}')

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
        if '/' in model_name_or_path:
            output_dir = f'./tmp/seq_cls_finetuner/{model_name_or_path.split("/")[-1]}/wongnai_reviews'
        else:
            output_dir = f'./tmp/seq_cls_finetuner/{model_name_or_path}/wongnai_reviews'
        training_args = TrainingArguments(output_dir=output_dir,
                            max_steps = 5,
                            warmup_steps = 1,
                            no_cuda = True,
                            run_name = None)


        print('training_args', training_args)
        
        # with test dataset
        eval_result = seq_cls_finetuner.finetune(training_args, 
                                   train_dataset=dataset_preprocessed['train'],
                                   val_dataset=None,
                                   test_dataset=dataset_preprocessed['test']
        )

        assert eval_result != None
        print(eval_result)

        assert os.path.exists(
            os.path.join(training_args.output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ) == True

        # without test dataset
        eval_result = seq_cls_finetuner.finetune(training_args, 
                                   train_dataset=dataset_preprocessed['train'],
                                   val_dataset=None,
                                   test_dataset=None,
        )
    
    @pytest.mark.parametrize("model_name_or_path,tokenizer_name_or_path,tokenizer_cls,skip_sefr", [
        ('airesearch/wangchanberta-base-att-spm-uncased', 'airesearch/wangchanberta-base-att-spm-uncased', CamembertTokenizer, False),
        ('airesearch/wangchanberta-base-wiki-spm', 'airesearch/wangchanberta-base-wiki-spm', ThaiRobertaTokenizer, False),
        ('airesearch/wangchanberta-base-wiki-newmm', 'airesearch/wangchanberta-base-wiki-newmm', ThaiWordsNewmmTokenizer, False),
        ('airesearch/wangchanberta-base-wiki-syllable', 'airesearch/wangchanberta-base-wiki-syllable', ThaiWordsSyllableTokenizer, False),
        ('airesearch/wangchanberta-base-wiki-sefr', 'airesearch/wangchanberta-base-wiki-sefr', FakeSefrCutTokenizer, skip_sefr),
        ('bert-base-multilingual-cased', 'bert-base-multilingual-cased', BertTokenizer, False),
        ('xlm-roberta-base', 'xlm-roberta-base', XLMRobertaTokenizer, False),
    ])
    @require_torch
    def test_finetune_models_on_prachathai67k(self, model_name_or_path, tokenizer_name_or_path, tokenizer_cls, skip_sefr):
        if skip_sefr:
            pytest.skip('Skip tests requiring SEFR tokenizer')
        
        os.environ['WANDB_DISABLED'] = 'true'
        # 1. Initiate Sequence classification finetuner
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=tokenizer_name_or_path,
        )
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=model_name_or_path,
            num_labels=5
        )
        assert seq_cls_finetuner.tokenizer != None
        assert seq_cls_finetuner.tokenizer.__class__.__name__ == tokenizer_cls.__name__
        assert ('ForMultiLabelSequenceClassification' in seq_cls_finetuner.model.__class__.__name__) == True

        prachathai_dataset_name = 'prachathai67k'
        prachathai_text_col_name = 'body_text'
        prachathai_label_col_name =  ['politics', 'human_rights', 'quality_of_life',
                                      'international', 'social', 'environment',
                                      'economics', 'culture', 'labor',
                                      'national_security', 'ict', 'education']

        dataset = load_dataset(prachathai_dataset_name)
        print(f'\n\n[INFO] Perform dataset splitting')
        dataset = defaultdict()
        dataset['train'] = load_dataset(prachathai_dataset_name, split='train[:100]')
        dataset['validation'] = load_dataset(prachathai_dataset_name, split='train[100:200]')
        dataset['test'] = load_dataset(prachathai_dataset_name, split='train[200:300]')
        

        print(f'\n\n[INFO] Done')
        print(f'# train examples: {len(dataset["train"])}')
        print(f'# val examples: {len(dataset["validation"])}')
        print(f'# test examples: {len(dataset["test"])}')
        
        label_encoder = preprocessing.LabelEncoder()
        dataset['train_full'] = load_dataset(prachathai_dataset_name, split='train')
        label_encoder.fit(get_dict_val(dataset['train_full'], keys='star_rating'))

        if tokenizer_cls.__name__ == 'FakeSefrCutTokenizer':
            
            space_token = '<_>'        
            def tokenize_fn(batch):
                results = []
                for tokens in sefr_cut_tokenize(get_dict_val(batch, prachathai_dataset_name), n_jobs=1):
                    results.append(SEFR_SPLIT_TOKEN.join([ SEFR_SPLIT_TOKEN.join([token] + [space_token]) for token in tokens ] ))
                return results

            for split_name in dataset.keys():
                if split_name == 'train_full':
                    continue
                dataset[split_name] = dataset[split_name].map(lambda batch: {
                                                 prachathai_dataset_name: tokenize_fn(batch) 
                                            }, batched=True, batch_size=1)

                print(f'[DEBUG] examples from {split_name} , {dataset[split_name][prachathai_dataset_name][:3]}')

        dataset_preprocessed = { split_name: SequenceClassificationDataset.from_dataset(
                        task=Task.MULTILABEL_CLS,
                        tokenizer=seq_cls_finetuner.tokenizer,
                        dataset=dataset[split_name],
                        text_column_name=prachathai_text_col_name,
                        label_column_name=prachathai_label_col_name,
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
                        label_encoder=None) for split_name in ['train', 'validation', 'test']
                    }


        # define training args
        if '/' in model_name_or_path:
            output_dir = f'./tmp/seq_cls_finetuner/{model_name_or_path.split("/")[-1]}/prachathai67k'
        else:
            output_dir = f'./tmp/seq_cls_finetuner/{model_name_or_path}/prachathai67k'
        training_args = TrainingArguments(output_dir=output_dir,
                            max_steps = 5,
                            warmup_steps = 1,
                            no_cuda = True,
                            run_name = None)

        print('training_args', training_args)

        # with test dataset
        eval_result = seq_cls_finetuner.finetune(training_args, 
                                   train_dataset=dataset_preprocessed['train'],
                                   val_dataset=dataset_preprocessed['val'],
                                   test_dataset=dataset_preprocessed['test']
        )

        assert eval_result != None
        print(eval_result)

        assert os.path.exists(
            os.path.join(training_args.output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ) == True

        # without test dataset
        eval_result = seq_cls_finetuner.finetune(training_args, 
                                   train_dataset=dataset_preprocessed['train'],
                                   val_dataset=dataset_preprocessed['val'],
                                   test_dataset=None,
        )