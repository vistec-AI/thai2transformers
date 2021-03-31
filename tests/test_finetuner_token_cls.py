import unittest
import os
import sys
import shutil
from typing import Collection, Iterable, List, Union, Dict, Callable
from functools import partial, lru_cache

import torch
import transformers
import pytest
from datasets import load_dataset, Dataset
from sklearn import preprocessing
from transformers import (
    CamembertTokenizer,
    XLMRobertaTokenizer,
    BertTokenizer,
    HfArgumentParser,
    TrainingArguments
)


from transformers.testing_utils import require_torch
from thai2transformers.finetuner import (
    BaseFinetuner,
    TokenClassificationFinetuner,
)
from thai2transformers.data_collator import (
    DataCollatorForTokenClassification
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

@pytest.fixture()
def skip_sefr(pytestconfig):
    return pytestconfig.getoption("skip_sefr")

class TestTokenClassificationFinetuner(unittest.TestCase):

    
    def test_init(self):
        token_cls_finetuner = TokenClassificationFinetuner()
        self.assertIsNotNone(token_cls_finetuner)

    def setUp(self):
        self.thainer_dataset = load_dataset('thainer')
        label_col = 'ner_tags'
        self.thainer_id2label = {i: name for i, name in
                    enumerate(self.thainer_dataset['train'].features[label_col].feature.names)}
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
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-syllable'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['syllable']
        
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(token_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsSyllableTokenizer')

    @pytest.mark.sefr
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
                name_or_path=pretrained_model_name,
                id2label=self.thainer_id2label
            )

        self.assertEqual(
            f'The task specified `{task}` is incorrect or not available for TokenClassificationFinetuner',
            str(context.exception))

    
    @require_torch
    def test_load_pretrained_model_for_chunk_level_cls_wangchanberta_spm_camembert_1(self):

        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate CamembertForTokenClassification without `num_labels`
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task='chunk_level_classification',
            name_or_path=pretrained_model_name,
            id2label=self.thainer_id2label

        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'CamembertForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'chunk_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
    
    
    @require_torch
    def test_load_pretrained_model_for_chunk_level_cls_wangchanberta_spm_camembert_2(self):
        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'

        # instantiate CamembertForTokenClassification with `num_labels` specified
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task='chunk_level_classification',
            name_or_path=pretrained_model_name,
            id2label=self.thainer_id2label
        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'CamembertForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'chunk_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, len(self.thainer_id2label.keys()))

    
    @require_torch
    def test_load_pretrained_model_for_chunk_level_cls_wangchanberta_spm_camembert_3(self):
        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate CamembertForTokenClassification with `num_labels` specified
        # instantiate CamembertForTokenClassification with `num_labels` specified and using Task enum variable
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task=Task.CHUNK_LEVEL_CLS,
            name_or_path=pretrained_model_name,
            id2label=self.thainer_id2label
        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'CamembertForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'chunk_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, len(self.thainer_id2label.keys()))

    
    @require_torch
    def test_load_pretrained_model_for_token_level_cls_wangchanberta_spm_camembert_1(self):

        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate CamembertForTokenClassification without `num_labels`
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task='token_level_classification',
            name_or_path=pretrained_model_name,
            id2label=self.thainer_id2label
        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'CamembertForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'token_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, len(self.thainer_id2label.keys()))
    
    
    @require_torch
    def test_load_pretrained_model_for_token_level_cls_wangchanberta_spm_camembert_2(self):

        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'

        # instantiate CamembertForTokenClassification with `num_labels` specified
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task='token_level_classification',
            name_or_path=pretrained_model_name,
            id2label=self.thainer_id2label
        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'CamembertForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'token_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, len(self.thainer_id2label.keys()))

    
    @require_torch
    def test_load_pretrained_model_for_token_level_cls_wangchanberta_spm_camembert_3(self):

        os.environ['WANDB_DISABLED'] = 'true'
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'

        # instantiate CamembertForTokenClassification with `num_labels` specified and using Task enum variable
        token_cls_finetuner = TokenClassificationFinetuner()
        token_cls_finetuner.load_pretrained_model(
            task=Task.TOKEN_LEVEL_CLS,
            name_or_path=pretrained_model_name,
            id2label=self.thainer_id2label
        )
        self.assertEqual(token_cls_finetuner.model.__class__.__name__,'CamembertForTokenClassification')
        self.assertEqual(token_cls_finetuner.metric.__name__, 'token_level_classification_metrics')
        self.assertEqual(token_cls_finetuner.config.num_labels, len(self.thainer_id2label.keys()))


class TestTokenClassificationFinetunerIntegration:

    def setUp(self):
        if os.path.exists('./tmp/token_cls_finetuner'):
            shutil.rmtree('./tmp/token_cls_finetuner')

    
    @staticmethod
    def _token_cls_preprocess(examples, tokenizer, text_col, label_col, space_token='<_>', lowercase=True):
        def pre_tokenize(token: str, space_token='<_>'):
            token = token.replace(' ', space_token)
            return token

        @lru_cache(maxsize=None)
        def cached_tokenize(token: str, space_token='<_>', lowercase=True):
            if lowercase:
                token = token.lower()
            token = pre_tokenize(token, space_token)
            ids = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(token))
            return ids

        tokens = []
        labels = []
        old_positions = []
        for example_tokens, example_labels in zip(examples[text_col], examples[label_col]):
            new_example_tokens = []
            new_example_labels = []
            old_position = []
            for i, (token, label) in enumerate(zip(example_tokens, example_labels)):
                # tokenize each already pretokenized tokens with our own tokenizer.
                toks = cached_tokenize(token, space_token=space_token, lowercase=True)
                n_toks = len(toks)
                new_example_tokens.extend(toks)
                # expand label to cover all tokens that get split in a pretokenized token
                new_example_labels.extend([label] * n_toks)
                # kept track of old position
                old_position.extend([i] * n_toks)
            tokens.append(new_example_tokens)
            labels.append(new_example_labels)
            old_positions.append(old_position)
        tokenized_inputs = tokenizer._batch_prepare_for_model(
            [(e, None) for e in tokens],
            truncation_strategy=transformers.tokenization_utils_base.TruncationStrategy.LONGEST_FIRST,
            add_special_tokens=True, max_length=416)
        # in case of needed truncation we need to chop off some of the labels manually
        max_length = max(len(e) for e in tokenized_inputs['input_ids'])
        # add -100 to first and last token which is special tokens for <s> and </s>
        # -100 is a convention for padding in higgingface transformer lib
        # and calculating loss should skip this
        tokenized_inputs['old_positions'] = [[-100] + e[:max_length - 2] + [-100]
                                            for e in old_positions]
        tokenized_inputs['labels'] = [[-100] + e[:max_length - 2] + [-100]
                                    for e in labels]
        return tokenized_inputs
    

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
    def test_finetune_wanchanbert_spm_camembert_on_thainer_ner(self,
            model_name_or_path,
            tokenizer_name_or_path,
            tokenizer_cls,
            skip_sefr):

        if skip_sefr:
            pytest.skip('Skip tests requiring SEFR tokenizer')

        # 1. Dowload dataset
        dataset = load_dataset('thainer')
        text_col = 'tokens'
        label_col = 'ner_tags'

        # Remove tag: `ไม่ยืนยัน`
        dataset['train'] = dataset['train'].map(
            lambda examples: {'ner_tags': [i if i not in [13, 26] else 27
                                            for i in examples[label_col]]}
        )

        id2label = {i: name for i, name in
                    enumerate(dataset['train'].features[label_col].feature.names)}
        labels = dataset['train'].features[label_col].feature.names
        num_labels = dataset['train'].features[label_col].feature.num_classes
        
        train_val_split = dataset['train'].train_test_split(test_size=0.5, shuffle=False)
        dataset['train'] = train_val_split['train'][:100]
        dataset['validation'] = train_val_split['test'][100:200]
        dataset['test'] = train_val_split['test'][:100]

        print(f'\n\n[INFO] Done')
        print(f'# train examples: {len(dataset["train"])}')
        print(f'# val examples: {len(dataset["validation"])}')
        print(f'# test examples: {len(dataset["test"])}')

        # 1. Initiate Token classification finetuner
        
        ner_token_cls_finetuner = TokenClassificationFinetuner()
        assert ner_token_cls_finetuner  != None

        ner_token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=model_name_or_path
        )
        assert ner_token_cls_finetuner.tokenizer != None
        assert ner_token_cls_finetuner.tokenizer.__class__.__name__ == tokenizer_cls.__name__

        ner_token_cls_finetuner.load_pretrained_model(
            task='chunk_level_classification',
            name_or_path='airesearch/wangchanberta-base-att-spm-uncased',
            num_labels=num_labels,
            id2label=id2label
        )
        assert ner_token_cls_finetuner.tokenizer != None
        assert ('ForTokenClassification' in ner_token_cls_finetuner.model.__class__.__name__) == True
        assert ner_token_cls_finetuner.num_labels == num_labels
        assert ner_token_cls_finetuner.model.num_labels == num_labels
       
        data_collator = DataCollatorForTokenClassification(
                            tokenizer=ner_token_cls_finetuner.tokenizer
                        )
        assert data_collator != None

        train_dataset = Dataset.from_dict(TestTokenClassificationFinetunerIntegration._token_cls_preprocess(dataset['train'],
                            tokenizer=ner_token_cls_finetuner.tokenizer,
                            text_col=text_col,
                            label_col=label_col))
        val_dataset = Dataset.from_dict(TestTokenClassificationFinetunerIntegration._token_cls_preprocess(dataset['validation'],
                            tokenizer=ner_token_cls_finetuner.tokenizer,
                            text_col=text_col,
                            label_col=label_col))
        test_dataset = Dataset.from_dict(TestTokenClassificationFinetunerIntegration._token_cls_preprocess(dataset['test'],
                            tokenizer=ner_token_cls_finetuner.tokenizer,
                            text_col=text_col,
                            label_col=label_col))
        
        val_dataset = Dataset.from_dict(data_collator(val_dataset))
        test_dataset = Dataset.from_dict(data_collator(test_dataset))
        # define training args
        output_dir = './tmp/seq_cls_finetuner/wangchanbert-base-att-spm-uncased/wongnai_reviews'
        training_args = TrainingArguments(output_dir=output_dir,
                            max_steps = 10,
                            warmup_steps = 2,
                            evaluation_strategy = 'steps',
                            eval_steps = 10,
                            logging_steps = 1,
                            no_cuda = not torch.cuda.is_available(),
                            run_name = None ,
                        )

        print('training_args', training_args)
        # Set wandb run name to None
        

        eval_result = ner_token_cls_finetuner.finetune(training_args, 
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   test_dataset=test_dataset
        )

        assert eval_result != None
        print(eval_result)

        assert os.path.exists(os.path.join(training_args.output_dir, 'checkpoint-final', 'pytorch_model.bin')) == True
    
        ner_token_cls_finetuner.finetune(training_args, 
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   test_dataset=None
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
    def test_finetune_wanchanbert_spm_camembert_on_thainer_pos(self,
            model_name_or_path,
            tokenizer_name_or_path,
            tokenizer_cls,
            skip_sefr
        ):
        
        if skip_sefr:
            pytest.skip('Skip tests requiring SEFR tokenizer')

        # 1. Dowload dataset
        dataset = load_dataset('thainer')
        text_col = 'tokens'
        label_col = 'pos_tags'
        # print(f'\n\n[INFO] Perform dataset splitting')
        # train_val_split = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=2020)

        id2label = {i: name for i, name in
                    enumerate(dataset['train'].features[label_col].feature.names)}
        labels = dataset['train'].features[label_col].feature.names
        num_labels = dataset['train'].features[label_col].feature.num_classes
        
        train_val_split = dataset['train'].train_test_split(test_size=0.5, shuffle=False)
        dataset['train'] = train_val_split['train'][:100]
        dataset['validation'] = train_val_split['test'][100:200]
        dataset['test'] = train_val_split['test'][:100]

        print(f'\n\n[INFO] Done')
        print(f'# train examples: {len(dataset["train"])}')
        print(f'# val examples: {len(dataset["validation"])}')
        print(f'# test examples: {len(dataset["test"])}')

        # 1. Initiate Token classification finetuner
        
        ner_token_cls_finetuner = TokenClassificationFinetuner()
        assert ner_token_cls_finetuner != None

        ner_token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path='airesearch/wangchanberta-base-att-spm-uncased'
        )
        assert ner_token_cls_finetuner.tokenizer != None
        assert ner_token_cls_finetuner.tokenizer.__class__.__name__ == tokenizer_cls.__name__

        ner_token_cls_finetuner.load_pretrained_model(
            task='token_level_classification',
            name_or_path='airesearch/wangchanberta-base-att-spm-uncased',
            num_labels=num_labels,
            id2label=id2label
        )
        assert ner_token_cls_finetuner.tokenizer != None
        assert ('ForTokenClassification' in ner_token_cls_finetuner.model.__class__.__name__) == True
        assert ner_token_cls_finetuner.num_labels == num_labels
        assert ner_token_cls_finetuner.model.num_labels == num_labels
       
        data_collator = DataCollatorForTokenClassification(
                            tokenizer=ner_token_cls_finetuner.tokenizer
                        )
        assert data_collator != None

        train_dataset = Dataset.from_dict(TestTokenClassificationFinetunerIntegration._token_cls_preprocess(dataset['train'],
                            tokenizer=ner_token_cls_finetuner.tokenizer,
                            text_col=text_col,
                            label_col=label_col))
        val_dataset = Dataset.from_dict(TestTokenClassificationFinetunerIntegration._token_cls_preprocess(dataset['validation'],
                            tokenizer=ner_token_cls_finetuner.tokenizer,
                            text_col=text_col,
                            label_col=label_col))
        test_dataset = Dataset.from_dict(TestTokenClassificationFinetunerIntegration._token_cls_preprocess(dataset['test'],
                            tokenizer=ner_token_cls_finetuner.tokenizer,
                            text_col=text_col,
                            label_col=label_col))
        
        val_dataset = Dataset.from_dict(data_collator(val_dataset))
        test_dataset = Dataset.from_dict(data_collator(test_dataset))
        # define training args
        output_dir = './tmp/seq_cls_finetuner/wangchanbert-base-att-spm-uncased/wongnai_reviews'
        training_args = TrainingArguments(output_dir=output_dir,
                            max_steps = 50,
                            warmup_steps = 8,
                            evaluation_strategy = 'steps',
                            eval_steps = 25,
                            logging_steps = 1,
                            no_cuda = not torch.cuda.is_available(),
                            run_name = None # Set wandb run name to None
                        )

        print('training_args', training_args)


        eval_result = ner_token_cls_finetuner.finetune(training_args, 
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   test_dataset=test_dataset
        )

        assert eval_result != None
        print(eval_result)

        assert os.path.exists(os.path.join(training_args.output_dir, 'checkpoint-final', 'pytorch_model.bin')) == True

        ner_token_cls_finetuner.finetune(training_args, 
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   test_dataset=None
        )