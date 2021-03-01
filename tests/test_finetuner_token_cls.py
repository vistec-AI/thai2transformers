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


class TestTokenClassificationFinetuner(unittest.TestCase):

    @pytest.mark.skip(reason="done")
    def test_init(self):
        token_cls_finetuner = TokenClassificationFinetuner()
        self.assertIsNotNone(token_cls_finetuner)

    @pytest.mark.skip(reason="done")
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

    @pytest.mark.skip(reason="done")
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
    
    @pytest.mark.skip(reason="done")
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
    
    @pytest.mark.skip(reason="done")
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

    @pytest.mark.skip(reason="done")
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
    
    @pytest.mark.skip(reason="done")
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
    
    @pytest.mark.skip(reason="done")
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
    
    @pytest.mark.skip(reason="done")
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

    @pytest.mark.skip(reason="done")
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
    
    @pytest.mark.skip(reason="done")
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

    @pytest.mark.skip(reason="done")
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

    @pytest.mark.skip(reason="done")
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
    
    @pytest.mark.skip(reason="done")
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

    @pytest.mark.skip(reason="done")
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
        
    @require_torch
    def test_finetune_wanchanbert_spm_camembert_on_thainer_ner(self):

        # 1. Dowload dataset
        dataset = load_dataset('thainer')
        text_col = 'tokens'
        label_col = 'ner_tags'
        # print(f'\n\n[INFO] Perform dataset splitting')
        # train_val_split = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=2020)

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
        self.assertIsNotNone(ner_token_cls_finetuner)

        ner_token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=CamembertTokenizer,
            name_or_path='airesearch/wangchanberta-base-att-spm-uncased'
        )
        self.assertIsNotNone(ner_token_cls_finetuner.tokenizer)
        self.assertEqual(ner_token_cls_finetuner.tokenizer.__class__.__name__, 'CamembertTokenizer')

        ner_token_cls_finetuner.load_pretrained_model(
            task='chunk_level_classification',
            name_or_path='airesearch/wangchanberta-base-att-spm-uncased',
            num_labels=num_labels,
            id2label=id2label
        )
        self.assertIsNotNone(ner_token_cls_finetuner.tokenizer)
        self.assertEqual(ner_token_cls_finetuner.model.__class__.__name__, 'RobertaForTokenClassification')
        self.assertEqual(ner_token_cls_finetuner.num_labels, num_labels)
        self.assertEqual(ner_token_cls_finetuner.model.num_labels, num_labels)
       
        data_collator = DataCollatorForTokenClassification(
                            tokenizer=ner_token_cls_finetuner.tokenizer
                        )
        self.assertIsNotNone(data_collator)

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
        training_args = TrainingArguments(output_dir=output_dir)

        print('training_args', training_args)
        training_args.max_steps = 10
        training_args.warmup_steps = 2
        training_args.evaluation_strategy = 'steps'
        training_args.eval_steps = 10
        training_args.logging_steps = 
        training_args.no_cuda = not torch.cuda.is_available()
        training_args.run_name = None # Set wandb run name to None
        

        eval_result = ner_token_cls_finetuner.finetune(training_args, 
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   test_dataset=test_dataset
        )

        self.assertIsNotNone(eval_result)
        print(eval_result)

        self.assertTrue(os.path.exists(
            os.path.join(training_args.output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ))
    
    @require_torch
    def test_finetune_wanchanbert_spm_camembert_on_thainer_pos(self):

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
        self.assertIsNotNone(ner_token_cls_finetuner)

        ner_token_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=CamembertTokenizer,
            name_or_path='airesearch/wangchanberta-base-att-spm-uncased'
        )
        self.assertIsNotNone(ner_token_cls_finetuner.tokenizer)
        self.assertEqual(ner_token_cls_finetuner.tokenizer.__class__.__name__, 'CamembertTokenizer')

        ner_token_cls_finetuner.load_pretrained_model(
            task='token_level_classification',
            name_or_path='airesearch/wangchanberta-base-att-spm-uncased',
            num_labels=num_labels,
            id2label=id2label
        )
        self.assertIsNotNone(ner_token_cls_finetuner.tokenizer)
        self.assertEqual(ner_token_cls_finetuner.model.__class__.__name__, 'RobertaForTokenClassification')
        self.assertEqual(ner_token_cls_finetuner.num_labels, num_labels)
        self.assertEqual(ner_token_cls_finetuner.model.num_labels, num_labels)
       
        data_collator = DataCollatorForTokenClassification(
                            tokenizer=ner_token_cls_finetuner.tokenizer
                        )
        self.assertIsNotNone(data_collator)

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
        training_args = TrainingArguments(output_dir=output_dir)

        print('training_args', training_args)
        training_args.max_steps = 10
        training_args.warmup_steps = 2
        training_args.evaluation_strategy = 'steps'
        training_args.eval_steps = 10
        training_args.logging_steps = 
        training_args.no_cuda = not torch.cuda.is_available()
        training_args.run_name = None # Set wandb run name to None
        

        eval_result = ner_token_cls_finetuner.finetune(training_args, 
                                   train_dataset=train_dataset,
                                   val_dataset=val_dataset,
                                   test_dataset=test_dataset
        )

        self.assertIsNotNone(eval_result)
        print(eval_result)

        self.assertTrue(os.path.exists(
            os.path.join(training_args.output_dir, 'checkpoint-final', 'pytorch_model.bin')
        ))