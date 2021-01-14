#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:12:12 2021

@author: z
"""

import pprint
import numpy as np
import torch
import logging
import transformers
import datasets
from dataclasses import dataclass, field
from typing import Optional
from custom_data_collator import DataCollatorForTokenClassification
from datasets import load_dataset, load_metric, Dataset
from functools import lru_cache
from seqeval.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
# thai2transformers
try:
    from thai2transformers.tokenizers import (
        ThaiRobertaTokenizer, ThaiWordsNewmmTokenizer,
        ThaiWordsSyllableTokenizer, FakeSefrCutTokenizer,)
    from thai2transformers import metrics as t2f_metrics
except ModuleNotFoundError:
    import sys
    sys.path.append('..')  # path hacking
    from thai2transformers import metrics as t2f_metrics
    from thai2transformers.tokenizers import (
        ThaiRobertaTokenizer, ThaiWordsNewmmTokenizer,
        ThaiWordsSyllableTokenizer, FakeSefrCutTokenizer,)

from transformers import (AutoConfig, RobertaForTokenClassification,
                          Trainer, TrainingArguments,
                          AutoModelForTokenClassification, AutoTokenizer,
                          BertForTokenClassification,
                          HfArgumentParser, CamembertTokenizer)

logger = logging.getLogger(__name__)


def is_main_process(rank):
    return rank in [-1, 0]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )
    tokenizer_name_or_path: str = field(
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    tokenizer_type: Optional[str] = field(
        default='AutoTokenizer',
        metadata={'help': 'type of tokenizer'}
    )


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        metadata={'help': 'name of dataset'}
    )
    label_name: str = field(
        metadata={'help': 'name of label column (ex. ner_tags, pos_tags)'}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={'help': 'max length of a sequence'}
    )


@dataclass
class CustomArguments:
    no_train_report: bool = field(
        default=False,
        metadata={'help': 'do not report training set metrics'}
    )
    no_eval_report: bool = field(
        default=False,
        metadata={'help': 'do not report evaluation set metrics'}
    )
    no_test_report: bool = field(
        default=False,
        metadata={'help': 'do not report test set metrics'}
    )
    lst20_data_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'path to lst20 dataset'}
    )
    space_token:str = field(
        default='<_>',
        metadata={'help': 'specify custom space token'}
    )
    lowercase:str = field(
        default=False,
        metadata={'help': 'Apply lowercase to input texts'}
    )


parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
                           TrainingArguments, CustomArguments))

model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
logger.info("Training/evaluation parameters %s", training_args)

logger.info("Data parameters %s", data_args)
logger.info("Model parameters %s", model_args)
logger.info("Custom args %s", custom_args)

if model_args.tokenizer_type == 'AutoTokenizer':
    # bert-base-multilingual-cased
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    tokenizer.add_tokens(custom_args.space_token)
elif model_args.tokenizer_type == 'ThaiRobertaTokenizer':
    tokenizer = ThaiRobertaTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path)
elif model_args.tokenizer_type == 'ThaiWordsNewmmTokenizer':
    tokenizer = ThaiWordsNewmmTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path)
elif model_args.tokenizer_type == 'ThaiWordsSyllableTokenizer':
    tokenizer = ThaiWordsSyllableTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path)
elif model_args.tokenizer_type == 'CamembertTokenizer':
    tokenizer = CamembertTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path)
    tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', custom_args.space_token]
    # Override "SPACE_TOKEN" variable with user supplied space token
    # SPACE_TOKEN = custom_args.space_token
    logger.info("[INFO] space_token = `%s`", custom_args.space_token)
elif model_args.tokenizer_type == 'skip':
    logging.info('Skip tokenizer')
else:
    raise NotImplementedError(f'tokenizer_type {model_args.tokenizer_type} is not implemeted.')

text_col = 'tokens'
label_col = data_args.label_name

if data_args.dataset_name == 'thainer':
    # we cannot use this in its current form since there is a bug
    # which cause ner_tags to appear as 27 for all examples
    # the current hacking solution is to run this first once
    # this should generate cache folder (depend on arguments given)
    # like this .cache/huggingface/modules/datasets_modules/datasets/thainer/.../thainer.py
    # add this line
    # >>> splits[2] = splits[2].rstrip()
    # ner_tag = splits[2] if splits[2] in self._NER_TAGS else "O"
    # this will strip the newline out of last spilted text first
    # you will also need to merge *ไม่ยืนยัน tags into "O"
    # by removing it from _NER_TAGS
    dataset = load_dataset("thainer")
    # Remove tag: ไม่ยืนยัน
    if label_col == 'ner_tags':
        dataset['train'] = dataset['train'].map(lambda examples: {'ner_tags': [i if i not in [13,26] else 27 for i in examples[label_col]]})

    label_maps = {i: name for i, name in
                  enumerate(dataset['train'].features[label_col].feature.names)}
    label_names = dataset['train'].features[label_col].feature.names
    num_labels = dataset['train'].features[label_col].feature.num_classes
elif data_args.dataset_name == 'lst20':
    dataset = load_dataset('lst20', data_dir=custom_args.lst20_data_dir)
    label_maps = {i: name for i, name in
                  enumerate(dataset['train'].features[label_col].feature.names)}
    label_names = dataset['train'].features[label_col].feature.names
    num_labels = dataset['train'].features[label_col].feature.num_classes
elif data_args.dataset_name == 'dummytest':
    def generat_dummy_dataset(size, max_length, max_token_length, label_names, label_sizes):
        d = {'tokens': []}
        c = {}
        chars = [chr(i) for i in range(97, 123, 1)]
        for label_name, label_size in zip(label_names, label_sizes):
            d[label_name] = []
            c[label_name] = list(range(label_size))
        for i in range(size):
            length = np.random.randint(1, max_length)
            d['tokens'].append([''.join(np.random.choice(chars, size=max_token_length))
                                for _ in range(length)])
            for label_name in label_names:
                dummy_labels = np.random.choice(c[label_name],
                                                size=length)
                d[label_name].append(dummy_labels)
        return Dataset.from_dict(d)
    dataset = datasets.DatasetDict(
        {'train': generat_dummy_dataset(50, 50, 8, ['ner_tags', 'pos_tags'], [10, 20]),
         'validation': generat_dummy_dataset(10, 50, 8, ['ner_tags', 'pos_tags'], [10, 20]),
         'test': generat_dummy_dataset(10, 50, 8, ['ner_tags', 'pos_tags'], [10, 20])
         })
    label_maps = {i: str(name) for i, name in
                  enumerate(range(20))}
    if 'ner' in label_col.lower():
        label_names = ['O'] + ['B-' + str(name) for i, name in enumerate(range(1, 20))]
    else:
        label_names = [str(name) for i, name in enumerate(range(20))]
    num_labels = 20
else:
    raise NotImplementedError


def pre_tokenize(token, space_token):
    token = token.replace(' ', space_token)
    return token


@lru_cache(maxsize=None)
def cached_tokenize(token, space_token, lowercase):
    if lowercase:
        token = token.lower()
    token = pre_tokenize(token, space_token)
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
    return ids


def preprocess(examples, space_token, lowercase):
    tokens = []
    labels = []
    old_positions = []
    for example_tokens, example_labels in zip(examples[text_col], examples[label_col]):
        new_example_tokens = []
        new_example_labels = []
        old_position = []
        for i, (token, label) in enumerate(zip(example_tokens, example_labels)):
            # tokenize each already pretokenized tokens with our own tokenizer.
            toks = cached_tokenize(token, space_token, lowercase=custom_args.lowercase)
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
        add_special_tokens=True, max_length=data_args.max_length)
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


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

if data_args.dataset_name == 'thainer':
    # exclude last example, thainer dataset generator adding blank into the last example.
    split = dataset['train'].train_test_split(test_size=1, shuffle=False)
    train_dataset = split['train']

    split = train_dataset.train_test_split(
        train_size=0.8, test_size=0.2, seed=2020)
    train_dataset = split['train']
    non_train_dataset = split['test']

    split = non_train_dataset.train_test_split(
        train_size=0.5, test_size=0.5, seed=2020)
    val_dataset = split['train']
    test_dataset = split['test']
    # preprocess
    train_dataset = Dataset.from_dict(preprocess(train_dataset, space_token=custom_args.space_token))
    val_dataset = Dataset.from_dict(preprocess(val_dataset, space_token=custom_args.space_token))
    test_dataset = Dataset.from_dict(preprocess(test_dataset, space_token=custom_args.space_token))
    # val set need padding to fix problem with trainer
    val_dataset = Dataset.from_dict(data_collator(val_dataset))
    test_dataset = Dataset.from_dict(data_collator(test_dataset))
elif data_args.dataset_name == 'lst20':
    # exclude last example, lst20 dataset generator adding blank into the last example.
    split = dataset['train'].train_test_split(test_size=1, shuffle=False)
    train_dataset = split['train']

    split = dataset['validation'].train_test_split(test_size=1, shuffle=False)
    val_dataset = split['train']

    split = dataset['test'].train_test_split(test_size=1, shuffle=False)
    test_dataset = split['train']

    train_dataset = Dataset.from_dict(preprocess(train_dataset, space_token=custom_args.space_token))
    val_dataset = Dataset.from_dict(preprocess(val_dataset, space_token=custom_args.space_token))
    test_dataset = Dataset.from_dict(preprocess(test_dataset, space_token=custom_args.space_token))
    # val set need padding to fix problem with trainer
    val_dataset = Dataset.from_dict(data_collator(val_dataset))
    test_dataset = Dataset.from_dict(data_collator(test_dataset))
elif data_args.dataset_name == 'dummytest':
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    train_dataset = Dataset.from_dict(preprocess(train_dataset, space_token=custom_args.space_token))
    val_dataset = Dataset.from_dict(preprocess(val_dataset, space_token=custom_args.space_token))
    test_dataset = Dataset.from_dict(preprocess(test_dataset, space_token=custom_args.space_token))
    val_dataset = Dataset.from_dict(data_collator(val_dataset))
    test_dataset = Dataset.from_dict(data_collator(test_dataset))
else:
    raise NotImplementedError

model = AutoModelForTokenClassification.from_pretrained(
    model_args.model_name_or_path, num_labels=num_labels)

if model.config.vocab_size == len(tokenizer) - len(tokenizer.get_added_vocab()):
    # resize to accomodate added token
    model.resize_token_embeddings(len(tokenizer))
elif model.config.vocab_size == len(tokenizer) and len(tokenizer.get_added_vocab()) > 0:
    logger.warning('model might already accomodate added token')
else:
    logger.warning(f'model vocab size ({model.config.vocab_size}) is not equal to'
                   f'tokenizer ({len(tokenizer)}), '
                   'this might cause from tokenizer missmatch with model or added vocabulary')
    raise ValueError

metric = load_metric("seqeval")


def get_batch(obj, batch_size):
    i = 0
    r = obj[i * batch_size: i * batch_size + batch_size]
    yield r
    i += 1
    while i * batch_size < len(obj):
        r = obj[i * batch_size: i * batch_size + batch_size]
        yield r
        i += 1


def agg_preds_labels(model, dataset, device=torch.device('cuda')):
    agg_chunk_preds = []
    agg_chunk_labels = []
    model.to(device)
    for step, batch in enumerate(get_batch(dataset, training_args.per_device_eval_batch_size)):
        labels = batch['labels']
        old_positions = batch['old_positions']
        dont_include = ['labels', 'old_positions']
        batch = {k: torch.tensor(v, dtype=torch.int64).to(device) for k, v in batch.items()
                 if k not in dont_include}

        preds, = model(**batch)
        preds = preds.argmax(2)
        preds = preds.tolist()

        use_idxs = [[i for i, e in enumerate(label) if e != -100]
                    for label in labels]
        true_preds = [[preds[j][i] for i in use_idx]
                      for j, use_idx in enumerate(use_idxs)]
        true_labels = [[labels[j][i] for i in use_idx]
                       for j, use_idx in enumerate(use_idxs)]
        true_old_positions = [[old_positions[j][i] for i in use_idx]
                              for j, use_idx in enumerate(use_idxs)]
        chunk_preds = []
        chunk_labels = []
        for i, old_position in enumerate(true_old_positions):
            cur_pos = -100
            chunk_preds.append([])
            chunk_labels.append([])
            for j, pos in enumerate(old_position):
                if pos != cur_pos:
                    cur_pos = pos
                    chunk_preds[-1].append(true_preds[i][j])
                    chunk_labels[-1].append(true_labels[i][j])
                elif pos < cur_pos:
                    raise ValueError('later position has higher value than previous one')
        agg_chunk_preds.extend(chunk_preds)
        agg_chunk_labels.extend(chunk_labels)
        print(f'\rProcessed: {len(agg_chunk_preds)} / {len(dataset)}',
              flush=True, end=' ')
    return agg_chunk_labels, agg_chunk_preds


def sk_classification_metrics(labels, preds):
    precision_macro, recall_macro, f1_macro, _ = \
        precision_recall_fscore_support(labels, preds, average='macro')
    precision_micro, recall_micro, f1_micro, _ = \
        precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'nb_samples': len(labels)
    }


def compute_token_metrics(agg_chunk_labels, agg_chunk_preds):
    report = sk_classification_report(sum(agg_chunk_labels, []),
                                      sum(agg_chunk_preds, []), target_names=label_names)
    results = sk_classification_metrics(sum(agg_chunk_labels, []),
                                        sum(agg_chunk_preds, []))
    return results, report


def compute_chunk_metrics(agg_chunk_labels, agg_chunk_preds):
    results = metric.compute(predictions=[[label_maps[e] for e in a] for a in agg_chunk_preds],
                             references=[[label_maps[e] for e in a] for a in agg_chunk_labels])
    report = classification_report([[label_maps[e] for e in a] for a in agg_chunk_labels],
                                   [[label_maps[e] for e in a] for a in agg_chunk_preds])
    return results, report


def t2t_chunk_metrics(agg_chunk_labels, agg_chunk_preds):
    class LabelsPreds:
        label_ids = agg_chunk_labels
        predictions = agg_chunk_preds
    return t2f_metrics.seqeval_classification_metrics(LabelsPreds)


def t2t_sk_classification_metrics(agg_chunk_labels, agg_chunk_preds):
    class LabelsPreds:
        label_ids = agg_chunk_labels
        predictions = agg_chunk_preds
    return t2f_metrics.sk_classification_metrics(LabelsPreds, pred_labs=True)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_maps[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_maps[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    if 'ner' in data_args.label_name:
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    else:
        result = t2t_sk_classification_metrics(sum(true_labels, []),
                                               sum(true_predictions, []))
        result = {k: v for k, v in result.items() if k != 'classification_report'}
        return result


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if training_args.do_train:
    trainer.train()
    trainer.save_model()

if training_args.do_eval:
    trainer.evaluate()


# Preprocess dataset again sometimes something funky occure causing error in report.
if data_args.dataset_name == 'thainer':
    # exclude last example, thainer dataset generator adding blank into the last example.
    split = dataset['train'].train_test_split(test_size=1, shuffle=False)
    train_dataset = split['train']

    split = train_dataset.train_test_split(
        train_size=0.8, test_size=0.2, seed=2020)
    train_dataset = split['train']
    non_train_dataset = split['test']

    split = non_train_dataset.train_test_split(
        train_size=0.5, test_size=0.5, seed=2020)
    val_dataset = split['train']
    test_dataset = split['test']
    # preprocess
    train_dataset = Dataset.from_dict(preprocess(train_dataset, space_token=custom_args.space_token))
    val_dataset = Dataset.from_dict(preprocess(val_dataset, space_token=custom_args.space_token))
    test_dataset = Dataset.from_dict(preprocess(test_dataset, space_token=custom_args.space_token))
    # val set need padding to fix problem with trainer
    train_dataset = Dataset.from_dict(data_collator(train_dataset))
    val_dataset = Dataset.from_dict(data_collator(val_dataset))
    test_dataset = Dataset.from_dict(data_collator(test_dataset))
elif data_args.dataset_name == 'lst20':
    # exclude last example, lst20 dataset generator adding blank into the last example.
    split = dataset['train'].train_test_split(test_size=1, shuffle=False)
    train_dataset = split['train']

    split = dataset['validation'].train_test_split(test_size=1, shuffle=False)
    val_dataset = split['train']

    split = dataset['test'].train_test_split(test_size=1, shuffle=False)
    test_dataset = split['train']

    train_dataset = Dataset.from_dict(preprocess(train_dataset, space_token=custom_args.space_token))
    val_dataset = Dataset.from_dict(preprocess(val_dataset, space_token=custom_args.space_token))
    test_dataset = Dataset.from_dict(preprocess(test_dataset, space_token=custom_args.space_token))
    # val set need padding to fix problem with trainer
    train_dataset = Dataset.from_dict(data_collator(train_dataset))
    val_dataset = Dataset.from_dict(data_collator(val_dataset))
    test_dataset = Dataset.from_dict(data_collator(test_dataset))
elif data_args.dataset_name == 'dummytest':
    train_dataset = dataset['train']
    train_dataset = Dataset.from_dict(preprocess(train_dataset, space_token=custom_args.space_token))
    train_dataset = Dataset.from_dict(data_collator(train_dataset))
    val_dataset = dataset['validation']
    val_dataset = Dataset.from_dict(preprocess(val_dataset, space_token=custom_args.space_token))
    val_dataset = Dataset.from_dict(data_collator(val_dataset))
else:
    raise NotImplementedError


if not custom_args.no_train_report:
    agg_chunk_labels, agg_chunk_preds = agg_preds_labels(model, train_dataset)
    agg_chunk_labels = [[label_maps[e] for e in a] for a in agg_chunk_labels]
    agg_chunk_preds = [[label_maps[e] for e in a] for a in agg_chunk_preds]
    if 'ner' in data_args.label_name:
        result = t2t_chunk_metrics(agg_chunk_labels, agg_chunk_preds)
    else:
        result = t2t_sk_classification_metrics(sum(agg_chunk_labels, []),
                                               sum(agg_chunk_preds, []))
    print('[ Train Result ]')
    pprint.pprint({k: v for k, v in result.items() if k != 'classification_report'})
    print(result['classification_report'])

if not custom_args.no_eval_report:
    agg_chunk_labels, agg_chunk_preds = agg_preds_labels(model, val_dataset)
    agg_chunk_labels = [[label_maps[e] for e in a] for a in agg_chunk_labels]
    agg_chunk_preds = [[label_maps[e] for e in a] for a in agg_chunk_preds]
    if 'ner' in data_args.label_name:
        result = t2t_chunk_metrics(agg_chunk_labels, agg_chunk_preds)
    else:
        result = t2t_sk_classification_metrics(sum(agg_chunk_labels, []),
                                               sum(agg_chunk_preds, []))
    print('[ Val Result ]')
    pprint.pprint({k: v for k, v in result.items() if k != 'classification_report'})
    print(result['classification_report'])


if not custom_args.no_test_report:
    agg_chunk_labels, agg_chunk_preds = agg_preds_labels(model, test_dataset)
    agg_chunk_labels = [[label_maps[e] for e in a] for a in agg_chunk_labels]
    agg_chunk_preds = [[label_maps[e] for e in a] for a in agg_chunk_preds]
    if 'ner' in data_args.label_name:
        result = t2t_chunk_metrics(agg_chunk_labels, agg_chunk_preds)
    else:
        result = t2t_sk_classification_metrics(sum(agg_chunk_labels, []),
                                               sum(agg_chunk_preds, []))
    print('[ Test Result ]')
    pprint.pprint({k: v for k, v in result.items() if k != 'classification_report'})
    print(result['classification_report'])
