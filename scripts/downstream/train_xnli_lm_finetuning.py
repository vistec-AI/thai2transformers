
import argparse
import os
import sys
import math
from typing import Dict, List, Optional, Union, Callable
import torch
import wandb
import numpy as np
import multiprocessing
from multiprocessing import Pool

from functools import partial
from dataclasses import dataclass, field

from thai2transformers.preprocess import (
    process_transformers
)

from thai2transformers.datasets import (
    SequenceClassificationDataset
)

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    RobertaForMaskedLM,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    CamembertTokenizer
)
from datasets import load_dataset, load_metric, list_metrics

from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
    SPACE_TOKEN as DEFAULT_SPACE_TOKEN,
    SEFR_SPLIT_TOKEN
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune,
    or train from scratch.
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
class CustomArguments:
    space_token: str = field(
        default='<th_roberta_space_token<',
        metadata={'help': 'specify custom space token'}
    )
    lowercase: bool = field(
        default=False,
        metadata={'help': 'Apply lowercase to input texts'}
    )
    warmup_ratio: float = field(
        default=0.1
    )
    max_seq_length: int = field(
        default=512
    )
    is_sefr: bool = field(
        default=False
    )

sentence1_key, sentence2_key = ("premise", "hypothesis")

import sefr_cut
sefr_cut.load_model('best')
sefr_tokenize = lambda x: sefr_cut.tokenize(x)

def sefr_tokenize_fn(text):
    result = []
    tokenized_text = sefr_tokenize(text.split())
    for i, tok_text in enumerate(tokenized_text):
        _result = []
        _result.extend(tok_text)
        if i != len(tokenized_text) - 1:
            _result.append('<_>')
        result.append('<|>'.join(_result))
    return '<|>'.join(result)

def _preprocess(text, space_token, lowercase, is_sefr) -> str:
    if lowercase:
        text = text.lower()
    if is_sefr:
        text = sefr_tokenize_fn(text)
        assert type(text) == str
        return text
    else:
        return text.replace('  ', space_token).replace(' ', '')

def preprocess_function(examples, tokenizer, max_length, space_token, lowercase, is_sefr):
    
    if is_sefr:
        with Pool(processes=multiprocessing.cpu_count()) as p:
            examples[sentence1_key] = list(p.map(partial(_preprocess,
                                            space_token=space_token,
                                            lowercase=lowercase,
                                            is_sefr=is_sefr), examples[sentence1_key]))
            examples[sentence2_key] = list(p.map(partial(_preprocess,
                                            space_token=space_token,
                                            lowercase=lowercase,
                                            is_sefr=is_sefr), examples[sentence2_key]))
    else:
        examples[sentence1_key] = list(map(partial(_preprocess,
                                            space_token=space_token,
                                            lowercase=lowercase,
                                            is_sefr=False), examples[sentence1_key]))
        examples[sentence2_key] = list(map(partial(_preprocess,
                                            space_token=space_token,
                                            lowercase=lowercase,
                                            is_sefr=False), examples[sentence2_key]))
        
    return tokenizer(examples[sentence1_key], examples[sentence2_key],
                     truncation=True,
                     max_length=max_length)

metric = load_metric('xnli')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':

    parser = HfArgumentParser((TrainingArguments, ModelArguments, CustomArguments))

    training_args, model_args, custom_args = parser.parse_args_into_dataclasses()


    if model_args.tokenizer_type == 'AutoTokenizer':
        # bert-base-multilingual-cased
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    elif model_args.tokenizer_type == 'ThaiRobertaTokenizer':
        tokenizer = ThaiRobertaTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path)
    elif model_args.tokenizer_type == 'ThaiWordsNewmmTokenizer':
        tokenizer = ThaiWordsNewmmTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path)
    elif model_args.tokenizer_type == 'ThaiWordsSyllableTokenizer':
        tokenizer = ThaiWordsSyllableTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path)
    elif model_args.tokenizer_type == 'FakeSefrCutTokenizer':
        tokenizer = FakeSefrCutTokenizer.from_pretrained(
                        model_args.tokenizer_name_or_path)
    elif model_args.tokenizer_type == 'CamembertTokenizer':
        tokenizer = CamembertTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path)
        tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', custom_args.space_token]
        logger.info("[INFO] space_token = `%s`", custom_args.space_token)
   

    print('DEBUD test tokenizer: ', tokenizer.tokenize(f"รถไฟฟ้ามีระยะทาง{custom_args.space_token}<mask> กิโลเมตร a b <mask>"))
    
    # Set seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)


    print('Load model')
    NUM_LABELS = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=NUM_LABELS
    )

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device},"
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)},"
        f"16-bits training: {training_args.fp16}"
    )
    logger.info("Training/evaluation parameters %s", training_args)

    dataset = load_dataset("xnli", 'th')

    encoded_dataset = dataset.map(partial(preprocess_function,
                                          tokenizer=tokenizer,
                                          max_length=custom_args.max_seq_length,
                                          space_token=custom_args.space_token,
                                          lowercase=custom_args.lowercase,
                                          is_sefr=custom_args.is_sefr)
                                  , batched=True)
    
    
    print('DEBUG: encoded_dataset[train][0] input_toks', tokenizer.convert_ids_to_tokens(encoded_dataset['train'][0]['input_ids']))
    metric_name = "accuracy"
    

    warmup_steps = math.ceil(len(encoded_dataset['train']) / training_args.per_device_train_batch_size * custom_args.warmup_ratio * training_args.num_train_epochs)
    training_args.warmup_steps = warmup_steps
    print('INFO: training_args: ', training_args)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()



    print('[INFO] Done.\n')
    print('[INDO] Begin saving best checkpoint.')
    trainer.save_model(os.path.join(training_args.output_dir, 'checkpoint-best'))

    print('[INFO] Done.\n')

    print('\nBegin model evaluation on test set.')
    
    _, label_ids, result = trainer.predict(
                test_dataset=encoded_dataset['test'])
    
    for key, value in result.items():
        print(f'{key} : {value:.4f}')
        wandb.run.summary[f'test-set_{key}'] = value


