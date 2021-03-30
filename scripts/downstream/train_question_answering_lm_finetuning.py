import argparse
import math
import os
from functools import partial
from tqdm.auto import tqdm
from typing import Collection, Callable
from pathlib import Path
from sklearn import preprocessing
import pandas as pd
import numpy as np
import wandb
import torch
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
    get_constant_schedule, 
    AutoTokenizer, 
    AutoModel,
    AutoModelForQuestionAnswering, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    CamembertTokenizerFast,
    BertTokenizerFast,
    BertConfig,
    XLMRobertaTokenizerFast,
    XLMRobertaConfig,
    DataCollatorWithPadding,
)

from datasets import (
    load_dataset,
    load_from_disk,
)

from thai2transformers.metrics import (
    squad_newmm_metric,
    question_answering_metrics,
)
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
)
from thai2transformers.preprocess import (
    prepare_qa_train_features,
    prepare_qa_validation_features,
)

TOKENIZERS = {
    'wangchanberta-base-att-spm-uncased': AutoTokenizer,
    'xlm-roberta-base': AutoTokenizer,
    'bert-base-multilingual-cased': AutoTokenizer,
    'wangchanberta-base-wiki-newmm': ThaiWordsNewmmTokenizer,
    'wangchanberta-base-wiki-ssg': ThaiWordsSyllableTokenizer,
    'wangchanberta-base-wiki-sefr': FakeSefrCutTokenizer,
    'wangchanberta-base-wiki-spm': ThaiRobertaTokenizer,
}
WANGCHANBERTA_MODELS = [
    'wangchanberta-base-att-spm-uncased',
    'wangchanberta-base-wiki-newmm',
    'wangchanberta-base-wiki-ssg',
    'wangchanberta-base-wiki-sefr',
    'wangchanberta-base-wiki-spm',
] 

#lowercase when using uncased model
def lowercase_example(example):
    example[args.question_col] =  example[args.question_col].lower()
    example[args.context_col] =  example[args.context_col].lower()
    example[args.answers_col][args.text_col] =  [example[args.answers_col][args.text_col][0].lower()]
    return example

def init_model_tokenizer(model_name, model_max_length):
    
    if model_name in TOKENIZERS.keys():
        tokenizer = TOKENIZERS[model_name].from_pretrained(
                        f'airesearch/{model_name}' if model_name in WANGCHANBERTA_MODELS else model_name,
                        revision='main',
                        model_max_length=model_max_length,)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        model_max_length=model_max_length,)
        
    
    model = AutoModelForQuestionAnswering.from_pretrained(
            f'airesearch/{model_name}' if model_name in WANGCHANBERTA_MODELS else model_name,
            revision='main',)

    print(f'\n[INFO] Model architecture: {model} \n\n')
    print(f'\n[INFO] tokenizer: {tokenizer} \n\n')

    return model, tokenizer

def init_trainer(model, 
                 train_dataset, 
                 val_dataset,
                 args, 
                 data_collator,
                 tokenizer,): 
        
    training_args = TrainingArguments(
                        num_train_epochs=args.num_train_epochs,
                        per_device_train_batch_size=args.batch_size,
                        per_device_eval_batch_size=args.batch_size,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        learning_rate=args.learning_rate,
                        warmup_ratio=args.warmup_ratio,
                        weight_decay=args.weight_decay,
                        adam_epsilon=args.adam_epsilon,
                        max_grad_norm=args.max_grad_norm,
                        #checkpoint
                        output_dir=args.output_dir,
                        overwrite_output_dir=True,
                        save_total_limit=3,
                        #logs
                        logging_dir=args.log_dir,
                        logging_first_step=False,
                        logging_steps=args.logging_steps,
                        #eval
                        evaluation_strategy='epoch',
                        load_best_model_at_end=True,
                        #others
                        seed=args.seed,
                        fp16=args.fp16,
                        fp16_opt_level=args.fp16_opt_level,
                        dataloader_drop_last=False,
                        no_cuda=args.no_cuda,
                        metric_for_best_model=args.metric_for_best_model,
                        prediction_loss_only=False,
                        run_name=args.run_name
                    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return trainer, training_args

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('--model_name', type=str, help='Model names on Huggingface for tokenizers and architectures')
    parser.add_argument('--dataset_name', help='Specify the dataset name to finetune. Currently, sequence classification datasets include `thaiqa_squad` and `iapp_wiki_qa_squad`.')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--lowercase', action='store_true', default=False)

    # Finetuning
    parser.add_argument('--model_max_length', type=int, default=416)
    parser.add_argument('--pad_on_right', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--greater_is_better', action='store_true')
    parser.add_argument('--metric_for_best_model', type=str, default='loss')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--n_best_size', type=int, default=20)
    parser.add_argument('--max_answer_length', type=int, default=100)
    parser.add_argument('--doc_stride', type=int, default=128)
    
    #column names; default to SQuAD naming
    parser.add_argument('--question_col', type=str, default='question')
    parser.add_argument('--context_col', type=str, default='context')
    parser.add_argument('--question_id_col', type=str, default='question_id')
    parser.add_argument('--answers_col', type=str, default='answers')
    parser.add_argument('--text_col', type=str, default='text')
    parser.add_argument('--start_col', type=str, default='answer_start')

    # wandb
    parser.add_argument('--run_name', type=str, default=None)

    args = parser.parse_args()

    # Set seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f'\n\n[INFO] Initialize model and tokenizer')
    model, tokenizer = init_model_tokenizer(model_name=args.model_name,
                                            model_max_length=args.model_max_length)
    data_collator = DataCollatorWithPadding(tokenizer,
                                            padding=True,
                                            pad_to_multiple_of=8 if args.fp16 else None)

    print(f'\n\n[INFO] Dataset: {args.dataset_name}')
    if args.dataset_name == 'iapp_thaiqa':
        print(f'\n\n[INFO] For `iapp_thaiqa` dataset where you run `combine_iapp_qa.py` and save combined dataset (use directory path as `dataset_name`)')
        datasets = load_from_disk(args.dataset_name)
    else:
        datasets = load_dataset(args.dataset_name)
    print(f'dataset: {datasets}')

    if args.lowercase:
        print(f'\n\n[INFO] Lowercaing datasets')
        datasets = datasets.map(lowercase_example)
        
    print(f'\n\n[INFO] Prepare training features')
    tokenized_datasets = datasets.map(lambda x: prepare_qa_train_features(x, tokenizer), 
                                      batched=True, 
                                      remove_columns=datasets["train"].column_names)

    print(f'\n[INFO] Number of train examples = {len(datasets["train"])}')
    print(f'[INFO] Number of batches per epoch (training set) = {math.ceil(len(datasets["train"]) / args.batch_size)}')
    print(f'[INFO] Number of validation examples = {len(datasets["validation"])}')
    print(f'[INFO] Number of batches per epoch (validation set) = {math.ceil(len(datasets["validation"]))}')
    print(f'[INFO] Warmup ratio = {args.warmup_ratio}')
    print(f'[INFO] Learning rate: {args.learning_rate}')
    print(f'[INFO] Logging steps: {args.logging_steps}')
    print(f'[INFO] FP16 training: {args.fp16}\n')

    trainer, training_args = init_trainer(model=model,
                                train_dataset=tokenized_datasets['train'],
                                val_dataset=tokenized_datasets['validation'],
                                args=args,
                                data_collator=data_collator,
                                tokenizer=tokenizer,)

    print('[INFO] TrainingArguments:')
    print(training_args)
    print('\n')

    print('\nBegin model finetuning.')
    trainer.train()
    print('Done.\n')


    print('[INFO] Done.\n')
    print('[INDO] Begin saving best checkpoint.')
    trainer.save_model(os.path.join(args.output_dir, 'checkpoint-best'))

    print('[INFO] Done.\n')
    print('\nBegin model evaluation on test set.')
    
    result = question_answering_metrics(datasets=datasets['test'], 
                                        trainer=trainer,
                                        metric=squad_newmm_metric,
                                        n_best_size=args.n_best_size,
                                        max_answer_length=args.max_answer_length,
                                        question_col=args.question_col,
                                        context_col=args.context_col,
                                        question_id_col=args.question_id_col,
                                        answers_col=args.answers_col,
                                        text_col=args.text_col,
                                        start_col=args.start_col,
                                        pad_on_right=args.pad_on_right,
                                        max_length=args.model_max_length,
                                        doc_stride=args.doc_stride,)

    print(f'Evaluation on test set (dataset: {args.dataset_name})')    
    
    for key, value in result.items():
        print(f'{key} : {value:.4f}')
        wandb.run.summary[f'test-set_{key}'] = value