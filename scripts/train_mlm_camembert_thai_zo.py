#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:54:52 2020

@author: zo

Written based on
https://github.com/huggingface/transformers/blob/v3.0.2/examples/language-modeling/run_language_modeling.py
https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py
"""

import logging
import glob
import torch
import os
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    CamembertTokenizer,
    TrainingArguments,
    HfArgumentParser,
    RobertaConfig,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer
)
from typing import Optional
from helper import (
    get_field, check_depreciated, check_required
    )


@dataclass
class ModelArguments:
    """
    Arguments for model config.
    """

    tokenizer_name_or_path: str = field(
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dir: str = field(
        metadata={"help": "The input training data dir (dir that contain text files)."}
    )  # Non-standard
    eval_dir: str = field(
        metadata={"help": "The input evaluation data dir (dir that contain text files)."},
    )  # Non-standard
    datasets_cache_dir: str = field(
        default=None, metadata={'help': 'The directory for datasets cache.'}
    )  # Non-standard
    mlm: bool = field(
        default=False,
        metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    # block_size: int = field(
    #     default=512,
    #     metadata={
    #         "help": "Optional input sequence length after tokenization."
    #         "The training dataset will be truncated in block of this size for training."
    #         "Default to the model max input length for single sentence inputs (take into account special tokens)."
    #     },
    # )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum length of sequence '}
        )  # Non-standard

@dataclass
class ArchitectureArguments:
    num_hidden_layers: int = field(
        default=12,
        metadata={'help': 'number of hidden layers (L)'}
        )
    hidden_size: int = field(
        default=768,
        metadata={'help': 'number of hidden size (H)'}
        )
    intermediate_size: int = field(
        default=3072,
        metadata={'help': 'number of intermediate_size'}
        )
    num_attention_head: int = field(
        default=12,
        metadata={'help': 'number of attention head (A)'}
        )


@dataclass
class CustomOthersArguments:
    add_space_token: bool = field(
        default=False, metadata={'help': 'Add spacial token for tokenizer.'}
        )
    ext: str = field(
        default='.txt', metadata={'help': 'Extension of training and evaluation files.'}
        )
    model_dir: Optional[str] = field(
        default=None, metadata={'help': 'Dir of the checkpoint.'}
        )


@dataclass
class CustomTrainingArgument(TrainingArguments):
    """
    Setting new default and copy type and help from parent.

    Use this for now, but this is very ugly do we have other method?
    """

    output_dir: get_field(TrainingArguments, 'output_dir').type = field(
        default='./results',
        metadata=get_field(TrainingArguments, 'output_dir').metadata
        )
    overwrite_output_dir: get_field(TrainingArguments, 'overwrite_output_dir').type = field(
        default=True,
        metadata=get_field(TrainingArguments, 'overwrite_output_dir').metadata
        )
    save_total_limit: get_field(TrainingArguments, 'save_total_limit').type = field(
        default=1,
        metadata=get_field(TrainingArguments, 'save_total_limit').metadata
        )
    save_steps: get_field(TrainingArguments, 'save_steps').type = field(
        default=500,
        metadata=get_field(TrainingArguments, 'save_steps').metadata
        )
    logging_dir: get_field(TrainingArguments, 'logging_dir').type = field(
        default='./logs',
        metadata=get_field(TrainingArguments, 'logging_dir').metadata
        )
    logging_steps: get_field(TrainingArguments, 'logging_steps').type = field(
        default=500,
        metadata=get_field(TrainingArguments, 'logging_steps').metadata
        )
    evaluate_during_training: get_field(TrainingArguments, 'evaluate_during_training').type = field(
        default=False,
        metadata=get_field(TrainingArguments, 'evaluate_during_training').metadata
        )
    eval_steps: get_field(TrainingArguments, 'eval_steps').type = field(
        default=500,
        metadata=get_field(TrainingArguments, 'eval_steps').metadata
        )
    per_device_train_batch_size: get_field(TrainingArguments, 'per_device_train_batch_size').type = field(
        default=32,
        metadata=get_field(TrainingArguments, 'per_device_train_batch_size').metadata
        )
    per_device_eval_batch_size: get_field(TrainingArguments, 'per_device_eval_batch_size').type = field(
        default=64,
        metadata=get_field(TrainingArguments, 'per_device_eval_batch_size').metadata
        )
    gradient_accumulation_steps: get_field(TrainingArguments, 'gradient_accumulation_steps').type = field(
        default=1,
        metadata=get_field(TrainingArguments, 'gradient_accumulation_steps').metadata
        )
    learning_rate: get_field(TrainingArguments, 'learning_rate').type = field(
        default=6e-4,
        metadata=get_field(TrainingArguments, 'learning_rate').metadata
        )
    warmup_steps: get_field(TrainingArguments, 'warmup_steps').type = field(
        default=500,
        metadata=get_field(TrainingArguments, 'warmup_steps').metadata
        )
    weight_decay: get_field(TrainingArguments, 'weight_decay').type = field(
        default=0.01,
        metadata=get_field(TrainingArguments, 'weight_decay').metadata
        )
    adam_epsilon: get_field(TrainingArguments, 'adam_epsilon').type = field(
        default=1e-6,
        metadata=get_field(TrainingArguments, 'adam_epsilon').metadata
        )
    max_grad_norm: get_field(TrainingArguments, 'max_grad_norm').type = field(
        default=1.0,
        metadata=get_field(TrainingArguments, 'max_grad_norm').metadata
        )
    dataloader_drop_last: get_field(TrainingArguments, 'dataloader_drop_last').type = field(
        default=False,
        metadata=get_field(TrainingArguments, 'dataloader_drop_last').metadata
        )
    seed: get_field(TrainingArguments, 'seed').type = field(
        default=1412,
        metadata=get_field(TrainingArguments, 'seed').metadata
        )
    fp16: get_field(TrainingArguments, 'fp16').type = field(
        default=False,
        metadata=get_field(TrainingArguments, 'fp16').metadata
        )
    fp16_opt_level: get_field(TrainingArguments, 'fp16_opt_level').type = field(
        default='O1',
        metadata=get_field(TrainingArguments, 'fp16_opt_level').metadata
        )
    num_train_epochs: get_field(TrainingArguments, 'num_train_epochs').type = field(
        default=None,
        metadata=get_field(TrainingArguments, 'num_train_epochs').metadata
        )
    max_steps: get_field(TrainingArguments, 'max_steps').type = field(
        default=None,
        metadata=get_field(TrainingArguments, 'max_steps').metadata
        )


# Arguments that will be removed but kept for now.
# We should suggest alternative or explain the reason for removing.
# COMPAT_WARN_LIST = [('train_max_length',
#                      DataTrainingArguments.train_max_length,
#                      FutureWarning('train_max_length will be removed, use `block_size` instead.')),
#                     ('eval_max_length',
#                      DataTrainingArguments.eval_max_length,
#                      FutureWarning('eval_max_length will be removed, use `block_size` instead.'))]


def main():
    pass
    """
    Training script for Roberta mask language model (mlm) using huggingface transformers and
    datasets library. The previous one use custom dataset training which does not have
    good performance. This script is re-written to be more conform with the examples
    provided by huggingface and should give better performance and scalibity out of the boxes?

    This try to be compatible with old script as possible but there might be some different?

    In the future, if we are able to drop this entirely and rely on there examples training
    scripts it might be better since that should be the best for performance, compatibility,
    extensiblity, reproducibility.
    """
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            CustomTrainingArgument,
            ArchitectureArguments,
            CustomOthersArguments
            ),
        description="train mlm for roberta with huggingface Trainer"
    )

    (model_args, data_args, training_args,
        arch_args, custom_args) = parser.parse_args_into_dataclasses()

    # Check for arguments that we kept for compatibility but we should remove it later
    # check_depreciated(data_args, COMPAT_WARN_LIST)
    # Workaround if we inherit class from another dataclass with default value
    # we will not be able to use field with required value.
    check_required(training_args)

    # Compatibility
    # if data_args.train_max_length != data_args.eval_max_length:
    #     raise ValueError('unable to set different max_length for training and evaluation')
    # else:
    #     if data_args.train_max_length is not None:
    #         data_args.block_size = data_args.train_max_length

    # Load tokenizer from pretrained model
    tokenizer = CamembertTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    if custom_args.add_space_token:
        logging.info('Special token `<th_roberta_space_token>` will be added'
                     'to the CamembertTokenizer instance.')
        tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED',
                                               '<th_roberta_space_token>']

    # Create config for LM model
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        type_vocab_size=1,
        # roberta base as default
        num_hidden_layers=arch_args.num_hidden_layers,  # L
        hidden_size=arch_args.hidden_size,  # H
        intermediate_size=arch_args.intermediate_size,
        num_attention_head=arch_args.num_attention_head,  # A
        # roberta large
        # num_hidden_layers=24,
        # hidden_size=1024,
        # intermediate_size=4096,
        # num_attention_head=16
    )

    # Initialize model
    model = RobertaForMaskedLM(config=config)

    data_args.train_files = glob.glob(f'{data_args.train_dir}/*.{custom_args.ext}')
    data_args.validation_files = glob.glob(f'{data_args.eval_dir}/*.{custom_args.ext}')

    if custom_args.ext == 'txt':
        # Skip downloading processing script
        dataset_processing_path = '../external_scripts/datasets/text.py'
        if not os.path.exists(dataset_processing_path):
            dataset_processing_path = 'text'
        datasets = load_dataset(dataset_processing_path,
                                data_files={'train': data_args.train_files,
                                            'validation': data_args.validation_files},
                                cache_dir=data_args.datasets_cache_dir)
    else:
        raise NotImplementedError(f'not supprt {custom_args.ext},'
                                  f'but this should be possible to support.')

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability)

    logging.info(" Number of devices: %d", training_args.n_gpu)
    logging.info(" Device: %s", training_args.device)
    logging.info(" Local rank: %s", training_args.local_rank)
    logging.info(" FP16 Training: %s", training_args.fp16)

    if custom_args.model_dir is not None:
        model_path = os.path.join(custom_args.model_dir, 'pytorch_model.bin')
        print(f'[INFO] Load pretrianed model (state_dict) from {model_path}')
        # Use strict=False to kept model compatible with older version,
        # so we can bumb transformers version up and use new datasets library
        # see this issues https://github.com/huggingface/transformers/issues/6882
        # The program itself will run but does it has any side effect?
        # Maybe bad idea?
        model.load_state_dict(state_dict=torch.load(model_path), strict=False)
        # If we did not add strict=False, this will raise Error since the keys are not match
        # RuntimeError: Error(s) in loading state_dict for RobertaForMaskedLM:
        #     Missing key(s) in state_dict: "roberta.embeddings.position_ids".
        #     Unexpected key(s) in state_dict: "roberta.pooler.dense.weight",
        # "roberta.pooler.dense.bias".

    # The following codes are from
    # https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py
    
    # line by line

    def tokenize_function(examples):
            # Remove empty lines
            import pdb; pdb.set_trace()

            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(examples["text"],
                             pad_to_max_length=False,
                             truncation=True,
                             max_length=data_args.max_seq_length)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=['text'],
        load_from_cache_file=not data_args.overwrite_cache,
    )
    
    # End datasets processing sections

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        prediction_loss_only=True
    )

    # train
    logging.info(" Start training.")
    if custom_args.model_dir is not None:
        trainer.train(model_path=custom_args.model_dir)
    else:
        trainer.train()
    # save
    output_model_dir = os.path.join(training_args.output_dir, 'roberta_thai')
    logging.info(" Save final model to '%s'.", output_model_dir)
    trainer.save_model(output_model_dir)

    # evaluate
    trainer.evaluate()


if __name__ == '__main__':
    main()
