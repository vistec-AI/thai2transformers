#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:39:32 2020

@author: zo
"""


def tokenizer_and_model_config_mismatch(config, tokenizer):
    """
    Check for tokenizer and model config miss match.

    Args:
        config:
            model config.
        tokenizer:
            tokenizer.

    Raises:
        ValueError: A special token id in config is different from tokenizer.
    """
    id_check_list = ['bos_token_id', 'eos_token_id', 'pad_token_id',
                     'mask_token_id', 'unk_token_id']
    for id_type in id_check_list:
        if getattr(config, id_type) != getattr(tokenizer, id_type):
            # We should tell how to resolve it.
            raise ValueError('A special token id in config is different from tokenizer')


def block_size_exceed_max_position_embeddings(config, block_size):
    # This will cause position ids automatically create from model
    # to go beyond embedding size of position id embedding.
    # And return not so useful error.
    # This sound like a bug in transformers library.
    # If we got this error the work around for now id to set
    # `max_position_embeddings` of model config to be higher than or equal to
    # `max_seq_len + config.pad_token_id + 1` at least to avoid problem.
    if(block_size > config.max_position_embeddings + config.pad_token_id + 1):
        recommend_block_size = config.max_position_embeddings + config.pad_token_id + 1
        raise ValueError(f'This block size will cause error due to max_position_embeddings. '
                         f'Use this block_size={recommend_block_size} or '
                         f'increase max_position_embeddings')
