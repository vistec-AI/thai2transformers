#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:12:12 2021

@author: z
"""

import datasets
import pickle

try:
    from thai2transformers.tokenizers import sefr_cut_tokenize, SPACE_TOKEN
except ModuleNotFoundError:
    import sys
    sys.path.append('..')  # path hacking
    from thai2transformers.tokenizers import sefr_cut_tokenize, SPACE_TOKEN


class SefrCacheTokenizer:
    def __init__(self):
        self.cached_tokenized_token = {}

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
            self.update(d)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.cached_tokenized_token, f)

    def tokenize(self, token):
        if token not in self.cached_tokenized_token:
            raise ValueError(f'Please cache tokenized value for "{token}" token first.')
        else:
            return self.cached_tokenized_token[token]

    def update(self, d):
        self.cached_tokenized_token.update(d)


def pre_tokenize(token):
    token = token.replace(' ', SPACE_TOKEN)
    return token


def main():
    sefr_cache_tokenizer = SefrCacheTokenizer()
    for dataset_name in ['thainer', 'lst20']:
        if dataset_name != 'lst20':
            dataset = datasets.load_dataset(dataset_name)
        else:
            dataset = datasets.load_dataset(
                dataset_name, data_dir='../data/input/datasets/LST20_Corpus')
        for key in dataset.keys():
            tokens = sum(dataset[key]['tokens'], [])
            tokens = [pre_tokenize(e) for e in tokens]
            tokenized_tokens = sefr_cut_tokenize(tokens, n_jobs=40)
            sefr_cached = {token: tokenized_token for token, tokenized_token
                           in zip(tokens, tokenized_tokens)}
            sefr_cache_tokenizer.update(sefr_cached)
    sefr_cache_tokenizer.save('sefr_cache_tokenizer_dict.pkl')


if __name__ == '__main__':
    main()
