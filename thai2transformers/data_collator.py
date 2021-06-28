import math
from typing import List, Dict, Union, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
import random
from bisect import bisect
from transformers.data.data_collator import DataCollatorForLanguageModeling, _collate_batch, tolist
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

SPECIAL_TOKEN_NAMES = ['bos_token', 'eos_token', 'sep_token', 'cls_token']

@dataclass
class DataCollatorForSpanLevelMask(DataCollatorForLanguageModeling):
    """
    Data collator used for span-level masked language modeling
     
    adapted from NGramMaskGenerator class
    
    https://github.com/microsoft/DeBERTa/blob/11fa20141d9700ba2272b38f2d5fce33d981438b/DeBERTa/apps/tasks/mlm_task.py#L36
    and
    https://github.com/zihangdai/xlnet/blob/0b642d14dd8aec7f1e1ecbf7d6942d5faa6be1f0/data_utils.py

    """
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    max_gram: int = 3
    keep_prob: float = 0.0
    mask_prob: float = 1.0
    max_preds_per_seq: int = None
    max_seq_len: int = 510

    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15, *args, **kwargs):
        super().__init__(tokenizer, mlm=mlm, mlm_probability=mlm_probability)

        assert self.mask_prob + self.keep_prob <= 1, \
            f'The prob of using [MASK]({self.mask_prob}) and the prob of using original token({self.keep_prob}) should between [0,1]'

        if self.max_preds_per_seq is None:
            self.max_preds_per_seq = math.ceil(self.max_seq_len * self.mlm_probability / 10) * 10
            self.mask_window = int(1 / self.mlm_probability) # make ngrams per window sized context
        self.vocab_words = list(self.tokenizer.get_vocab().keys())
        self.vocab_mapping = self.tokenizer.get_vocab()
        
        self.special_token_ids = [ self.vocab_mapping[self.tokenizer.special_tokens_map[special_token_name]] \
                           for special_token_name in SPECIAL_TOKEN_NAMES]
    
        self.ngrams = np.arange(1, self.max_gram + 1, dtype=np.int64)
        _pvals = 1. / np.arange(1, self.max_gram + 1)
        self.pvals = _pvals / _pvals.sum(keepdims=True)

    def _choice(self, rng, data, p):
        cul = np.cumsum(p)
        x = rng.random()*cul[-1]
        id = bisect(cul, x)
        return data[id]

    def _per_token_mask(self, idx, tokens, rng, mask_prob, keep_prob):
        label = tokens[idx]
        mask = self.tokenizer.mask_token
        rand = rng.random()
        if rand < mask_prob:
            new_label = mask
        elif rand < mask_prob + keep_prob:
            new_label = label
        else:
            new_label = rng.choice(self.vocab_words)

        tokens[idx] = new_label

        return label

    def _mask_tokens(self, tokens: List[str], rng=random, **kwargs):

        indices = [i for i in range(len(tokens)) if tokens[i] not in self.special_token_ids]
        unigrams = [ [idx] for idx in indices ]
        num_to_predict = min(self.max_preds_per_seq, max(1, int(round(len(tokens) * self.mlm_probability))))
           
        offset = 0
        mask_grams = np.array([False]*len(unigrams))
        while offset < len(unigrams):
            n = self._choice(rng, self.ngrams, p=self.pvals)
            ctx_size = min(n * self.mask_window, len(unigrams)-offset)
            m = rng.randint(0, ctx_size-1)
            s = offset + m
            e = min(offset + m + n, len(unigrams))
            offset = max(offset+ctx_size, e)
            mask_grams[s:e] = True

        target_labels = [None]*len(tokens)
        w_cnt = 0
        for m,word in zip(mask_grams, unigrams):
            if m:
                for idx in word:
                    label = self._per_token_mask(idx, tokens, rng, self.mask_prob, self.keep_prob)
                    target_labels[idx] = label
                    w_cnt += 1
                if w_cnt >= num_to_predict:
                    break

        target_labels = [self.vocab_mapping[x] if x else -100 for x in target_labels]
        return tokens, target_labels    


    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = []
        inputs_masked = []
        
        for input in inputs:
            input_tokens = self.tokenizer.convert_ids_to_tokens(input)
            input_masked, _labels = self._mask_tokens(input_tokens)
            input_masked_ids = self.tokenizer.convert_tokens_to_ids(input_masked)
            inputs_masked.append(input_masked_ids)
            
            labels.append(_labels)
      
        return inputs_masked, labels
