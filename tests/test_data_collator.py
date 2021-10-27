import unittest
import pytest
import os
import sys
import shutil
import tempfile
from typing import Collection, Iterable, List, Union, Dict, Callable

import torch
from transformers import BertTokenizer, is_torch_available, set_seed
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import DataCollatorForLanguageModeling
    from thai2transformers.data_collator import DataCollatorForSpanLevelMask


@require_torch
class DataCollatorIntegrationTest(unittest.TestCase):
    """
    Code is from https://github.com/huggingface/transformers/blob/master/tests/test_data_collator.py#L37
    """
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.txt")
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def _test_no_pad_and_pad(self, no_pad_features, pad_features):
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 16)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 16)))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 16)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 16)))

        tokenizer._pad_token = None
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        with self.assertRaises(ValueError):
            # Expect error due to padding token missing
            data_collator(pad_features)

        set_seed(42)  # For reproducibility
        tokenizer = BertTokenizer(self.vocab_file)
        data_collator = DataCollatorForLanguageModeling(tokenizer)
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 10)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 10)))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

        data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8)
        batch = data_collator(no_pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 16)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 16)))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))

        batch = data_collator(pad_features)
        self.assertEqual(batch["input_ids"].shape, torch.Size((2, 16)))
        self.assertEqual(batch["labels"].shape, torch.Size((2, 16)))

        masked_tokens = batch["input_ids"] == tokenizer.mask_token_id
        self.assertTrue(torch.any(masked_tokens))
        self.assertTrue(all(x == -100 for x in batch["labels"][~masked_tokens].tolist()))


    def test_data_collator_for_language_modeling(self):
        no_pad_features = [{"input_ids": list(range(10))}, {"input_ids": list(range(10))}]
        pad_features = [{"input_ids": list(range(5))}, {"input_ids": list(range(10))}]
        self._test_no_pad_and_pad(no_pad_features, pad_features)

        no_pad_features = [list(range(10)), list(range(10))]
        pad_features = [list(range(5)), list(range(10))]
        self._test_no_pad_and_pad(no_pad_features, pad_features)

    