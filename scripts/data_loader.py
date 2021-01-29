#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:54:52 2020

@author: zo

"""

import torch
import tempfile
import os
from torch.utils.data.dataset import Dataset
from transformers import (
    PreTrainedTokenizer,
    logging
    )
import mmap
import struct
import bisect
from contextlib import contextmanager
from thai2transformers.helper import get_file_size, _readline_clean_and_strip


logger = logging.get_logger(__name__)


@contextmanager
def callback(func):
    """
    Execute function after exiting context.

    Args:
        func:
            function to execute after exiting context.
    """
    try:
        yield
    finally:
        func()


class MemmapIndexDataset:
    """
    Memory-mapped backed dataset of list of int list.

    Args:
        arr_fname:
            path to array binary.
        idx_arr_fname:
            path to index binary.

    Examples::

        >>> import tempfile
        >>> _, arr_fname = tempfile.mkstemp()
        >>> _, idx_arr_fname = tempfile.mkstemp()
        >>> data = [list(range(i)) for i in range(1, 5)]
        >>> memmap_dataset = MemmapIndexDataset(arr_fname, idx_arr_fname)
        >>> memmap_dataset.clear()  # need to do this if file already exists but empty
        >>> memmap_dataset.add(data)
        >>> memmap_dataset
        ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))
    """

    def __init__(self, arr_fname='arr.dat', idx_arr_fname='idx_arr.dat'):
        self.n = 0
        self.idx_arr_offset = 0
        self.arr_fname = arr_fname
        self.idx_arr_fname = idx_arr_fname
        self.mm_arr = None
        self.mm_idx_arr = None
        self.success = False

    def add(self, lstoflst):
        """
        Add list of int list to the file. This will append to the already exists data.

        Args:
            lstoflst:
                list of list of int.
        """
        if os.path.exists(self.arr_fname) and os.path.exists(self.idx_arr_fname):
            write_mode = 'ab'
            _, _, last_arr_p, last_idx_arr_p = self.load()
        else:
            write_mode = 'wb'

        def check_sucess_if_not_revert():
            # Handle the case when user interupt the writing clear the file or revert.
            if not self.success:
                if write_mode == 'ab':
                    self.truncate(last_arr_p, last_idx_arr_p)
                    self.load()
                else:
                    self.clear()
                raise InterruptedError('Writing to buffer is not yet completed. File '
                                       'have been reverted to original or cleared.')
        with callback(check_sucess_if_not_revert):
            with open(self.arr_fname, write_mode) as arr_f, \
                 open(self.idx_arr_fname, write_mode) as idx_arr_f:
                self.success = False
                if write_mode == 'wb':
                    idx_arr_f.write(struct.pack('<I', 0))
                for lst in lstoflst:
                    length = len(lst)
                    arr_f.write(struct.pack(f'<{length}I', *lst))
                    self.idx_arr_offset += len(lst)
                    self.n += 1
                    idx_arr_f.write(struct.pack('<I', self.idx_arr_offset))
            self.success = True
            self.load()

    def load(self):
        """Load from files."""
        with open(self.arr_fname, 'r') as arr_f, \
             open(self.idx_arr_fname, 'r') as idx_arr_f:
            self.mm_arr = mmap.mmap(arr_f.fileno(), length=0, access=mmap.ACCESS_READ)
            self.mm_idx_arr = mmap.mmap(idx_arr_f.fileno(), length=0, access=mmap.ACCESS_READ)
            idx_arr_f.seek(idx_arr_f.tell(), os.SEEK_END)
            self.n = (idx_arr_f.tell() - 4) // 4
            self.idx_arr_offset, = struct.unpack('<I', self.mm_idx_arr[-4:])
            arr_f.seek(arr_f.tell(), os.SEEK_END)
            return self.n, self.idx_arr_offset, arr_f.tell(), idx_arr_f.tell()

    def truncate(self, last_arr_p, last_idx_arr_p):
        """Remove trailing data from last file pointers."""
        with open(self.arr_fname, 'r+') as arr_f, \
             open(self.idx_arr_fname, 'r+') as idx_arr_f:
            arr_f.seek(last_arr_p)
            arr_f.truncate()
            idx_arr_f.seek(last_idx_arr_p)
            idx_arr_f.truncate()

    def clear(self):
        """Remove the files."""
        if os.path.exists(self.arr_fname):
            os.remove(self.arr_fname)
        if os.path.exists(self.idx_arr_fname):
            os.remove(self.idx_arr_fname)
        self.n = 0
        self.idx_arr_offset = 0
        self.mm_arr = None
        self.mm_idx_arr = None
        self.success = False

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(len(self))))
        if idx > self.n - 1:
            raise IndexError
        elif idx < 0:
            if abs(idx) > self.n:
                raise IndexError
            return self.__getitem__(self.n + idx)
        start, stop = struct.unpack('<2I', self.mm_idx_arr[4 * idx: 4 * idx + 8])
        return struct.unpack(f'<{stop - start}I', self.mm_arr[4 * start: 4 * stop])

    def __len__(self):
        return self.n

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(tuple(tuple(e) for e in self))


class MemmapLineByLineTextDataset(Dataset):
    """
    LineByLineTextDataset storing examples in persistant storage instaed of RAM.

    Args:
        tokenizer:
            tokenizer for encoding text.
        file_path:
            path to text file.
        block_size:
            size of each block or maximum sequence length.
        datasets_cache_dir:
            dir for MemmapIndexDataset files or cache.
        chunk_size:
            size of chunk for each tokenization pass.
        overwrite_cache:
            clear cache folder.

    Examples::

        >>> MemmapLineByLineTextDataset(
                tokenizer, train_file, data_args.max_seq_length,
                os.path.join(data_args.datasets_cache_dir, 'train'),
                custom_args.tokenize_chunksize, data_args.overwrite_cache
                )
    """

    def __init__(self, tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 datasets_cache_dir: str = None,
                 chunk_size: int = 2500,
                 overwrite_cache: bool = False):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        if datasets_cache_dir is None:
            datasets_cache_dir = tempfile.mkdtemp()
        else:
            found_cache = not overwrite_cache and os.path.exists(datasets_cache_dir)
            os.makedirs(datasets_cache_dir, exist_ok=True)
        self.memmap_index_dataset = MemmapIndexDataset(
            os.path.join(datasets_cache_dir, 'arr.dat'),
            os.path.join(datasets_cache_dir, 'idx_arr.dat')
            )
        if found_cache:
            logger.info("Found cached features at %s", datasets_cache_dir)
            self.memmap_index_dataset.load()
            return
        else:
            # Handle overwrite_cache case
            self.memmap_index_dataset.clear()
        logger.info("Creating features from dataset file at %s", file_path)
        lines = []
        with open(file_path, encoding="utf-8") as f:
            for line in _readline_clean_and_strip(f):
                lines.append(line)
                if len(lines) >= chunk_size:
                    batch_encoding = tokenizer(lines, add_special_tokens=True,
                                               truncation=True, max_length=block_size)
                    chunk_ex = batch_encoding["input_ids"]
                    self.memmap_index_dataset.add(chunk_ex)
                    lines = []
            if len(lines) > 0:
                batch_encoding = tokenizer(lines, add_special_tokens=True,
                                           truncation=True, max_length=block_size)
                chunk_ex = batch_encoding["input_ids"]
                self.memmap_index_dataset.add(chunk_ex)
                lines = []

    def __len__(self):
        return len(self.memmap_index_dataset)

    def __getitem__(self, i):
        return torch.tensor(self.memmap_index_dataset[i], dtype=torch.long)


class MemmapConcatFullSentenceTextDataset(Dataset):
    """
    Group multiple text into block of specific size and
    storing examples in persistant storage instaed of RAM.

    Args:
        tokenizer:
            tokenizer for encoding text.
        file_path:
            path to text file.
        block_size:
            size of each block or maximum sequence length.
        datasets_cache_dir:
            dir for MemmapIndexDataset files or cache.
        chunk_size:
            size of chunk for each tokenization pass.
        overwrite_cache:
            clear cache folder.
        progess:
            show progress.

    Examples::

        >>> MemmapConcatFullSentenceTextDataset(
                tokenizer, train_file, data_args.max_seq_length,
                os.path.join(data_args.datasets_cache_dir, 'train'),
                custom_args.tokenize_chunksize, data_args.overwrite_cache
            )
    """

    def __init__(self, tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 datasets_cache_dir: str = None,
                 chunk_size: int = 2500,
                 overwrite_cache: bool = False,
                 progress: bool = True):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        if datasets_cache_dir is None:
            datasets_cache_dir = tempfile.mkdtemp()
        else:
            found_cache = (
                not overwrite_cache and
                os.path.exists(os.path.join(datasets_cache_dir, 'arr.dat')) and
                os.path.exists(os.path.join(datasets_cache_dir, 'idx_arr.dat'))
                )
            os.makedirs(datasets_cache_dir, exist_ok=True)
        self.memmap_index_dataset = MemmapIndexDataset(
            os.path.join(datasets_cache_dir, 'arr.dat'),
            os.path.join(datasets_cache_dir, 'idx_arr.dat')
            )
        if found_cache:
            logger.info("Found cached features at %s", datasets_cache_dir)
            self.memmap_index_dataset.load()
            return
        else:
            # Handle overwrite_cache case
            self.memmap_index_dataset.clear()
        logger.info("Creating features from dataset file at %s", file_path)
        eos_token_id = tokenizer.eos_token_id
        bos_token_id = tokenizer.bos_token_id
        tokenizer_vocab = tokenizer.get_vocab()
        if '▁' in tokenizer_vocab:
            newline_token_id = tokenizer_vocab['▁']
        elif '\n' in tokenizer_vocab:
            newline_token_id = tokenizer_vocab['\n']
        usable_block_size = block_size - 2

        def add_to_block(ids, block, blocks):
            """
            Add indices to block, if the combined size of indices and block + 2 (bos, eos)
            exceed block_size, add the block to blocks if block is not empty
            then try to add indices again.
            """
            size = len(block) + len(ids)
            if block:
                size += 1
                if size > usable_block_size:
                    blocks.append([bos_token_id] + block + [eos_token_id])
                    return add_to_block(ids, [], blocks)
                else:
                    block.append(newline_token_id)
                    block.extend(ids)
                    return block
            else:
                if size > usable_block_size:
                    return []
                else:
                    return ids

        skipped_n = 0
        lines = []
        block = []
        with open(file_path, encoding="utf-8") as f:
            file_size = get_file_size(f)
            for line in _readline_clean_and_strip(f):
                lines.append(line)
                if len(lines) >= chunk_size:
                    batch_encoding = tokenizer(lines, add_special_tokens=False,
                                               truncation=True, max_length=usable_block_size + 1)
                    input_ids = batch_encoding["input_ids"]
                    blocks = []
                    for ids in input_ids:
                        block = add_to_block(ids, block, blocks)
                        if not block:
                            skipped_n += 1
                    lines = []
                    self.memmap_index_dataset.add(blocks)
                    if progress:
                        print(f'\rProcessed {f.tell() / file_size * 100:.2f}%',
                              flush=True, end=' ')
            if len(lines) > 0:
                batch_encoding = tokenizer(lines, add_special_tokens=False,
                                           truncation=True, max_length=usable_block_size + 1)
                input_ids = batch_encoding["input_ids"]
                blocks = []
                for ids in input_ids:
                    block = add_to_block(ids, block, blocks)
                    if not block:
                        skipped_n += 1
            if block:
                blocks.append(block)
                self.memmap_index_dataset.add(blocks)
            if progress:
                print(f'\rProcessed {f.tell() / file_size * 100:.2f}%',
                      flush=True, end=' ')
        print()
        logger.info(f'Skipped {skipped_n}')

    def __len__(self):
        return len(self.memmap_index_dataset)

    def __getitem__(self, i):
        return torch.tensor(self.memmap_index_dataset[i], dtype=torch.long)


class PaddedDataset(Dataset):
    """
    Pad dataset to specified block_size when __getitem__ is called.

    Args:
        dataset:
            torch dataset (or dataset that return torch tensors).
        padding_idx:
            index (interger) that use for padding.
        block_size:
            block size or max sequence length.

    Examples::

        >>> datasets = MemmapConcatFullSentenceTextDataset(..)
        >>> padded_datasets = PaddedDataset(datasets, tokenizer.pad_token_id,
                                            data_args.max_seq_length)
    """

    def __init__(self, dataset, padding_idx, block_size):
        self.dataset = dataset
        self.padding_idx = padding_idx
        self.block_size = block_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        r = self.dataset[i]
        padded = torch.tensor((), dtype=torch.long)
        padded = padded.new_full((self.block_size, ), self.padding_idx)
        padded[:r.shape[0]] = r
        return padded


class ConcatDataset(Dataset):
    """
    Concatenate multiple datasets together to make a bigger dataset.

    Args:
        datasets:
            list or tuple of torch dataset.

    Examples::

        >>> datasets1 = MemmapConcatFullSentenceTextDataset(..)
        >>> datasets2 = MemmapConcatFullSentenceTextDataset(..)
        >>> concat_datasets = ConcatDataset([datasets1, datasets2])
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.length = 0
        self.cumulative_length = [0]
        for dataset in datasets:
            dataset_len = len(dataset)
            self.length += dataset_len
            self.cumulative_length.append(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i > len(self) - 1:
            raise IndexError
        elif i < 0:
            if abs(i) > len(self):
                raise IndexError
            return self.__getitem__(len(self) + i)
        select_dataset_idx = bisect.bisect_right(self.cumulative_length, i) - 1
        idx = i - self.cumulative_length[select_dataset_idx]
        return self.datasets[select_dataset_idx][idx]
