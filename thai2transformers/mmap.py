import numpy as np
import random
import os
import time
import mmap
import struct
import torch
from contextlib import contextmanager
from typing import Iterable

@contextmanager
def timeit(ops):
    try:
        start = time.perf_counter()
        yield
    finally:
        end = time.perf_counter()
        print(f'[{ops}] Perf: ', end - start)


@contextmanager
def callback(func):
    try:
        yield
    finally:
        func()


class MemmapIndexDataset:
    """Memory-mapped backed dataset for a list of lists of integers."""

    def __init__(self, arr_fname='arr.dat', idx_arr_fname='idx_arr.dat'):
        self.n = 0
        self.idx_arr_offset = 0
        self.arr_fname = arr_fname
        self.idx_arr_fname = idx_arr_fname
        self.mm_arr = None
        self.mm_idx_arr = None
        self.success = False

    def add(self, lstoflst: Iterable[Iterable[int]]):
        """Add list of int list to the file. This will append to the already exists data."""
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


class TorchMemmapIndexDataset(torch.utils.data.Dataset):

    def __init__(self, memmap_index_dataset):
        self.memmap_index_dataset = memmap_index_dataset

    def __len__(self):
        return len(self.memmap_index_dataset)

    def __getitem__(self, idx):
        return torch.LongTensor(self.memmap_index_dataset[idx])
