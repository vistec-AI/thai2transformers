#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:39:32 2020

@author: zo
"""

import warnings
import os
import multiprocessing


class REQUIRED:
    """Dummy class for checking required arguments"""
    pass


def get_field(cls, field):
    return cls.__dataclass_fields__[field]


def check_depreciated(args, warn_list):
    """
    Check if arguments in warn_list appear in args. Raise warning when found.
    """
    for arg, exp_v, warn in warn_list:
        if getattr(args, arg) != exp_v:
            warnings.warn(str(warn.args), warn.__class__)


def check_required(args):
    required = []
    kwargs = vars(args)
    for k, v in kwargs.items():
        if v == REQUIRED:
            required.append(k)
    if required:
        raise ValueError(f'{tuple(required)} are required.')


def get_file_size(f):
    old_file_position = f.tell()
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(old_file_position, os.SEEK_SET)
    return size


def readline_clean_and_strip(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in _readline_clean_and_strip(f):
            yield line


def _readline_clean_and_strip(f):
    while True:
        line = f.readline()
        if line:
            line = line.strip()
            if len(line) > 0 and not line.isspace():
                yield line
        else:
            break


def multi_imap(data, chunk_size, f,
               n_cores, process=True):
    chunks = [data[i: i + chunk_size]
              for i in range(0, len(data), chunk_size)]
    with multiprocessing.Pool(n_cores) as pool:
        results = []
        for r in pool.imap(f, chunks):
            results.extend(r)
            if process:
                print(f'\rProcessed {len(results) / len(data) * 100:.2f}%',
                      flush=True, end=' ')
    return results
