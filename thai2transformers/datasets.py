import os
import glob
import pandas as pd
import math
from tqdm.auto import tqdm
import multiprocessing
import torch
from torch.utils.data import Dataset
import pickle
import gc
from contextlib import contextmanager
from filelock import FileLock
import logging, time
from typing import Optional
import joblib
from .utils import get_dict_val
from .conf import Task
nb_cores = multiprocessing.cpu_count()

@contextmanager
def disable_gc():
    gc.disable()
    try:
        yield
    finally:
        gc.enable()

class MLMDatasetOneFile(Dataset):
    def __init__(
        self,
        tokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,

    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = joblib.load(handle)
                logging.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logging.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    joblib.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

class MLMDataset(Dataset):
    def __init__(
        self, tokenizer, data_dir, max_length=512, binarized_path=None, ext=".txt", bs=20000,
        parallelize=True, chunksize=1_000_000, chunk_process=False
    ):
        self.fnames = glob.glob(f"{data_dir}/*{ext}")
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bs = bs
        self.features = []
        self.binarized_path = binarized_path
        self.chunksize = chunksize
        self.chunk_process = chunk_process

        if self.binarized_path is not None and self.load_binarized_features():
            assert type(self.features) == list
            if type(self.features[0]) != torch.Tensor:
                print('[INFO] Loaded data is not a list of torch.LongTensor.')
                print('[INFO] Begin converting to torch.LongTensor.\n')
                # convert on load is actually faster
                # if we try pickling torch.tensor directly
                # there are some weird too many files open bug.
                self.convert()
                print('[INFO] Done.')
        else:
            if parallelize:
                self._build_parallel(self.chunk_process)
            else:
                self._build()
                self.write_binarized_features(self.chunksize)
            self.convert()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def _build(self):
        print('[INFO] Build features (non parallel).')
        for fname in tqdm(self.fnames):
            with open(fname, "r") as f:
                df = f.readlines()
                for i in tqdm(range(math.ceil(len(df) / self.bs))):
                    texts = list(df[i * self.bs: (i + 1) * self.bs])
                    # tokenize
                    tokenized_inputs = self.tokenizer(
                        texts,
                        max_length=self.max_length,
                        truncation=True,
                        pad_to_max_length=False
                    )
                    # add to list
                    self.features += [e for e in tokenized_inputs['input_ids']]

    def _build_one(self, fname):
        features = []
        with open(fname, "r") as f:
            df = f.readlines()
            for i in tqdm(range(math.ceil(len(df) / self.bs))):
                texts = list(df[i * self.bs: (i + 1) * self.bs])
                # tokenize
                tokenized_inputs = self.tokenizer(
                    texts,
                    max_length=self.max_length,
                    truncation=True,
                    pad_to_max_length=False
                )
                # add to list
                features += [e for e in tokenized_inputs['input_ids']]
        return features

    def _build_parallel(self, chunk_process=False):
        if chunk_process:
            print('[INFO] Build features (parallel-chunk_process).')
            # Half the core count and bumping batch size this should get
            # the same performance as small batch size but lots of cores.
            # This should save memory?
            with multiprocessing.Pool(nb_cores // 2) as pool:
                # Use imap instead this allow us to iterate on the results
                # the order is still preserved. This method save memory.
                results = pool.imap(self._build_one, self.fnames)
                start = 0
                for chunk in results:
                    self.dump_chunk(chunk, start, start + len(chunk), self.binarized_path)
                    start += len(chunk)
            print('[INFO] Start groupping results.')
            self.load_binarized_features()
            print('[INFO] Done.')
        else:
            print('[INFO] Build features (parallel).')
            with multiprocessing.Pool(nb_cores // 2) as pool:
                results = pool.map(self._build_one, self.fnames)
            print('[INFO] Start groupping results.')
            self.features = [i for lst in results for i in lst]
            print('[INFO] Done.')

    def convert(self):
        # Implementing this as multiprocessing might be slower since we will need to
        # pickle the result and sent it back later.
        with disable_gc():
            for i in range(len(self.features)):
                self.features[i] = torch.tensor(self.features[i], dtype=torch.long)
                if i % 10_000_000 == 0:
                    # Manually garbage collection.
                    gc.collect()

    @staticmethod
    def dump_chunk(data, start, stop, binarized_path):
        # Implemented as static method to pickle on chunks of data instead of entire class
        dirname = os.path.dirname(binarized_path)
        basename = os.path.basename(binarized_path)
        ext = os.path.splitext(binarized_path)[-1]
        basename = basename[:-len(ext)]
        fname = os.path.join(dirname, f'{basename}_{start}_{stop}{ext}')
        print(f'[INFO] Start writing binarized data to `{fname}`.')
        with open(fname, 'wb') as fp:
            pickle.dump(data, fp)

    def write_binarized_features(self, chunksize=None, overwrite=False):
        if self.binarized_path is not None and \
                (len(self.get_bin_fnames()) == 0 or overwrite):
            if chunksize is None:
                os.makedirs(os.path.dirname(self.binarized_path), exist_ok=True)
                print(f'[INFO] Start writing binarized data to `{self.binarized_path}`.')
                with open(self.binarized_path, 'wb') as fp:
                    pickle.dump(self.features, fp)
            else:
                os.makedirs(os.path.dirname(self.binarized_path), exist_ok=True)
                chunks = [(self.features[start: start + chunksize], start, start + chunksize,
                           self.binarized_path)
                          for start in range(0, len(self.features), chunksize)]
                # Writing list of list of int out in parallel should be fine.
                # Writing list of tensors is not working.
                with multiprocessing.Pool(nb_cores) as pool:
                    pool.starmap(self.dump_chunk, chunks)

    def _load_binarized_features(self, binarized_path):
        print(f'[INFO] Start loading binarized data from `{binarized_path}`.')
        with FileLock(lock_path):
            with open(binarized_path, 'rb') as fp:
                return pickle.load(fp)

    def load_binarized_features(self):
        print('[INFO] Load binarized data')
        bin_fnames = self.get_bin_fnames()
        if self.binarized_path is not None and \
                len(bin_fnames) > 0:
            if len(bin_fnames) == 1:
                self.features = self._load_binarized_features(bin_fnames[0])
            else:
                # This can not be parallelized for now. Since the multiprocessing
                # will need to serialize back and forth, so it will be as fast as
                # single core implementation.
                with disable_gc():
                    # Disable garbage collector for speed.
                    for i, fname in enumerate(bin_fnames):
                        self.features.extend(self._load_binarized_features(fname))
                        if (i + 1) % 10 == 0:
                            # Manually collect garbage.
                            # Do we need this?
                            gc.collect()
                            break
        return len(bin_fnames) > 0

    def get_bin_fnames(self):
        dirname = os.path.dirname(self.binarized_path)
        basename = os.path.basename(self.binarized_path)
        ext = os.path.splitext(self.binarized_path)[-1]
        basename = basename[:-len(ext)]
        bin_fnames = glob.glob(self.binarized_path) + \
            glob.glob(os.path.join(dirname, f'{basename}_*_*{ext}'))
        starts = []
        for fname in bin_fnames:
            start, _ = os.path.basename(fname).split('_')[-2:]
            starts.append(int(start))
        # Sort by starting point.
        bin_fnames = list(sorted(zip(starts, bin_fnames)))
        # Get only filenames.
        bin_fnames = [e[1] for e in bin_fnames]
        return bin_fnames


class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        task=Task.MULTICLASS_CLS,
        max_length=128,
        ext=".csv",
        bs=10000,
        preprocessor=None,
        input_ids=[],
        attention_masks=[],
        labels=[],
        label_encoder=None
    ):
        self.fnames = glob.glob(f"{data_dir}/*{ext}")
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bs = bs
        self.preprocessor = preprocessor
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.task = task
        self.label_encoder = label_encoder
        self._build()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.task == Task.MULTICLASS_CLS:
            return {
                "input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_masks[i], dtype=torch.long),
                "label": torch.tensor(self.labels[i], dtype=torch.long),
            }
        elif self.task == Task.MULTILABEL_CLS:
            return {
                "input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_masks[i], dtype=torch.long),
                "label_ids": torch.tensor(self.labels[i], dtype=torch.float),
            }
        else:
            raise NotImplementedError

    @classmethod
    def from_dataset(cls,
                     task,
                     tokenizer,
                     dataset,
                     text_column_name,
                     label_column_name,
                     prepare_for_tokenization,
                     max_length=128,
                     bs=1000,
                     space_token='<_>',
                     preprocessor=None,
                     label_encoder=None):
        
        input_ids, attention_masks, labels = SequenceClassificationDataset._build_from_dataset(
                     task,
                     tokenizer,
                     dataset,
                     text_column_name,
                     label_column_name,
                     max_length=max_length,
                     bs=bs,
                     prepare_for_tokenization=prepare_for_tokenization,
                     space_token=space_token,
                     preprocessor=preprocessor,
                     label_encoder=label_encoder)

        return cls(
            tokenizer=tokenizer,
            data_dir=None,
            max_length=max_length,
            bs=bs,
            input_ids=input_ids,
            attention_masks=attention_masks,
            labels=labels,
            task=task,
            preprocessor=preprocessor,
            label_encoder=label_encoder
        )
    
    @staticmethod
    def _build_from_dataset(task, tokenizer, dataset,
                            text_column_name, label_column_name,
                            space_token, max_length, bs,
                            prepare_for_tokenization,
                            label_encoder,
                            preprocessor=None):
        texts = get_dict_val(dataset, text_column_name)
        if task == Task.MULTICLASS_CLS:
            labels = get_dict_val(dataset, label_column_name)

            if label_encoder != None:
                labels = label_encoder.transform(labels)

        elif task == Task.MULTILABEL_CLS:
            _labels = []
            for i, name in enumerate(label_column_name):
                # print(name)
                _labels.append(get_dict_val(dataset, name))
            labels = list(zip(*_labels))
        else:
            raise NotImplementedError
        
        input_ids = []
        attention_masks = []

        if preprocessor != None:
            print('[INFO] Apply preprocessor to texts.')
            texts = list(map(preprocessor, texts))

        for i in tqdm(range(math.ceil(len(texts) / bs))):

            batched_texts = texts[i * bs: (i+1) * bs]

            tokenized_inputs = tokenizer(
                batched_texts,
                max_length=max_length,
                truncation=True,
                # padding='max_length'            
            )
            # add to list
            input_ids += tokenized_inputs["input_ids"]
            attention_masks += tokenized_inputs["attention_mask"]
        return input_ids, attention_masks, labels

    def _build(self):
        for fname in tqdm(self.fnames):
            df = pd.read_csv(fname)
            for i in tqdm(range(math.ceil(df.shape[0] / self.bs))):
                if self.preprocessor:
                    texts = list(
                        df.iloc[i * self.bs: (i + 1) * self.bs, 0].map(
                            self.preprocessor
                        )
                    )
                else:
                    texts = list(df.iloc[i * self.bs: (i + 1) * self.bs, 0])

                # tokenize
                tokenized_inputs = self.tokenizer(
                    texts,
                    max_length=self.max_length,
                    truncation=True,
                    # padding='max_length'
                )

                # add to list
                self.input_ids += tokenized_inputs["input_ids"]
                self.attention_masks += tokenized_inputs["attention_mask"]
                self.labels += list(df.iloc[i * self.bs: (i + 1) * self.bs, 1])


class TokenClassificationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length=128,
        ext=".csv",
        label_pad_token="0",
        label_first_subword=False,
    ):
        self.fnames = glob.glob(f"{data_dir}/*{ext}")
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.label_pad_token = label_pad_token
        self.label_first_subword = label_first_subword
        self.features = []
        self.word2sub = []

        self._build()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        feature = self.features[i]
        return {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
            "label": torch.tensor(feature["label"], dtype=torch.long),
            "word_ids": torch.tensor(feature["word_ids"], dtype=torch.long),
        }

    def normalize(self, src_):
        sub_1 = [
            self.tokenizer.decode(i)
            for i in self.tokenizer.encode(
                src_,
                add_special_tokens=False,
                truncation=False,
                pad_to_max_length=False,
            )
        ]

        # keep subwords only up to max length plus 3 special tokens <s>, space, </s>
        sub_2 = []
        cnt = 0
        for s in sub_1:
            if cnt > self.max_length - 3:
                break
            sub_2.append(s)
            if s != "|":
                cnt += 1

        res = "".join(sub_2)
        res = res[:-1] if res[-1] == "|" else res  # remove last space
        res = res[1:] if res[0] == "|" else res  # remove first space
        res = [" " if i == "" else i for i in res.split("|")]
        return res

    def _add_special_tokens(self, src_, lbl_):
        src = [self.tokenizer.bos_token, " "] + \
            src_ + [self.tokenizer.eos_token]
        lbl = (
            [self.label_pad_token, self.label_pad_token] +
            lbl_ + [self.label_pad_token]
        )
        txt = "".join(src)
        return src, lbl, txt

    def _get_subword_df(self, src, lbl, sub):
        # construct character df; label characters in words
        word_df = []
        for w_i in range(len(src)):
            for c in src[w_i]:
                word_df.append((c, w_i, int(lbl[w_i])))
        word_df = pd.DataFrame(word_df)
        word_df.columns = ["char", "word_i", "label"]

        # label characters in subwords
        sub_i = []
        for sw_i in range(len(sub)):
            for c in sub[sw_i]:
                sub_i.append(sw_i)
        word_df["sub_i"] = sub_i

        # map subwords to labels
        subword_df = (
            word_df.groupby(["sub_i"]).agg(
                {"label": max, "word_i": max}).reset_index()
        )

        # label for only the first subword of token
        if self.label_first_subword:
            subword_df["rnk"] = subword_df.groupby("word_i").cumcount()
            subword_df.loc[subword_df.rnk > 0, "label"] = 0

        return subword_df[["sub_i", "word_i", "label"]]

    def _build_one(self, src_, lbl_):
        # encode-decode to make sure characters are the same as in tokenizer vocab; e.g. เ เ and แ
        src_ = self.normalize(src_)
        lbl_ = lbl_.split("|")[: len(src_)]

        # # add special tokens
        src, lbl, txt = self._add_special_tokens(src_, lbl_)

        # tokenize
        tokenized_inputs = self.tokenizer(txt, add_special_tokens=False,)
        ids, attn = tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]

        # pad subwords
        to_pad = self.max_length - len(ids)
        ids += [self.tokenizer.pad_token_id] * to_pad
        attn += [0] * to_pad
        sub = [i.replace("▁", " ")
               for i in self.tokenizer.convert_ids_to_tokens(ids)]
        sub_txt = "".join(sub)

        # pad labels and words
        lbl += [self.label_pad_token] * to_pad
        src += [self.tokenizer.pad_token] * to_pad
        txt += self.tokenizer.pad_token * to_pad

        # checks
        assert len(attn) == len(ids)  # attention mask matches ids
        assert len(lbl) == len(src)  # labels match ids
        assert len(ids) == len(sub)  # ids match subwords
        assert len(txt) == len(
            sub_txt
        )  # reconstructed source text matches reconstruct subword text

        # get subword_df for metrics and matching subwords to labels
        subword_df = self._get_subword_df(src, lbl, sub)

        return {
            "input_ids": ids,
            "attention_mask": attn,
            "label": list(subword_df.label),
            "word_ids": list(subword_df.word_i),
        }

    def _build(self):
        input_ids = []
        attention_masks = []
        labels = []
        for fname in tqdm(self.fnames):
            df = pd.read_csv(fname)
            for i, row in tqdm(df.iterrows()):
                try:
                    feature = self._build_one(row[0], row[1])
                    self.features.append(feature)
                except:
                    print(row[0], row[1])