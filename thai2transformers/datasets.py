import os
import glob
import pandas as pd
import math
from tqdm.auto import tqdm
import multiprocessing
nb_cores = multiprocessing.cpu_count()
import torch
from torch.utils.data import Dataset
from transformers.data.processors.utils import InputFeatures
import pickle

class MLMDataset(Dataset):
    def __init__(
        self, tokenizer, data_dir, max_length=512, binarized_path=None,ext=".txt", bs=10000,
        parallelize=True
    ):
        self.fnames = glob.glob(f"{data_dir}/*{ext}")
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bs = bs
        self.features = []
        if parallelize:
            self._build_parallel()
        else:
            self._build()
            

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return torch.tensor(self.features[i], dtype=torch.long)

    def _build(self):
        if self.binarized_path != None and os.path.exists(self.binarized_path):
            print('The binarized directory exists, load the binarized data.')
            self.features = pickle.load(open(self.binarized_dir, 'rb'))
            return

        for fname in tqdm(self.fnames):
            with open(fname, "r") as f:
                df = f.readlines()
                for i in tqdm(range(math.ceil(len(df) / self.bs))):
                    texts = list(df[i * self.bs : (i + 1) * self.bs])
                    # tokenize
                    tokenized_inputs = self.tokenizer(
                        texts,
                        max_length=self.max_length,
                        truncation=True,
                        pad_to_max_length=False,
                    )
                    # add to list
                    self.features += tokenized_inputs["input_ids"]
        
        with open(self.binarized_path, 'wb') as fp:
            pickle.dump(self.features, fp)

    def _build_one(self, fname):
        features = []
        with open(fname, "r") as f:
            df = f.readlines()
            for i in tqdm(range(math.ceil(len(df) / self.bs))):
                texts = list(df[i * self.bs : (i + 1) * self.bs])
                # tokenize
                tokenized_inputs = self.tokenizer(
                    texts,
                    max_length=self.max_length,
                    truncation=True,
                    pad_to_max_length=False,
                )
                # add to list
                features += tokenized_inputs["input_ids"]
        return features
                
    def _build_parallel(self):
        with multiprocessing.Pool(nb_cores) as pool:
            results = pool.map(self._build_one, self.fnames)
        self.features = [i for l in results for i in l]


class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length=128,
        ext=".csv",
        bs=10000,
        preprocessor=None,
    ):
        self.fnames = glob.glob(f"{data_dir}/*{ext}")
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bs = bs
        self.preprocessor = preprocessor
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        self._build()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[i], dtype=torch.long),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }

    def _build(self):
        for fname in tqdm(self.fnames):
            df = pd.read_csv(fname)
            for i in tqdm(range(math.ceil(df.shape[0] / self.bs))):
                if self.preprocessor:
                    texts = list(
                        df.iloc[i * self.bs : (i + 1) * self.bs, 0].map(
                            self.preprocessor
                        )
                    )
                else:
                    texts = list(df.iloc[i * self.bs : (i + 1) * self.bs, 0])

                # tokenize
                tokenized_inputs = self.tokenizer(
                    texts,
                    max_length=self.max_length,
                    truncation=True,
                    pad_to_max_length=True,
                )

                # add to list
                self.input_ids += tokenized_inputs["input_ids"]
                self.attention_masks += tokenized_inputs["attention_mask"]
                self.labels += list(df.iloc[i * self.bs : (i + 1) * self.bs, 1])


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
        src = [self.tokenizer.bos_token, " "] + src_ + [self.tokenizer.eos_token]
        lbl = (
            [self.label_pad_token, self.label_pad_token] + lbl_ + [self.label_pad_token]
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
            word_df.groupby(["sub_i"]).agg({"label": max, "word_i": max}).reset_index()
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
        sub = [i.replace("▁", " ") for i in self.tokenizer.convert_ids_to_tokens(ids)]
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
