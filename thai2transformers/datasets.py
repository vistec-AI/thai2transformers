import glob
import pandas as pd
import math
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from transformers.data.processors.utils import InputFeatures


class MLMDataset(Dataset):
    def __init__(
        self, tokenizer, data_dir, max_length=512, ext=".txt", bs=10000,
    ):
        self.fnames = glob.glob(f"{data_dir}/*{ext}")
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bs = bs
        self.features = []
        self._build()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return torch.tensor(self.features[i], dtype=torch.long)

    def _build(self):
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


class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length=128,
        ext=".csv",
        bs=10000,
        preprocessor=None,
        huggingface_format=False,
    ):
        self.fnames = glob.glob(f"{data_dir}/*{ext}")
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bs = bs
        self.preprocessor = preprocessor
        self.features = []
        if huggingface_format:
            self._build_hf()
        else:
            self._build()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def _build(self):
        input_ids = []
        attention_masks = []
        labels = []
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
                self.features
                input_ids += tokenized_inputs["input_ids"]
                attention_masks += tokenized_inputs["attention_mask"]
                labels += list(df.iloc[i * self.bs : (i + 1) * self.bs, 1])

        for i in tqdm(range(len(input_ids))):
            feature = {
                "input_ids": torch.tensor(input_ids[i], dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks[i], dtype=torch.long),
                "label": torch.tensor(labels[i], dtype=torch.long),
            }
            self.features.append(feature)

    def _build_hf(self):
        """
        Experimental; mimic huggingface datasets
        """
        input_ids = []
        attention_masks = []
        labels = []
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
                self.features
                input_ids += tokenized_inputs["input_ids"]
                attention_masks += tokenized_inputs["attention_mask"]
                labels += list(df.iloc[i * self.bs : (i + 1) * self.bs, 1])

        for i in tqdm(range(len(input_ids))):
            feature = InputFeatures(
                input_ids=input_ids[i],
                attention_mask=attention_masks[i],
                label=labels[i],
            )
            self.features.append(feature)


class TokenClassificationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length=512,
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

        self._build()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def _build_one(self, src_, lbl_):
        # encode-decode to make sure characters are the same as in tokenizer vocab; e.g. เ เ and แ
        src_ = "".join(
            [
                self.tokenizer.decode(i)
                for i in self.tokenizer.encode(src_, add_special_tokens=False)
            ]
        )
        src_ = src_.split("|")
        lbl_ = lbl_.split("|")
        src = [self.tokenizer.bos_token, " "] + src_ + [self.tokenizer.eos_token]
        txt = "".join(src)
        lbl = (
            [self.label_pad_token, self.label_pad_token] + lbl_ + [self.label_pad_token]
        )

        # totkenize
        tokenized_inputs = self.tokenizer(txt, add_special_tokens=False,)

        # pad subwords
        ids = tokenized_inputs["input_ids"]
        to_pad = self.max_length - len(ids)
        ids += [self.tokenizer.pad_token_id] * to_pad
        attn = tokenized_inputs["attention_mask"]
        attn += [0] * to_pad
        sub = [i.replace("▁", " ") for i in self.tokenizer.convert_ids_to_tokens(ids)]
        sub_txt = "".join(sub)

        # pad labels and words
        lbl += [label_pad_token] * to_pad
        src += [self.tokenizer.pad_token] * to_pad
        txt += self.tokenizer.pad_token * to_pad

        assert len(attn) == len(ids)  # attention mask matches ids
        assert len(lbl) == len(src)  # labels match ids
        assert len(ids) == len(sub)  # ids match subwords
        assert len(txt) == len(
            sub_txt
        )  # reconstructed source text matches reconstruct subword text

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

        # map subwords to words
        word_agg = word_df.groupby("word_i").sub_i.max().reset_index()
        word2sub = {w: s for w, s in zip(word_agg["word_i"], word_agg["sub_i"])}

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "label": torch.tensor(subword_df.label, dtype=torch.long),
            "word2sub": word2sub,
        }

    def _build(self):
        input_ids = []
        attention_masks = []
        labels = []
        for fname in tqdm(self.fnames):
            df = pd.read_csv(fname)
            for i, row in tqdm(df.iterrows()):
                feature = self._build_one(row[0], row[1])
                self.features.append(feature)