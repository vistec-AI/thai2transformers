import glob
import pandas as pd
import math
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from transformers.data.processors.utils import InputFeatures

class MLMDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length=512,
        ext='.txt',
        bs = 10000,
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
        input_ids = []
        for fname in tqdm(self.fnames):
            with open(fname,'r') as f:
                df = f.readlines()
                for i in tqdm(range(math.ceil(len(df)/self.bs))):
                    texts = list(df[i*self.bs:(i+1)*self.bs])
                    #tokenize
                    tokenized_inputs = self.tokenizer(
                        texts,
                        max_length=self.max_length,
                        truncation=True,
                        pad_to_max_length=False,
                    )
                    #add to list
                    input_ids+=tokenized_inputs['input_ids']

        for i in tqdm(range(len(input_ids))):
            self.features.append(input_ids[i])

class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_length=128,
        ext=".csv",
        bs = 10000,
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
            for i in tqdm(range(math.ceil(df.shape[0]/self.bs))):
                if self.preprocessor:
                    texts = list(df.iloc[i*self.bs:(i+1)*self.bs,0].map(self.preprocessor))
                else:
                    texts = list(df.iloc[i*self.bs:(i+1)*self.bs,0])

                #tokenize
                tokenized_inputs = self.tokenizer(
                    texts,
                    max_length=self.max_length,
                    truncation=True,
                    pad_to_max_length=True,
                )
  
                #add to list
                self.features
                input_ids+=tokenized_inputs['input_ids']
                attention_masks+=tokenized_inputs['attention_mask']
                labels+=list(df.iloc[i*self.bs:(i+1)*self.bs,1])

        for i in tqdm(range(len(input_ids))):
            feature = {'input_ids': torch.tensor(input_ids[i], dtype=torch.long), 
                        'attention_mask': torch.tensor(attention_masks[i], dtype=torch.long),
                        'label': torch.tensor(labels[i], dtype=torch.long),
                      }
            self.features.append(feature)

    def _build_hf(self):
        '''
        Experimental; mimic huggingface datasets
        '''
        input_ids = []
        attention_masks = []
        labels = []
        for fname in tqdm(self.fnames):
            df = pd.read_csv(fname)
            for i in tqdm(range(math.ceil(df.shape[0]/self.bs))):
                if self.preprocessor:
                    texts = list(df.iloc[i*self.bs:(i+1)*self.bs,0].map(self.preprocessor))
                else:
                    texts = list(df.iloc[i*self.bs:(i+1)*self.bs,0])

                #tokenize
                tokenized_inputs = self.tokenizer(
                    texts,
                    max_length=self.max_length,
                    truncation=True,
                    pad_to_max_length=True,
                )
  
                #add to list
                self.features
                input_ids+=tokenized_inputs['input_ids']
                attention_masks+=tokenized_inputs['attention_mask']
                labels+=list(df.iloc[i*self.bs:(i+1)*self.bs,1])

        for i in tqdm(range(len(input_ids))):
            feature = InputFeatures(input_ids=input_ids[i], 
                                    attention_mask=attention_masks[i],
                                    label=labels[i],)
            self.features.append(feature)
