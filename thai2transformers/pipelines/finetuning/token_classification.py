import os
import sys
from typing import List, Union, Dict, Callable, Optional
from functools import reduce, lru_cache

from datasets import load_dataset, Dataset
from transformers import (
    TrainingArguments,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy

from .base import BaseFinetuningPipeline
from thai2transformers.conf import Task
from thai2transformers.data_collator import DataCollatorForTokenClassification
from thai2transformers.finetuner import BaseFinetuner, TokenClassificationFinetuner
from thai2transformers.utils import get_dict_val


class TokenClassificationFinetuningPipeline(BaseFinetuningPipeline):

    def __init__(self,
                 task: Union[str, Task],
                 train_dataset: Dataset = None,
                 val_dataset: Dataset = None,
                 test_dataset: Dataset = None,
                 finetuner: BaseFinetuner = None):
        if isinstance(task, Task):
            self.task = task.value
        elif isinstance(task, str):
            self.task = task
        self._dataset = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.data_collator = None
        self.tokenizer = None
        self.num_labels = None
        self.id2label = None
        self.finetuner = TokenClassificationFinetuner()

    def load_dataset(self,
                     dataset_name_or_path: Union[str, os.PathLike],
                     text_column_name: str,
                     label_column_name: Union[str, List[str]],
                     task: Union[str, Task]= None,
                     train_dataset_name: str = 'train'):
        if isinstance(task, Task):
            self.task = task.value
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        if isinstance(dataset_name_or_path, str):
            self._dataset = load_dataset(dataset_name_or_path)

        if self.task not in [Task.CHUNK_LEVEL_CLS.value, Task.TOKEN_LEVEL_CLS.value]:
            raise ValueError()

        self.id2label = {i: name for i, name in 
                         enumerate(self._dataset[train_dataset_name].features[self.label_column_name].feature.names)}
        self.num_labels = self._dataset[train_dataset_name].features[self.label_column_name].feature.num_classes


    def load_tokenizer(self, tokenizer_cls: Union[str, PreTrainedTokenizer], name_or_path):

        self.finetuner.load_pretrained_tokenizer(
                        tokenizer_cls=tokenizer_cls,
                        name_or_path=name_or_path)
        self.tokenizer = self.finetuner.tokenizer    
   
    def load_model(self, name_or_path, num_labels:int = None):

        if self.num_labels == None and num_labels != None:
            self.num_labels = num_labels

        self.finetuner.load_pretrained_model(
                        task=self.task,
                        name_or_path=name_or_path,
                        num_labels=self.num_labels,
                        id2label=self.id2label)
    @staticmethod
    def _preprocess(examples,
                    tokenizer,
                    text_column_name,
                    label_column_name,
                    space_token='<_>',
                    max_length=256,
                    lowercase=True):

        def pre_tokenize(token: str, space_token='<_>'):
            token = token.replace(' ', space_token)
            return token

        @lru_cache(maxsize=None)
        def cached_tokenize(token: str, space_token='<_>', lowercase=True):
            if lowercase:
                token = token.lower()
            token = pre_tokenize(token, space_token=space_token)
            ids = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(token))
            return ids

        tokens = []
        labels = []
        old_positions = []
        for example_tokens, example_labels in zip(examples[text_column_name], examples[label_column_name]):
            new_example_tokens = []
            new_example_labels = []
            old_position = []
            for i, (token, label) in enumerate(zip(example_tokens, example_labels)):
                # tokenize each already pretokenized tokens with our own tokenizer.
                toks = cached_tokenize(token, space_token=space_token, lowercase=lowercase)
                n_toks = len(toks)
                new_example_tokens.extend(toks)
                # expand label to cover all tokens that get split in a pretokenized token
                new_example_labels.extend([label] * n_toks)
                # kept track of old position
                old_position.extend([i] * n_toks)
            tokens.append(new_example_tokens)
            labels.append(new_example_labels)
            old_positions.append(old_position)
        tokenized_inputs = tokenizer._batch_prepare_for_model(
            [(e, None) for e in tokens],
            truncation_strategy=TruncationStrategy.LONGEST_FIRST,
            add_special_tokens=True, max_length=max_length)
        # in case of needed truncation we need to chop off some of the labels manually
        max_length = max(len(e) for e in tokenized_inputs['input_ids'])
        # add -100 to first and last token which is special tokens for <s> and </s>
        # -100 is a convention for padding in higgingface transformer lib
        # and calculating loss should skip this
        tokenized_inputs['old_positions'] = [[-100] + e[:max_length - 2] + [-100]
                                            for e in old_positions]
        tokenized_inputs['labels'] = [[-100] + e[:max_length - 2] + [-100]
                                    for e in labels]
        return tokenized_inputs

    def process_dataset(self,
                        tokenizer: PreTrainedTokenizer = None,
                        lowercase:bool = False,
                        max_length=128,
                        space_token='<_>',
                        train_dataset_name='train',
                        val_dataset_name='val',
                        test_dataset_name='test',
                        num_train_examples: int=None,
                        num_val_examples: int=None,
                        num_test_examples: int=None):
 
        if self.tokenizer == None and tokenizer != None and isinstance(tokenizer, PreTrainedTokenizer):
            self.tokenizer = tokenizer
        elif self.tokenizer == None and tokenizer == None:
            raise AssertionError('A Tokenizer has never been specified')

     
        if num_train_examples != None and train_dataset_name in self._dataset.keys():
            self._dataset[train_dataset_name] = self._dataset[train_dataset_name][:num_train_examples]
        if num_val_examples != None and val_dataset_name in self._dataset.keys():
            self._dataset[val_dataset_name] = self._dataset[val_dataset_name][:num_val_examples]
        if num_test_examples != None and test_dataset_name in self._dataset.keys():
            self._dataset[test_dataset_name] = self._dataset[test_dataset_name][:num_test_examples]

      
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        if train_dataset_name in self._dataset.keys():
          
            self.train_dataset = Dataset.from_dict(TokenClassificationFinetuningPipeline._preprocess(self._dataset[train_dataset_name],
                                                    tokenizer=self.tokenizer,
                                                    text_column_name=self.text_column_name,
                                                    label_column_name=self.label_column_name,
                                                    space_token=space_token))

        if val_dataset_name in self._dataset.keys():

            self.val_dataset = Dataset.from_dict(TokenClassificationFinetuningPipeline._preprocess(self._dataset[val_dataset_name],
                                                    tokenizer=self.tokenizer,
                                                    text_column_name=self.text_column_name,
                                                    label_column_name=self.label_column_name,
                                                    space_token=space_token))
            self.val_dataset = Dataset.from_dict(self.data_collator(self.val_dataset))


        if test_dataset_name in self._dataset.keys():
            self.test_dataset = Dataset.from_dict(TokenClassificationFinetuningPipeline._preprocess(self._dataset[test_dataset_name],
                                                    tokenizer=self.tokenizer,
                                                    text_column_name=self.text_column_name,
                                                    label_column_name=self.label_column_name,
                                                    space_token=space_token))
            self.test_dataset = Dataset.from_dict(self.data_collator(self.test_dataset))


    def finetune(self, output_dir:str, eval_on_test_set:bool = False, **kwargs):

        training_args = TrainingArguments(output_dir=output_dir,
                                          **kwargs)

        if eval_on_test_set and self.test_dataset:
            return self.finetuner.finetune(training_args,
                                train_dataset=self.train_dataset,
                                val_dataset=self.val_dataset,
                                test_dataset=self.test_dataset)
        elif eval_on_test_set and self.test_dataset == None:
            raise AssertionError('test_dataset is not specified, while argument eval_on_test_set is True')
        
        if not eval_on_test_set:
            self.finetuner.finetune(training_args,
                                train_dataset=self.train_dataset,
                                val_dataset=self.val_dataset,
                                test_dataset=None)
