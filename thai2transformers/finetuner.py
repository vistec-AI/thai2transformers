import os
from typing import List, Dict, Union, Optional, Callable
from functools import partial

import torch
import pprint
from datasets import ( Dataset )
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.tokenization_utils import (
    PreTrainedTokenizer,
)
from .auto import (
    AutoModelForMultiLabelSequenceClassification,
)
from .conf import (
    Task,
)
from .datasets import (
    SequenceClassificationDataset,
    TokenClassificationDataset,
)
from .data_collator import (
    DataCollatorForTokenClassification,
)
from .metrics import (
    classification_metrics,
    multilabel_classification_metrics,
    token_classification_metrics,
    chunk_level_classification_metrics,
    token_level_classification_metrics,
)

AIRESEARCH_MODEL_PREFIX = 'airesearch/wangchanberta'
AIRESEARCH_MODEL_NAME = {
    f'{AIRESEARCH_MODEL_PREFIX}-base-att-spm-uncased': {
        'space_token': '<_>'
    }
}

FINETUNE_METRIC_MAPPING = {
    'SequenceClassification': { 
        Task.MULTICLASS_CLS.value: classification_metrics,
        Task.MULTILABEL_CLS.value: multilabel_classification_metrics,
    },
    'TokenClassification': { 
        Task.CHUNK_LEVEL_CLS.value: chunk_level_classification_metrics,
        Task.TOKEN_LEVEL_CLS.value: token_level_classification_metrics,
    },
}

FINETUNE_MODEL_MAPPING = {
    'SequenceClassification': { 
        Task.MULTICLASS_CLS.value: AutoModelForSequenceClassification,
        Task.MULTILABEL_CLS.value: AutoModelForMultiLabelSequenceClassification,
    },
    'TokenClassification': { 
        Task.CHUNK_LEVEL_CLS.value: AutoModelForTokenClassification,
        Task.TOKEN_LEVEL_CLS.value: AutoModelForTokenClassification,
    },
}


class BaseFinetuner:

    def load_pretrained_tokenizer(self):
        pass

    def load_pretrained_model(self):
        pass

    def finetune(self):
        pass


class SequenceClassificationFinetuner(BaseFinetuner):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer = None,
                 config=None,
                 task: Union[str, Task] = None,
                 num_labels: int = None,
                 metric: Union[str, Callable, Task] = None):

        self.tokenizer = tokenizer
        self.config = config
        self.task = task.value if isinstance(task, Task) else task
        self.num_labels = num_labels
        self.metric = metric
        self.training_args = None
        self.trainer = None

    def load_pretrained_tokenizer(self,
                                  tokenizer_cls: PreTrainedTokenizer,
                                  name_or_path: Union[str, os.PathLike],
                                  revision: str = None):
        """
        Load a pretrained tokenizer to the finetuner instance
        """

        self.tokenizer = tokenizer_cls.from_pretrained(name_or_path,
                                                       revision=revision)

        if tokenizer_cls.__name__ == 'CamembertTokenizer':

            if name_or_path in AIRESEARCH_MODEL_NAME.keys():

                self.tokenizer.additional_special_tokens = [
                    '<s>NOTUSED',
                    '</s>NOTUSED',
                    AIRESEARCH_MODEL_NAME[name_or_path]['space_token']
                ]

    def load_pretrained_model(self,
                              task: Union[str, Task],
                              name_or_path: Union[str, os.PathLike],
                              revision: str = None,
                              num_labels: int = None):
        """
        Load a pretrained model to the finetuner instance and modify classification head
        and metric according to the specified `task` in the method argument.

        Arguments:
            task: Union[str, Task],
        """
        if num_labels == None:
            self.config = AutoConfig.from_pretrained(name_or_path)
            self.num_labels = self.config.num_labels
        else:
            self.config = AutoConfig.from_pretrained(
                name_or_path,
                num_labels=num_labels
            )
            self.num_labels = num_labels

        if isinstance(task, Task):
            task = task.value

        self.task = task

        if task not in FINETUNE_MODEL_MAPPING['SequenceClassification'].keys():
            raise NotImplementedError(
                f"The task specified `{task}` is incorrect or not available for {self.__class__.__name__}")

        self.model = FINETUNE_MODEL_MAPPING['SequenceClassification'][task] \
                        .from_pretrained(name_or_path,
                                         config=self.config,
                                         revision=revision)
        self.metric = FINETUNE_METRIC_MAPPING['SequenceClassification'][task]

    def _init_trainer(self,
                      training_args,
                      train_dataset: SequenceClassificationDataset,
                      val_dataset: SequenceClassificationDataset = None):

        self.training_args = training_args
        data_collator = DataCollatorWithPadding(self.tokenizer,
                                                padding=True,
                                                pad_to_multiple_of=8 if training_args.fp16 else None)
        if self.task == Task.MULTILABEL_CLS.value:
            metric = partial(self.metric, n_labels=self.num_labels)
        else:
            metric = self.metric

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            compute_metrics=metric,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

    def finetune(self,
                 training_args,
                 train_dataset: SequenceClassificationDataset,
                 val_dataset: SequenceClassificationDataset = None,
                 test_dataset: SequenceClassificationDataset = None):

        self._init_trainer(
            training_args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        self.trainer.train()
        self.trainer.save_model(
            os.path.join(training_args.output_dir,
                         'checkpoint-final')
        )

        if test_dataset != None:

            _, label_ids, result = self.trainer.predict(
                test_dataset=test_dataset)

            print(f'Evaluation on test set')

            for key, value in result.items():
                print(f'{key} : {value:.4f}')

            return result


class TokenClassificationFinetuner(BaseFinetuner):

    def __init__(self, num_labels:int = None):
        self.tokenizer = None
        self.num_labels = num_labels
        self.id2label = None
        self.metric = None
        self.training_args = None

    def load_pretrained_tokenizer(self,
                                  tokenizer_cls: PreTrainedTokenizer,
                                  name_or_path: Union[str, os.PathLike],
                                  revision: str = None):
        """
        Load a pretrained tokenizer to the finetuner instance
        """

        self.tokenizer = tokenizer_cls.from_pretrained(name_or_path,
                                                       revision=revision)

        if tokenizer_cls.__name__ == 'CamembertTokenizer':

            if name_or_path in AIRESEARCH_MODEL_NAME.keys():

                self.tokenizer.additional_special_tokens = [
                    '<s>NOTUSED',
                    '</s>NOTUSED',
                    AIRESEARCH_MODEL_NAME[name_or_path]['space_token']
                ]

    def load_pretrained_model(self,
                              name_or_path: Union[str, os.PathLike],
                              task: Union[str, Task],
                              id2label: Dict[int,str],
                              revision: str = None,
                              num_labels: int = None,
                              ):
        """
        Load a pretrained model to the finetuner instance and modify classification head
        and metric according to the specified `task` in the method argument.

        Arguments:
            task: Union[str, Task],
        """
        if num_labels == None:
            self.config = AutoConfig.from_pretrained(name_or_path)
            self.num_labels = self.config.num_labels
        else:
            self.config = AutoConfig.from_pretrained(
                name_or_path,
                num_labels=num_labels
            )
            self.num_labels = num_labels

        if isinstance(task, Task):
            task = task.value

        self.task = task

        if task not in FINETUNE_MODEL_MAPPING['TokenClassification'].keys():
            raise NotImplementedError(
                f"The task specified `{task}` is incorrect or not available for {self.__class__.__name__}")

        self.model = FINETUNE_MODEL_MAPPING['TokenClassification'][task] \
                            .from_pretrained(name_or_path,
                                             config=self.config,
                                             revision=revision)
        self.metric = partial(token_classification_metrics,
                        task=self.task,
                        id2label=id2label,
                      )
        self.id2label = id2label

    def _init_trainer(self,
                      training_args,
                      train_dataset: Dataset,
                      val_dataset: Dataset = None):

        self.training_args = training_args
        data_collator = DataCollatorForTokenClassification(self.tokenizer)


        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            compute_metrics=self.metric,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

    @staticmethod
    def _get_batch(obj, batch_size):
        i = 0
        r = obj[i * batch_size: i * batch_size + batch_size]
        yield r
        i += 1
        while i * batch_size < len(obj):
            r = obj[i * batch_size: i * batch_size + batch_size]
            yield r
            i += 1

    def _evaluate(self, dataset, training_args=None):

        # override self.training_args 
        if self.training_args == None and training_args != None:
            self.training_args = training_args

        device = self.model.device
        
        agg_chunk_preds = []
        agg_chunk_labels = []

        for step, batch in enumerate(TokenClassificationFinetuner._get_batch(dataset, self.training_args.per_device_eval_batch_size)):
            labels = batch['labels']
            old_positions = batch['old_positions']
            dont_include = ['labels', 'old_positions']
            batch = {k: torch.tensor(v, dtype=torch.int64).to(device) for k, v in batch.items()
                    if k not in dont_include}

            preds, = self.model(**batch)
            preds = preds.argmax(2)
            preds = preds.tolist()

            use_idxs = [[i for i, e in enumerate(label) if e != -100]
                        for label in labels]
            true_preds = [[preds[j][i] for i in use_idx]
                        for j, use_idx in enumerate(use_idxs)]
            true_labels = [[labels[j][i] for i in use_idx]
                        for j, use_idx in enumerate(use_idxs)]
            true_old_positions = [[old_positions[j][i] for i in use_idx]
                                for j, use_idx in enumerate(use_idxs)]
            chunk_preds = []
            chunk_labels = []
            for i, old_position in enumerate(true_old_positions):
                cur_pos = -100
                chunk_preds.append([])
                chunk_labels.append([])
                for j, pos in enumerate(old_position):
                    if pos != cur_pos:
                        cur_pos = pos
                        chunk_preds[-1].append(true_preds[i][j])
                        chunk_labels[-1].append(true_labels[i][j])
                    elif pos < cur_pos:
                        raise ValueError('later position has higher value than previous one')
            agg_chunk_preds.extend(chunk_preds)
            agg_chunk_labels.extend(chunk_labels)
            print(f'\rProcessed: {len(agg_chunk_preds)} / {len(dataset)}',
                flush=True, end=' ')
    
        return agg_chunk_labels, agg_chunk_preds


    def finetune(self,
                 training_args,
                 train_dataset: Dataset,
                 val_dataset: Dataset = None,
                 test_dataset: Dataset = None):
        
        self.training_args = training_args

        self._init_trainer(
            training_args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        self.trainer.train()
        self.trainer.save_model(
            os.path.join(training_args.output_dir,
                         'checkpoint-final')
        )

        if test_dataset != None:

            agg_chunk_labels, agg_chunk_preds = self._evaluate(test_dataset)
            agg_chunk_labels = [[self.id2label[e] for e in a] for a in agg_chunk_labels]
            agg_chunk_preds = [[self.id2label[e] for e in a] for a in agg_chunk_preds]

            if self.task == Task.CHUNK_LEVEL_CLS.value:
                result = chunk_level_classification_metrics(agg_chunk_labels, agg_chunk_preds)
            else:
                result = token_level_classification_metrics(sum(agg_chunk_labels, []),
                                                            sum(agg_chunk_preds, []))
            print('[ Test Result ]')
            pprint.pprint({k: v for k, v in result.items() if k != 'classification_report'})
            print(result['classification_report'])

            return result
