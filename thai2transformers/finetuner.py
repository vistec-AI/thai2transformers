import os
from typing import List, Dict, Union, Optional, Callable

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers.tokenization_utils import (
    PreTrainedTokenizer,
)
from .auto import (
    AutoModelForMultiLabelSequenceClassification
)
from .conf import (
    Task
)

AIRESEARCH_MODEL_PREFIX = 'airesearch/wangchanberta'
AIRESEARCH_MODEL_NAME = {
    f'{AIRESEARCH_MODEL_PREFIX}-base-att-spm-uncased': {
        'space_token': '<_>'
    }
}

FINETUNE_SEQ_CLS_MODEL_MAPPING = {
    Task.MULTICLASS_CLS.value: AutoModelForSequenceClassification,
    Task.MULTILABEL_CLS.value: AutoModelForMultiLabelSequenceClassification
}

class BaseFinetuner:

    def load_pretrained_tokenizer(self):
        pass

    def load_pretrained_model(self):
        pass

    def finetune(self, *kwargs):
        pass

class SequenceClassificationFinetuner:

    def __init__(self,
                tokenizer: PreTrainedTokenizer = None,
                config = None,
                task: Union[str, Task] = None,
                num_labels: int = None,
                *kwargs):

        self.tokenizer = tokenizer
        self.config = config
        self.task = task.value if type(task) == Task else task
        self.num_labels = num_labels

    def load_pretrained_tokenizer(self,
            tokenizer_cls: PreTrainedTokenizer,
            name_or_path: Union[str, os.PathLike]):
        """
        Load a pretrained tokenizer to the finetuner instance
        """

        self.tokenizer = tokenizer_cls.from_pretrained(name_or_path)

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
        num_labels: int = None):
        """
        Load a pretrained model to the finetuner instance and modify classification head
        according to the specified `task` in the method argument.

        Arguments:
            task: Union[str, Task],
        """
        if num_labels == None:
            self.config = AutoConfig.from_pretrained(name_or_path)
        else:
            self.config = AutoConfig.from_pretrained(
                name_or_path,
                num_labels=num_labels
            )

        if type(task) == Task:
            task = task.value
        
        self.task = task

        if  task not in FINETUNE_SEQ_CLS_MODEL_MAPPING.keys():
            raise NotImplementedError(f"The task specified `{task}` is incorrect or not available for {self.__class__.__name__}")

        self.model = FINETUNE_SEQ_CLS_MODEL_MAPPING[task].from_pretrained(name_or_path,
                                                        config=self.config)