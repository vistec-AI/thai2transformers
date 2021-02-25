import os
from typing import List, Dict, Union, Optional, Callable


from transformers.tokenization_utils import (
    PreTrainedTokenizer
)


AIRESEARCH_MODEL_PREFIX = 'airesearch/wangchanberta'
AIRESEARCH_MODEL_NAME = {
    f'{AIRESEARCH_MODEL_PREFIX}-base-att-spm-uncased': {
        'space_token': '<_>'
    }
}


class BaseFinetuner:

    def load_pretrained_tokenizer(self):
        pass

    def load_pretrained_model(self):
        pass

    def finetune(self, *kwargs):
        pass

class SequenceClassificationFinetuner:

    def __init__(self, tokenizer=None, *kwargs):

        self.tokenizer = tokenizer

    def load_pretrained_tokenizer(self,
            tokenizer_cls: PreTrainedTokenizer,
            name_or_path: Union[str, os.PathLike]):
        """
        Load a tokenizer from pretrained tokenizer to the finetuner instance
        """
        
        self.tokenizer = tokenizer_cls.from_pretrained(name_or_path)

        if tokenizer_cls.__name__ == 'CamembertTokenizer':

            if name_or_path in AIRESEARCH_MODEL_NAME.keys():
            
                self.tokenizer.additional_special_tokens = [
                    '<s>NOTUSED',
                    '</s>NOTUSED',
                    AIRESEARCH_MODEL_NAME[name_or_path]['space_token']
                ]

