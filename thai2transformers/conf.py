from enum import Enum

from transformers import DataCollatorForLanguageModeling

from .data_collator import DataCollatorForSpanLevelMask

class Task(Enum):
    MULTICLASS_CLS = 'multiclass_classification'
    MULTILABEL_CLS = 'multilabel_classification'

class MLMObjective(Enum):
    SUBWORD_LEVEL = 'subword-level'
    SPAN_LEVEL = 'span-leve'



DATA_COLLATOR_CLASS_MAPPING= {
    MLMObjective.SUBWORD_LEVEL.value : DataCollatorForLanguageModeling,
    MLMObjective.SPAN_LEVEL.value : DataCollatorForSpanLevelMask
}