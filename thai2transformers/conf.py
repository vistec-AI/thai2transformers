from enum import Enum


class Task(Enum):
    MULTICLASS_CLS = 'multiclass_classification'
    MULTILABEL_CLS = 'multilabel_classification'
    CHUNK_LEVEL_CLS = 'chunk_level_classification'
    TOKEN_LEVEL_CLS = 'token_level_classification'