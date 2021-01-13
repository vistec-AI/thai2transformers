from collections import OrderedDict

from transformers import (
    AutoConfig,
    PretrainedConfig
)

from transformers.modeling_bert import BertConfig
from transformers.modeling_roberta import RobertaConfig
from transformers.modeling_xlm_roberta import XLMRobertaConfig

from .models import (
    XLMRobertaForMultiLabelSequenceClassification,
    BertForMultiLabelSequenceClassification,
    RobertaForMultiLabelSequenceClassification
)

MODEL_FOR_MULTI_LABEL_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (XLMRobertaConfig, XLMRobertaForMultiLabelSequenceClassification),
        (BertConfig, BertForMultiLabelSequenceClassification),        
        (RobertaConfig, RobertaForMultiLabelSequenceClassification),
    ]
)

class AutoModelForMultiLabelSequenceClassification:
    
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForMultiLabelSequenceClassification is designed to be instantiated "
            "using the `AutoModelForMultiLabelSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForMultiLabelSequenceClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a sequence classification head---from a
        configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use
            :meth:`~transformers.AutoModelForMultiLabelSequenceClassification.from_pretrained` to load the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForMultiLabelSequenceClassification
            >>> # Download configuration from S3 and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModelForMultiLabelSequenceClassification.from_config(config)
        """
        if type(config) in MODEL_FOR_MULTI_LABEL_SEQUENCE_CLASSIFICATION_MAPPING.keys():
            return MODEL_FOR_MULTI_LABEL_SEQUENCE_CLASSIFICATION_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MULTI_LABEL_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForMultiLabelSequenceClassification

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModelForMultiLabelSequenceClassification.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelForMultiLabelSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelForMultiLabelSequenceClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_MULTI_LABEL_SEQUENCE_CLASSIFICATION_MAPPING.keys():
            return MODEL_FOR_MULTI_LABEL_SEQUENCE_CLASSIFICATION_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MULTI_LABEL_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
            )
        )
