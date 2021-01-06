from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


from thai2transformers.metrics import classification_metrics
from thai2transformers.datasets import (
    SequenceClassificationDataset,
    TokenClassificationDataset
)
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModel,
    AutoConfig
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput
)

from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertClassificationHead
)

from transformers.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaClassificationHead
)

from transformers.modeling_xlmroberta import (
    XLMRobertaConfig
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
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()),
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


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels > 1` a multilabel classification loss is computed (BCELoss).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultiLabelSequenceClassification(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()  

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels > 1` a multilabel classification loss is computed (BCELoss).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels > 1:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class XLMRobertaForMultiLabelSequenceClassification(RobertaForMultiLabelSequenceClassification):
    """
    This class overrides :class:`~thai2transformers.RobertaForMultiLabelSequenceClassificationn`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
