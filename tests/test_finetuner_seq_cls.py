import unittest

from transformers.testing_utils import require_torch

from thai2transformers.finetuner import (
    BaseFinetuner,
    SequenceClassificationFinetuner,
)

from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer
)

from thai2transformers.conf import Task

from transformers import (
    CamembertTokenizer,
    XLMRobertaTokenizer,
    BertTokenizer
)

TOKENIZER_CLS_MAPPING = {
    'spm_camembert': CamembertTokenizer,
    'spm': ThaiRobertaTokenizer,
    'newmm': ThaiWordsNewmmTokenizer,
    'syllable': ThaiWordsSyllableTokenizer,
    'sefr_cut': FakeSefrCutTokenizer,
    'xlmr': XLMRobertaTokenizer,
    'mbert': BertTokenizer,
}


class TestSequenceClassificationFinetuner(unittest.TestCase):

    def test_init(self):
        seq_cls_finetuner = SequenceClassificationFinetuner()
        self.assertIsNotNone(seq_cls_finetuner)
    
    def test_base_finetuner(self):
        base_finetuner = BaseFinetuner()
        self.assertIsNotNone(base_finetuner)
        base_finetuner.load_pretrained_tokenizer()
        base_finetuner.load_pretrained_model()
        base_finetuner.finetune()

    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_spm_camembert(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['spm_camembert']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'CamembertTokenizer')
        self.assertEqual(seq_cls_finetuner.tokenizer.additional_special_tokens,
                         ['<s>NOTUSED', '</s>NOTUSED', '<_>'])

    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_spm(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-spm'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['spm']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiRobertaTokenizer')
    
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_newmm(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-newmm'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['newmm']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsNewmmTokenizer')
    
    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_syllable(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-ssg'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['syllable']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsSyllableTokenizer')

    @require_torch
    def test_load_pretrained_tokenizer_wangchanberta_sefr(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-sefr'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['sefr_cut']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'FakeSefrCutTokenizer')
    
    @require_torch
    def test_load_pretrained_tokenizer_xlmr(self):
        
        pretrained_tokenizer_name = 'xlm-roberta-base'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['xlmr']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'XLMRobertaTokenizer')
    
    @require_torch
    def test_load_pretrained_tokenizer_mbert(self):
        
        pretrained_tokenizer_name = 'bert-base-multilingual-cased'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['mbert']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'BertTokenizer')
    
    @require_torch
    def test_load_pretrained_model_for_seq_cls_incorrect_task(self):
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate RobertaModelForSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        task = 'token_classification'
        with self.assertRaises(NotImplementedError) as context:
            seq_cls_finetuner.load_pretrained_model(
                task=task,
                name_or_path=pretrained_model_name
            )

        self.assertEqual(
            f'The task specified `{task}` is incorrect or not available for SequenceClassificationFinetuner',
            str(context.exception))

    @require_torch
    def test_load_pretrained_model_for_seq_cls_wangchanberta_spm_camembert(self):
        
        pretrained_model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
        
        # instantiate RobertaForSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name

        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate RobertaForSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multiclass_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate RobertaForSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTICLASS_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)

        # instantiate RobertaForMultiLabelSequenceClassification without `num_labels`
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 2) # num_labels = 2 is the default value
        
        # instantiate RobertaForMultiLabelSequenceClassification with `num_labels` specified
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task='multilabel_classification',
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)


        # instantiate RobertaForMultiLabelSequenceClassification with `num_labels` specified and using Task enum variable
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_model(
            task=Task.MULTILABEL_CLS,
            name_or_path=pretrained_model_name,
            num_labels=10
        )
        self.assertEqual(seq_cls_finetuner.model.__class__.__name__,'RobertaForMultiLabelSequenceClassification')
        self.assertEqual(seq_cls_finetuner.metric.__name__, 'multilabel_classification_metrics')
        self.assertEqual(seq_cls_finetuner.config.num_labels, 10)
