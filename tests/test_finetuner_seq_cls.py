import unittest

from thai2transformers.finetuner import (
    SequenceClassificationFinetuner,
)

from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer
)


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

    def test_load_pretrained_tokenizer_wangchanberta_spm(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-spm'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['spm']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiRobertaTokenizer')

    def test_load_pretrained_tokenizer_wangchanberta_newmm(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-newmm'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['newmm']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsNewmmTokenizer')
    
    def test_load_pretrained_tokenizer_wangchanberta_syllable(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-ssg'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['syllable']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'ThaiWordsSyllableTokenizer')

    def test_load_pretrained_tokenizer_wangchanberta_sefr(self):
        
        pretrained_tokenizer_name = 'airesearch/wangchanberta-base-wiki-sefr'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['sefr_cut']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'FakeSefrCutTokenizer')
    
    def test_load_pretrained_tokenizer_xlmr(self):
        
        pretrained_tokenizer_name = 'xlm-roberta-base'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['xlmr']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'XLMRobertaTokenizer')
      
    def test_load_pretrained_tokenizer_mbert(self):
        
        pretrained_tokenizer_name = 'bert-base-multilingual-cased'
        tokenizer_cls = TOKENIZER_CLS_MAPPING['mbert']
        
        seq_cls_finetuner = SequenceClassificationFinetuner()
        seq_cls_finetuner.load_pretrained_tokenizer(
            tokenizer_cls=tokenizer_cls,
            name_or_path=pretrained_tokenizer_name
        )

        self.assertEqual(seq_cls_finetuner.tokenizer.__class__.__name__,'BertTokenizer')
      


    # def test_load_pretrained_model(self):