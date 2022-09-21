
# thai2transformers

<p>
<a href="https://console.tiyaro.ai/explore/airesearch-wangchanberta-base-att-spm-uncased"> <img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/try_on_tiyaro_badge.svg"></a>
</p>

**Pretraining transformer-based Thai language models**


<br>

thai2transformers provides customized scripts to pretrain transformer-based masked language model on Thai texts with various types of tokens as follows:

- __spm__: a subword-level token from [SentencePiece](https://github.com/google/sentencepiece) library.
- __newmm__ : a dictionary-based Thai word tokenizer based on maximal matching from [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp).
- __syllable__: a dictionary-based Thai syllable tokenizer based on maximal matching from [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp). The list of syllables used is from [pythainlp/corpus/syllables_th.txt](https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/corpus/syllables_th.txt).
- __sefr__: a ML-based Thai word tokenizer based on Stacked Ensemble Filter and Refine (SEFR) [[Limkonchotiwat et al., 2020]](https://www.aclweb.org/anthology/2020.emnlp-main.315/) based on probabilities from CNN-based [deepcut](https://github.com/rkcosmos/deepcut) and SEFR tokenizer is loaded with `engine="best"`.


<br>

<br>

### Thai texts for language model pretraining

<br>

We curate a list of sources that can be used to pretrain language model.
The statistics for each data source are listed in this [page](./docs/_dataset_statistics.md). 

Also, you can download current version of cleaned datasets from [here](https://github.com/vistec-AI/thai2transformers/releases/tag/att-v1.0).

<br>

<br>

### Model pretraining and finetuning instructions:

<br>

**a) Instruction for RoBERTa BASE model pretraining on Thai Wikipedia dump:**
 
In this example, we demonstrate how pretrain RoBERTa base model on Thai Wikipedia dump from scratch

1. Install required libraries: [1_installation.md](./docs/1_installation.md)  
2. Prepare `thwiki` dataset from Thai Wikipedia dump: [2_thwiki_data-preparation.md](./docs/2_thwiki_data-preparation.md)  


3. Tokenizer training and vocabulary building : 
    
    a) For SentencePiece BPE (`spm`), word-level token (`newmm`),  syllable-level token (`syllable`): [3_train_tokenizer.md](./docs/3_train_tokenizer.md)  
    
    b) For word-level token from [Limkonchotiwat et al., 2020](https://github.com/mrpeerat/SEFR_CUT) (`sefr-cut`) : [3b_sefr-cut_pretokenize.md](./docs/2b_sefr-cut_pretokenize.md)  

4. Pretrain a masked langauge model: [4_run_mlm.md](./docs/3_run_mlm.md)  

<br>

**b) Instruction for RoBERTa model finetuning on existing Thai text classification, and NER/POS tagging datasets.**


In this example, we demonstrate how to finetune WanchanBERTa, a RoBERTa base model pretrained on Thai Wikipedia dump and Thai assorted texts. 


- Finetune model for sequence classification task from exisitng datasets including `wisesight_sentiment`, `wongnai_reviews`, `generated_reviews_enth` (review star prediction), and `prachathai67k`:
[5a_finetune_sequence_classificaition.md](./docs/5a_finetune_sequence_classificaition.md)  

- Finetune model for token classification task (NER and POS tagging) from exisitng datasets including `thainer` and `lst20`:
[5b_finetune_token_classificaition.md](./docs/5b_finetune_token_classificaition.md)  

<br>

<br>


###  BibTeX entry and citation info


```
@misc{lowphansirikul2021wangchanberta,
      title={WangchanBERTa: Pretraining transformer-based Thai Language Models}, 
      author={Lalita Lowphansirikul and Charin Polpanumas and Nawat Jantrakulchai and Sarana Nutanong},
      year={2021},
      eprint={2101.09635},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```