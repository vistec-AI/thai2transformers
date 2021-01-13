## `thai2transformers`

<br>

Pretraining transformers in Thai and English

<br>

### Pretraining Datasets

<br>

Developing. See this [spreadsheet](https://docs.google.com/spreadsheets/d/1lQ06FT2RvBE8twKzvXeSe4w5CHnU29f8ZWMUcJdmRks/edit?usp=sharing). Download current version of cleaned datasets [here](https://drive.google.com/file/d/1oF7_COZJqGdIaDGMNI1rKdDCOEzVoZHq/view?usp=sharing).

<br>

### Instruction:
 
<br>

1. Install required libraries: [1_installation.md](./docs/1_installation.md)  
2. Prepare `thwiki` dataset from Thai Wikipedia dump: [2_thwiki_data-preparation.md](./docs/2_thwiki_data-preparation.md)  


3. Tokenizer training and vocabulary building : 
    
    a) For SentencePiece BPE (`spm`), word-level token (`newmm`),  syllable-level token (`syllable`): [3_train_tokenizer.md](./docs/3_train_tokenizer.md)  
    
    b) For word-level token from [Limkonchotiwat et al., 2020](https://github.com/mrpeerat/SEFR_CUT) (`sefr-cut`) : [3b_sefr-cut_pretokenize.md](./docs/2b_sefr-cut_pretokenize.md)  

4. Pretrain a masked langauge model: [4_run_mlm.md](./docs/3_run_mlm.md)  


