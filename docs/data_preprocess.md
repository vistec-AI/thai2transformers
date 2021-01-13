
## Data preprocessing


### Text cleaning/filtering

We run the followig command to clean and filter segments. This is the configuration that we used to preprocess Thai Wikipedia text dataset for Thai RoBERTa base model pretraining (with `transformers.LineByLineDataset`).

```bash
python clean_data.py \
../cleaned_data/thwiki.csv \
../dataset/cleaned/thwiki/thwiki.csv \
--drop_na \
--break_long_sentence \
--max_sentence_length 300 \
--drop_no_thai_char \
--min_newmm_token_len 4 \
--max_newmm_token_len 300 \
--remove_thwiki_section
```