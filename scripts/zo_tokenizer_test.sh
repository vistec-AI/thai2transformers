TOKENIZER_TYPE=fake_sefr_cut

python3 train_tokenizer.py \
 --ext txt \
 --train_dir  ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020_pre_tokenized/val \
 --eval_dir  ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020_pre_tokenized/val \
 --output_file "../data/input/zo_test/thwiki-for-ddp_concat_12.11.2020_sefr_cut_tokenizer/$TOKENIZER_TYPE.json" \
 --pre_tokenizer_type "$TOKENIZER_TYPE" \
 --overwrite_output_file \
 --debug \
 --vocab_min_freq 1
