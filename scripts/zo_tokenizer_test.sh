TOKENIZER_TYPE=newmm

python3 train_tokenizer.py \
 --ext txt \
 --train_dir  ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/train \
 --eval_dir  ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/val \
 --output_file "../data/input/zo_test/thwiki-for-ddp_concat_12.11.2020_newmm_tokenizer/$TOKENIZER_TYPE.json" \
 --pre_tokenizer_type "$TOKENIZER_TYPE" \
 --debug \
 --vocab_min_freq 3
