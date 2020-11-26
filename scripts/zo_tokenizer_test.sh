TOKENIZER_TYPE=newmm

python3 train_tokenizer.py \
 --ext txt \
 --train_dir  ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/val \
 --eval_dir  ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/val \
 --output_file "$TOKENIZER_TYPE.json" \
 --pre_tokenizer_type "$TOKENIZER_TYPE" \
 --overwrite_output_file \
 --vocab_size 1000
