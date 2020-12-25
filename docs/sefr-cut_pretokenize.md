# Tokenize text word SEFR tokenizer for `fake_sefr_cut` tokenizer

The word level tokenizer that we use has pre_tokenize hook that will pre tokenize text before passing to the tokenizer unfortunately, this operate on instance by instance basis make this unsuitable for sefr cut since it is very slow compare to other pre tokenizer. To speed up the process we use multiprocessing to accelerate the process and output text with special token that will be use as a split token for fake sefr cut later. For example, if the text `hello world` get split into `['hello', 'world']` by sefr cut we will output the preprocessed text as `hello<|>world` instead if `<|>` is the split token that we specified.

## Instruction

The following command will read text line by line from `input_folder` and output it to `output_folder` with chunk_size of number of process multiply by 200.

```bash
arguements=$(cat <<-EOF
 --input_folder ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/val
 --output_folder ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020_pre_tokenized/val
 --chunk_size $(($(nproc) * 200))
 --overwrite
EOF
)

python3 sefr_cut_pre_tokenizer.py $arguements
```
