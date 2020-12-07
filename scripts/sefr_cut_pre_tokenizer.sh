arguements=$(cat <<-EOF
 --input_folder ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/val
 --output_folder ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020_pre_tokenized/val
 --chunk_size $(($(nproc) * 200))
 --overwrite
EOF
)

python3 sefr_cut_pre_tokenizer.py $arguements
