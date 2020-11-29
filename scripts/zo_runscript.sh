 EXP_NAME="zo_test"


arguements=$(cat <<-EOF
 --tokenizer_name_or_path ../data/input/zo_test/thwiki-for-ddp_concat_12.11.2020_newmm_tokenizer \
 --train_dir ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/zo_val \
 --eval_dir ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/zo_val \
 --max_seq_length 512 \
 --learning_rate 1e-3 --weight_decay 0.01 \
 --adam_epsilon 1e-6 \
 --max_steps 31250 \
 --adam_beta2 0.98 \
 --per_device_train_batch_size 2 \
 --per_device_eval_batch_size 2 \
 --gradient_accumulation_steps 1 \
 --warmup_steps 1250 \
 --seed 2020 \
 --save_steps 200 \
 --logging_steps 5 \
 --save_total_limit 10 \
 --evaluation_strategy steps \
 --prediction_loss_only \
 --eval_steps 2500 \
 --logging_dir "../data/output/$EXP_NAME/logs" \
 --output_dir "../data/output/$EXP_NAME/model" \
 --add_space_token \
 --datasets_cache_dir ../cache2/$EXP_NAME \
 --datasets_type MemmapConcatFullSentenceTextDataset \
 --tokenizer_type ThaiWordsSyllableTokenizer \
 --overwrite_cache \
 --mlm \
 --architecture roberta-base
EOF
)

arguements="$arguements --ext txt"

python3 -m pdb run_mlm.py $arguements

