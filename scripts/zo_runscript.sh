 EXP_NAME="zo_test"

python3 run_mlm.py \
 --tokenizer_name_or_path ../data/input/zo_test/thwiki-for-ddp_concat_12.11.2020_spm_vs-24k_v2 \
 --ext txt \
 --train_dir ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/val \
 --eval_dir ../data/input/datasets/thwiki-for-ddp_concat_12.11.2020/val \
 --max_seq_length 512 \
 --learning_rate 3e-4 --weight_decay 0.01 \
 --adam_epsilon 1e-6 \
 --max_steps 100 \
 --per_device_train_batch_size 2 \
 --per_device_eval_batch_size 2 \
 --gradient_accumulation_steps 1 \
 --warmup_steps 24000 \
 --seed 2020 \
 --save_steps 100 \
 --logging_steps 5 \
 --save_total_limit 50 \
 --evaluate_during_training \
 --eval_steps 2500 \
 --logging_dir "../data/output/$EXP_NAME/logs" \
 --output_dir "../data/output/$EXP_NAME/model" \
 --add_space_token \
 --datasets_cache_dir ../cache/$EXP_NAME \
 --datasets_type MemmapConcatFullSentenceTextDataset \
 --architecture roberta-base \
 --build_dataset_only
