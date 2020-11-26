#!/usr/bin/env bash

singularity exec --bind /ist/ist-share/scads/zo/:/ist/ist-share/scads/zo/ ../docker-images/zo_th2tfm.sif bash <( cat <<EOF

python3 run_mlm.py \
 --tokenizer_name_or_path "$PROJECT_TOKENIZER_PATH"  \
 --ext txt \
 --train_dir "$PROJECT_TRAIN_DATASET_DIR" \
 --eval_dir "$PROJECT_EVAL_DATASET_DIR" \
 --max_seq_length "$PROJECT_MAX_SEQ_LENGTH" \
 --learning_rate "$PROJECT_LEARNING_RATE" --weight_decay 0.01 \
 --adam_beta2 "$PROJECT_ADAM_BETA2" \
 --adam_epsilon 1e-6 \
 --max_steps "$PROJECT_MAX_STEPS" \
 --per_device_train_batch_size "$PROJECT_BATCH_SIZE" \
 --per_device_eval_batch_size "$PROJECT_BATCH_SIZE" \
 --gradient_accumulation_steps "$PROJECT_GRAD_ACC_STEPS" \
 --warmup_steps "$PROJECT_WARMUP_STEPS" \
 --seed "$PROJECT_SEED" \
 --save_steps "$PROJECT_SAVE_STEPS" \
 --logging_steps 10 \
 --save_total_limit 100 \
 --evaluation_strategy steps \
 --eval_steps "$PROJECT_EVAL_STEPS" \
 --logging_dir "$PROJECT_LOG_DIR" \
 --output_dir "$PROJECT_OUTPUT_DIR" \
 --datasets_cache_dir "$PROJECT_CACHE_DIR" \
 --datasets_type MemmapConcatFullSentenceTextDataset \
 --architecture roberta-base \
 --tokenizer_type "$PROJECT_TOKENIZER_TYPE" \
 --mlm \
 --no_cuda \
 --build_dataset_only

EOF
)
