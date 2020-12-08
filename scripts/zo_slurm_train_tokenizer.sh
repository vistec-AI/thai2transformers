#!/usr/bin/env bash

if [ "$PROJECT_PRE_TOKENIZER_TYPE" = "skip" ]; then
    echo "Skip tokenizer training process."
    exit 0
fi

singularity exec --bind /ist/ist-share/scads/zo/:/ist/ist-share/scads/zo/ ../docker-images/zo_th2tfm.sif bash <( cat <<EOF

python3 train_tokenizer.py \
 --ext txt \
 --train_dir "$PROJECT_TRAIN_DATASET_DIR" \
 --eval_dir "$PROJECT_EVAL_DATASET_DIR" \
 --output_file "$PROJECT_TOKENIZER_PATH/$PROJECT_PRE_TOKENIZER_TYPE.json" \
 --pre_tokenizer_type "$PROJECT_PRE_TOKENIZER_TYPE" \
 --overwrite_output_file \
 --vocab_min_freq "$PROJECT_VOCAB_MIN_FREQ"

EOF
)
