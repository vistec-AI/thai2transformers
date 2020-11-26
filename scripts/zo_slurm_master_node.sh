#!/usr/bin/env bash

if [ "$NODE_RANK" -ne 0 ]; then
    sleep "$WAIT_MASTER_NODE"
    MASTER_HOSTNAME=$(scontrol show job "$MASTER_NODE_JOBID" | grep ' NodeList' | awk -F'=' '{print $2}')  # cant use in image
    export MASTER_ADDR="$MASTER_HOSTNAME"  # Need to export in this case since singularity container spawn another process?
else
    MASTER_HOSTNAME=$(hostname -s)
    export MASTER_ADDR="$MASTER_HOSTNAME"
fi

distributed_command=$(cat <<-EOF
echo "Node rank: $NODE_RANK"
echo "Master Address: $MASTER_ADDR"

python3 -m torch.distributed.launch \
    --nproc_per_node="$N_GPUS_PER_NODE" \
    --nnodes="$N_NODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT"
EOF
)

if [ "$PROJECT_LOCAL_CACHE" = true ]; then
    # Rsync to tmp folder on the node. This increase start up time but should yield better performance
    # in practice it is the almost the same. Might be worth trying with larger dataset?
    NEW_PROJECT_CACHE_DIR="/tmp/$PROJECT_CACHE_DIR/"
    mkdir -p "$NEW_PROJECT_CACHE_DIR"
    rsync -av "$PROJECT_CACHE_DIR/" "$NEW_PROJECT_CACHE_DIR"
    PROJECT_CACHE_DIR="$NEW_PROJECT_CACHE_DIR"
fi

echo -e "Start runing command inside singularity container\n"

# Hacking this for now but this should be possible to wrap this in another helper script
singularity exec --bind /ist/ist-share/scads/zo/:/ist/ist-share/scads/zo/ --nv ../docker-images/zo_th2tfm.sif bash <( cat <<EOF

$distributed_command run_mlm.py \
 --tokenizer_name_or_path "$PROJECT_TOKENIZER_PATH"  \
 --ext txt \
 --train_dir "$PROJECT_TRAIN_DATASET_DIR" \
 --eval_dir "$PROJECT_EVAL_DATASET_DIR" \
 --fp16 \
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
 --prediction_loss_only \
 --logging_dir "$PROJECT_LOG_DIR" \
 --output_dir "$PROJECT_OUTPUT_DIR" \
 --add_space_token \
 --datasets_cache_dir "$PROJECT_CACHE_DIR" \
 --datasets_type MemmapConcatFullSentenceTextDataset \
 --architecture roberta-base \
 --tokenizer_type "$PROJECT_TOKENIZER_TYPE" \
 --mlm

EOF
)
