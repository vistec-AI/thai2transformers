#!/usr/bin/env bash

jobname="wiki-test"

# Load appropiate packages

module load Anaconda3

# Fix since when you use `module load` inside bash script
# it does not source conda correctly
. /ist/apps/modules/software/Anaconda3/5.3.0/etc/profile.d/conda.sh

conda activate dev

# Sanity check

echo "$CONDA_DEFAULT_ENV"
python3 --version

# export some variable we will use later

export ZO_SLURM_MAIN_FOLDER="$jobname"
export ZO_SLURM_LOG_OUTPUT_DIR="$jobname/slurm-logs"

# Prepare folder structure

if [ -e "$ZO_SLURM_MAIN_FOLDER" ]; then
    echo "Main Slurm Output Folder: $ZO_SLURM_MAIN_FOLDER exist."
    while true; do
        read -p "Do you want to clear it out (y/n)?" answer
        case "$answer" in
            [Yy]* )
                rm -r "$ZO_SLURM_MAIN_FOLDER"
                break;;
            [Nn]* ) break;;
            * ) echo "Try again.";;
        esac
    done
fi
mkdir -p "$ZO_SLURM_LOG_OUTPUT_DIR"

# User defined directory

PROJECT_DATA_ROOT="/ist/ist-share/scads/zo/thai2transformers/"

export EXP_NAME="thwiki-ddp-concat-test-002"

export PROJECT_TOKENIZER_PATH="$PROJECT_DATA_ROOT/dataset/spm/thwiki-for-ddp_concat_12.11.2020_spm_vs-24k_v2"
export PROJECT_TRAIN_DATASET_DIR="$PROJECT_DATA_ROOT/dataset/split/thwiki-for-ddp_concat_12.11.2020/val"
export PROJECT_EVAL_DATASET_DIR="$PROJECT_DATA_ROOT/dataset/split/thwiki-for-ddp_concat_12.11.2020/val"
export PROJECT_CACHE_DIR="$PROJECT_DATA_ROOT/cache/$EXP_NAME "
export PROJECT_OUTPUT_DIR="$PROJECT_DATA_ROOT/data/output/$EXP_NAME/model"
export PROJECT_LOG_DIR="$PROJECT_DATA_ROOT/data/output/$EXP_NAME/logs"

# User defined hyperparameters

export PROJECT_MAX_SEQ_LENGTH=512
export PROJECT_LEARNING_RATE=3e-4
export PROJECT_MAX_STEPS=100
export PROJECT_BATCH_SIZE=2
export PROJECT_GRAD_ACC_STEPS=2
export PROJECT_WARMUP_STEPS=24000
export PROJECT_SEED=2020
export PROJECT_SAVE_STEPS=100
export PROJECT_EVAL_STEPS=100

# Define multi-nodes variables

export N_NODES=4
export N_GPUS_PER_NODE=4
export MASTER_PORT=13335
# We will define master address later
# export MASTER_ADDR=$HOSTNAME

# Node cluster spec

MEMORY_NEED=64GB
TIME_NEED="30"
PARTITION_NAME="gpu-cluster"
ACCOUNT_NAME="scads"

# Preprocessing

echo "Queue Preprocess job."

ZO_SLURM_PREPROCESS_JOBID=$(
sbatch \
    --partition=cpu \
    --cpus-per-task=4 \
    --time=60 \
    --mem=32GB \
    --account="$ACCOUNT_NAME" \
    --output="$ZO_SLURM_LOG_OUTPUT_DIR/%j.out" \
    --job-name="$jobname-preprocess" \
    zo_slurm_preprocessing.sh | awk '{ print $4 }'
)

# Master Node

echo "Queue Master Node job."

ZO_SLURM_MASTER_NODE_JOBID=$( env NODE_RANK=0 \
sbatch \
    --dependency=afterok:"$ZO_SLURM_PREPROCESS_JOBID" \
    --partition="$PARTITION_NAME" \
    --time="$TIME_NEED" \
    --mem="$MEMORY_NEED" \
    --cpus-per-gpu=2 \
    --gpus="$N_GPUS_PER_NODE" \
    --account="$ACCOUNT_NAME" \
    --output="$ZO_SLURM_LOG_OUTPUT_DIR/master-node_%j.out" \
    --job-name="$jobname-master-node" \
    zo_slurm_master_node.sh | awk '{ print $4 }'
)

LAST_JOB_ID=$ZO_SLURM_MASTER_NODE_JOBID

# Worker Node

for ((NODE_RANK=1; NODE_RANK<N_NODES; NODE_RANK++)); do

echo "Queue Worker Node ($NODE_RANK) job."

# It should be able to delay ab bit before submit job for now we just
# use sleep to wait it
# One possible way to work around this is to launch dummy sleep task
# then set worker node to depend on it
# Or lastest slurm can set sbatch --dependency with time
# but not the version we got on cluster.

ZO_SLURM_WORKER_NODE_JOBID=$( \
env NODE_RANK=$NODE_RANK \
    MASTER_NODE_JOBID=$ZO_SLURM_MASTER_NODE_JOBID \
    WAIT_MASTER_NODE=10 \
sbatch \
    --dependency=after:"$ZO_SLURM_MASTER_NODE_JOBID" \
    --partition="$PARTITION_NAME" \
    --time="$TIME_NEED" \
    --mem="$MEMORY_NEED" \
    --cpus-per-gpu=2 \
    --gpus="$N_GPUS_PER_NODE" \
    --account="$ACCOUNT_NAME" \
    --output="$ZO_SLURM_LOG_OUTPUT_DIR/worker-node-$NODERANK_%j.out" \
    --job-name="$jobname-worker-node-$NODE_RANK" \
    zo_slurm_master_node.sh | awk '{ print $4 }'
)

LAST_JOB_ID=$ZO_SLURM_WORKER_NODE_JOBID

done

squeue -u "$USER" -o "%.8A %.4C %.10m %.20E"
