#!/usr/bin/env bash

jobname="th-wiki-concat-fake-sefr-cut-tokenizer-002"

# export some variable we will use later

export LANG=C.UTF-8  # The env we set in docker image won't transfer to singularity image
export LC_ALL=C.UTF-8  # The env we set in docker image won't transfer to singularity image
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
            [Nn]* ) echo Quit; break;;
            * ) echo "Try again.";;
        esac
    done
fi
mkdir -p "$ZO_SLURM_LOG_OUTPUT_DIR"

# User defined directory

PROJECT_DATA_ROOT="/ist/ist-share/scads/zo/thai2transformers/"
export EXP_NAME="th-wiki-concat-fake-sefr-cut-tokenizer-002"

export PROJECT_TOKENIZER_PATH="$PROJECT_DATA_ROOT/dataset/fake-sefr-cut/thwiki-for-ddp_concat_12.11.2020_fake-sefr-cut_tokenizer_min_freq_4"
export PROJECT_TRAIN_DATASET_DIR="$PROJECT_DATA_ROOT/dataset/split/thwiki-for-ddp_concat_12.11.2020/train_pretokenized"
export PROJECT_EVAL_DATASET_DIR="$PROJECT_DATA_ROOT/dataset/split/thwiki-for-ddp_concat_12.11.2020/val_pretokenized"
export PROJECT_CACHE_DIR="$PROJECT_DATA_ROOT/cache/fake-sefr-cut-min-freq-4"
export PROJECT_OUTPUT_DIR="$PROJECT_DATA_ROOT/data/output/$EXP_NAME/model"
export PROJECT_LOG_DIR="$PROJECT_DATA_ROOT/data/output/$EXP_NAME/logs"
export PROJECT_LOCAL_CACHE=false
#export PROJECT_MODEL_DIR="/ist/ist-share/scads/zo/thai2transformers//data/output/th-wiki-concat-newmm-test-003/model/checkpoint-2000/"

# User defined hyperparameters

export PROJECT_MAX_SEQ_LENGTH=512
export PROJECT_LEARNING_RATE=7e-4
export PROJECT_ADAM_BETA2=0.98  # According to paper, they said 0.98 instead of default 0.999 improve stability
export PROJECT_MAX_STEPS=31250
export PROJECT_BATCH_SIZE=16
export PROJECT_GRAD_ACC_STEPS=32
export PROJECT_WARMUP_STEPS=1250  # Warmup step is usally around <5-10% of total step
export PROJECT_SEED=2020
export PROJECT_SAVE_STEPS=500
export PROJECT_EVAL_STEPS=500
export PROJECT_TOKENIZER_TYPE="FakeSefrCutTokenizer"  # This need to go together with pretokenizer type
export PROJECT_PRE_TOKENIZER_TYPE="skip"  # if this = skip it will skip training tokenizer process
export PROJECT_VOCAB_MIN_FREQ=9

# Define multi-nodes variables

export N_NODES=4
export N_GPUS_PER_NODE=4
export MASTER_PORT=13335
# We will define master address later
# export MASTER_ADDR=$HOSTNAME

# Node cluster spec

MEMORY_NEED=128GB
TIME_NEED="2-00:00:00"
PARTITION_NAME="gpu-cluster"
ACCOUNT_NAME="scads"
N_CPUS_PER_GPU=2  # 1 CPU per GPU yield almost 100% utilization on CPU but have no effect on performance.
                  # idk why but set this to 2 for now so we have a bit of head room.

# Confirm Information

cat <<EOF
=================Info==================

PWD: $PWD
Jobname: $jobname
Slurm Output Dir: $ZO_SLURM_MAIN_FOLDER

PROJECT_DATA_ROOT: $PROJECT_DATA_ROOT
EXP_NAME: $EXP_NAME

===============Resource================
N_NODES: $N_NODES
N_GPUS_PER_NODE: $N_GPUS_PER_NODE
N_CPUS_PER_GPU: $N_CPUS_PER_GPU
TIME_ALLOC: $TIME_NEED

=============Configuration=============

PROJECT_TOKENIZER_PATH: $PROJECT_TOKENIZER_PATH
PROJECT_TRAIN_DATASET_DIR: $PROJECT_TRAIN_DATASET_DIR
PROJECT_EVAL_DATASET_DIR: $PROJECT_EVAL_DATASET_DIR
PROJECT_CACHE_DIR: $PROJECT_CACHE_DIR
PROJECT_OUTPUT_DIR: $PROJECT_OUTPUT_DIR
PROJECT_LOG_DIR: $PROJECT_LOG_DIR
PROJECT_LOCAL_CACHE: $PROJECT_LOG_DIR
PROJECT_MODEL_DIR: $PROJECT_MODEL_DIR

============Hyperparameters============

PROJECT_MAX_SEQ_LENGTH: $PROJECT_MAX_SEQ_LENGTH
PROJECT_LEARNING_RATE: $PROJECT_LEARNING_RATE
PROJECT_MAX_STEPS: $PROJECT_MAX_STEPS
PROJECT_BATCH_SIZE: $PROJECT_BATCH_SIZE
PROJECT_GRAD_ACC_STEPS: $PROJECT_GRAD_ACC_STEPS
PROJECT_WARMUP_STEPS: $PROJECT_WARMUP_STEPS
PROJECT_SEED: $PROJECT_SEED
PROJECT_SAVE_STEPS: $PROJECT_SAVE_STEPS
PROJECT_EVAL_STEPS: $PROJECT_EVAL_STEPS

==============Tokenizer================

PROJECT_TOKENIZER_TYPE: $PROJECT_TOKENIZER_TYPE
PROJECT_PRE_TOKENIZER_TYPE: $PROJECT_PRE_TOKENIZER_TYPE
PROJECT_VOCAB_MIN_FREQ: $PROJECT_VOCAB_MIN_FREQ

Effective Batchsize: $((PROJECT_BATCH_SIZE * PROJECT_GRAD_ACC_STEPS * N_GPUS_PER_NODE * N_NODES))

EOF

while true; do
    read -p "Is the configuration correct (y/n)?" answer
    case "$answer" in
        [Yy]* ) break;;
        [Nn]* ) echo Quit; exit;;
        * ) echo "Try again.";;
    esac
done

# Train Tokenizer

echo "Queue Train tokenizer job."

ZO_SLURM_TRAIN_TOKENIZER_JOBID=$(
sbatch \
    --partition=cpu \
    --cpus-per-task=4 \
    --time=60 \
    --mem=32GB \
    --account="$ACCOUNT_NAME" \
    --output="$ZO_SLURM_LOG_OUTPUT_DIR/train_tokenizer_%j.out" \
    --job-name="$jobname-train-tokenizer" \
    zo_slurm_train_tokenizer.sh | awk '{ print $4 }'
)

# Preprocessing

echo "Queue Preprocess job."

ZO_SLURM_PREPROCESS_JOBID=$(
sbatch \
    --dependency=afterok:"$ZO_SLURM_TRAIN_TOKENIZER_JOBID" \
    --partition=cpu \
    --cpus-per-task=4 \
    --time=60 \
    --mem=32GB \
    --account="$ACCOUNT_NAME" \
    --output="$ZO_SLURM_LOG_OUTPUT_DIR/preprocess_%j.out" \
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
    --cpus-per-gpu=$N_CPUS_PER_GPU \
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

# It should be able to delay a bit before submit job for now we just
# use sleep to wait it
# One possible way to work around this is to launch dummy sleep task
# then set worker node to depend on it
# Or latest slurm can set sbatch --dependency with time
# but not the version we got on cluster.

ZO_SLURM_WORKER_NODE_JOBID=$( \
env NODE_RANK="$NODE_RANK" \
    MASTER_NODE_JOBID="$ZO_SLURM_MASTER_NODE_JOBID" \
    WAIT_MASTER_NODE=0 \
sbatch \
    --dependency=after:"$ZO_SLURM_MASTER_NODE_JOBID" \
    --partition="$PARTITION_NAME" \
    --time="$TIME_NEED" \
    --mem="$MEMORY_NEED" \
    --cpus-per-gpu=$N_CPUS_PER_GPU \
    --gpus="$N_GPUS_PER_NODE" \
    --account="$ACCOUNT_NAME" \
    --output="$ZO_SLURM_LOG_OUTPUT_DIR/worker-node-${NODE_RANK}_%j.out" \
    --job-name="$jobname-worker-node-$NODE_RANK" \
    zo_slurm_master_node.sh | awk '{ print $4 }'
)

LAST_JOB_ID=$ZO_SLURM_WORKER_NODE_JOBID

done

squeue -u "$USER" -o "%.8A %.4C %.10m %.20E"