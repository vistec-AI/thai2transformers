#!/bin/bash
N_NODES=$1
HOSTNAME=$2
N_PROC_PER_NODE=$3
JOBID=$4
MAX_STEPS=$5
WARMUP_STEPS=$6
SAVE_STEPS=$7
EVAL_STEPS=$8
EXP_NAME=${9}
LR=${10}
BATCH_SIZE=${11}
GRAD_ACC=${12}
MODEL_CHECKPOINT_DIR=${13}

NODE_RANK=${SLURM_PROCID}

N_GPUS=`expr $N_NODES \* $N_PROC_PER_NODE `



echo "Number of GPU : $N_GPUS" |& tee -a ./slurm_logs/thwiki.ddp.6.11.2020.rank-$NODE_RANK.out
echo "Node rank $NODE_RANK" |& tee -a ./slurm_logs/thwiki.ddp.6.11.2020.rank-$NODE_RANK.out

export MASTER_PORT=9999
export MASTER_ADDR=$HOSTNAME

echo "Total steps = 500,000 / 32 (15,625 steps per GPU)"
echo "--learning_rate $LR "

module load CUDA/10.2

# EXP_NAME=exp012_thwiki-for-ddp_6.11.2020_spm_vs-24k_fp16_bz32_maxstep-500k_ngpus-32_maxseqlen-512_mlmdataset
if [[ "$MODEL_CHECKPOINT_DIR" != "" ]]; then
  echo "Resume model training from $MODEL_CHECKPOINT_DIR"
fi

if [[ "$NODE_RANK" != "0" ]]; then
  delay=`expr 5 + $NODE_RANK `     # Whitespace for expr is important
  echo "Node: $NODE_RANK is mot the master node, wait for $delay secs"
  sleep $delay
  echo "Done."
fi

echo "Global max_steps     = $MAX_STEPS     " |& tee -a ./slurm_logs/thwiki.ddp.6.11.2020.rank-$NODE_RANK.out
echo "Global warmup_steps  = $WARMUP_STEPS  " |& tee -a ./slurm_logs/thwiki.ddp.6.11.2020.rank-$NODE_RANK.out
echo "Global save_steps    = $SAVE_STEPS    " |& tee -a ./slurm_logs/thwiki.ddp.6.11.2020.rank-$NODE_RANK.out
echo "Global eval_steps    = $EVAL_STEPS    " |& tee -a ./slurm_logs/thwiki.ddp.6.11.2020.rank-$NODE_RANK.out


export WANDB_WATCH=true
export WANDB_MODE=dryrun
export WANDB_PROJECT=thai2transformers
export WANDB_ENTITY=lalital
export WANDB_DIR=/ist/ist-share/scads/aires/thai2transformers_store/wandb_logs/ \
export WANDB_NAME=$EXP_NAME
python -m torch.distributed.launch \
		--nproc_per_node=$N_PROC_PER_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
   ./train_mlm_camembert_thai.ddp.py \
    --tokenizer_name_or_path ../dataset/spm/thwiki-for-ddp_6.11.2020_spm_vs-24k/sentencepiece.bpe.model \
    --ext txt \
    --train_path ../dataset/split/thwiki-for-ddp_6.11.2020/train/train.txt \
    --eval_path ../dataset/split/thwiki-for-ddp_6.11.2020/val/val.txt \
    --block_size 510 \
    --learning_rate $LR --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --fp16 True \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --warmup_steps $WARMUP_STEPS \
    --seed 2020 \
    --save_steps $SAVE_STEPS \
    --logging_steps 10 \
    --save_total_limit 100 \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEPS \
    --logging_dir /ist/ist-share/scads/aires/thai2transformers_store/logs/$EXP_NAME/ \
    --output_dir /ist/ist-share/scads/aires/thai2transformers_store/checkpoints/$EXP_NAME/ \
    --add_space_token \
    --datasets_cache_dir ../dataset/binarized/thwiki-for-ddp_6.11.2020/ \
    --model_directory $MODEL_CHECKPOINT_DIR \
    --dataset_loader_name linebyline |& tee -a ./slurm_logs/$EXP_NAME.job-$JOBID.rank-$NODE_RANK.out
