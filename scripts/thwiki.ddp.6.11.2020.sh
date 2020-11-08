#!/bin/bash
N_NODES=$1
NODE_RANK=${SLURM_PROCID}
HOSTNAME=$2
N_GPU_NODE=$3
JOBID=$4
echo "Node rank $NODE_RANK" |& tee -a ./slurm_logs/thwiki.ddp.6.11.2020.rank-$NODE_RANK.out

export MASTER_PORT=9999
export MASTER_ADDR=$HOSTNAME

echo "Total steps = 500,000 / 8 (15,625 steps per GPU)"
echo "--learning_rate 6e-4 "

module load CUDA/10.0

EXP_NAME=exp012_thwiki-for-ddp_6.11.2020_spm_vs-24k_fp16_bz32_maxstep-500k_ngpus-32_maxseqlen-512_mlmdataset

if [[ "$NODE_RANK" != "0" ]]; then
  delay=`expr 5 + $NODE_RANK `     # Whitespace for expr is important
  echo "Node: $NODE_RANK is mot the master node, wait for $delay secs"
  sleep $delay
  echo "Done."
fi

WANDB_WATCH=true WANDB_MODE=dryrun WANDB_PROJECT=thai2transformers WANDB_ENTITY=lalital WANDB_DIR=/ist/ist-share/scads/aires/thai2transformers_store/wandb_logs/ \
WANDB_NAME=$EXP_NAME CUDA_VISBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
		--nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
   ./train_mlm_camembert_thai.ddp.py \
    --tokenizer_name_or_path ../dataset/spm/thwiki-for-ddp_6.11.2020_spm_vs-24k/sentencepiece.bpe.model \
    --ext txt \
    --train_path ../dataset/split/thwiki-for-ddp_6.11.2020/train/train.txt \
    --eval_path ../dataset/split/thwiki-for-ddp_6.11.2020/val/val.txt \
    --block_size 512 \
    --learning_rate 6e-4 --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --fp16 True \
    --max_steps 15625 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --warmup_steps 24000 \
    --seed 2020 \
    --save_steps 5000 \
    --logging_steps 5 \
    --save_total_limit 100 \
    --evaluation_strategy steps \
    --eval_steps 2500 \
    --logging_dir /ist/ist-share/scads/aires/thai2transformers_store/logs/exp012_thwiki-for-ddp_6.11.2020_spm_vs-24k_fp16_bz32_maxstep-500k_ngpus-32_maxseqlen-512_mlmdataset/ \
    --output_dir /ist/ist-share/scads/aires/thai2transformers_store/checkpoints/exp012thwiki-for-ddp_6.11.2020_spm_vs-24k_fp16_bz32_maxstep-500k_ngpus-32_maxseqlen-512_mlmdataset/ \
    --add_space_token \
    --datasets_cache_dir ../dataset/binarized/thwiki-for-ddp_6.11.2020/ |& tee -a ./slurm_logs/thwiki.ddp.6.11.2020.j-$JOBID.rank-$NODE_RANK.out
