export N_NODES=$1
export NODE_RANK=$2
export HOSTNAME=$3
export N_GPU_NODE=$4

echo "Rank $NODE_RANK"

export MASTER_PORT=9999
export MASTER_ADDR=$HOSTNAME

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    python train_mlm_camembert_thai_ddp.py \
    --tokenizer_name_or_path ../dataset/spm/thwiki-for-ddp_6.11.2020_spm_vs-24k/sentencepiece.bpe.model \
    --ext txt \
    --train_dir ../dataset/split/thwiki-for-ddp_6.11.2020/train/ \
    --eval_dir ../dataset/split/thwiki-for-ddp_6.11.2020/val/ \
    --block_size 512 \
    --learning_rate 3e-4 --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --fp16 True \
    --max_steps 15625 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --warmup_steps 500 \
    --seed 2020 \
    --save_steps 2500 \
    --logging_steps 1 \
    --save_total_limit 50 \
    --evaluate_during_training \
    --eval_steps 1000 \
    --logging_dir ../logs/exp012_thwiki-for-ddp_6.11.2020_spm_vs-24k_fp16_bz32_maxstep-500k_ngpus-32_maxseqlen-512_mlmdataset \
    --output_dir ../checkpoints/exp011_thwiki-for-ddp_6.11.2020_spm_vs-24k_fp16_bz32_maxstep-500k_ngpus-32_maxseqlen-512_mlmdataset \
    --add_space_token \
    --datasets_cache_dir ../dataset/binarized/thwiki-for-ddp_4.11.2020 \
    --preprocessing_num_workers 20