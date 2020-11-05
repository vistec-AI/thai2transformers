
# MASTER
export BASE_DIR=~/app/thai2transformers
export MASTER_HOSTNAME=10.0.0.20
export CUDA_VISIBLE_DEVICES=0,1,2,3,4 

python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --nnodes=2 \
    --master_addr=$MASTER_HOSTNAME \
    --master_port=2222 \
    --node_rank=0 \
    ./train_mlm_camembert_thai_zo.py \
    --tokenizer_name_or_path $BASE_DIR/dataset/spm/thwiki-for-ddp_4.11.2020_spm_vs-24k/sentencepiece.bpe.model \
    --ext txt \
    --train_dir $BASE_DIR/dataset/split/thwiki-for-ddp_4.11.2020/train/ \
    --eval_dir $BASE_DIR/dataset/split/thwiki-for-ddp_4.11.2020/val/ \
    --block_size 416 \
    --learning_rate 3e-4 --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps 10000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --warmup_steps 500 \
    --seed 2020 \
    --save_steps 2500 \
    --logging_steps 1 \
    --save_total_limit 50 \
    --evaluate_during_training \
    --eval_steps 1000 \
    --logging_dir $BASE_DIR/logs/exp011_thwiki-for-ddp_4.11.2020_spm_vs-24k_fp16_bz32_maxstep-10k_ngpus-8_maxseqlen-416_mlmdataset \
    --output_dir $BASE_DIR/checkpoints/exp011_thwiki-for-ddp_4.11.2020_spm_vs-24k_fp16_bz32_maxstep-10k_ngpus-8_maxseqlen-416_mlmdataset \
    --add_space_token \
    --datasets_cache_dir $BASE_DIR/binarized/thwiki-for-ddp_4.11.2020 \
    --preprocessing_num_workers 20


# WORKER
export BASE_DIR=~/app/thai2transformers
export MASTER_HOSTNAME=10.0.0.20
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --nnodes=2 \
    --master_addr=$MASTER_HOSTNAME \
    --master_port=2222 \
    --node_rank=1 \
    ./train_mlm_camembert_thai_zo.py \
    --tokenizer_name_or_path $BASE_DIR/dataset/spm/thwiki-for-ddp_4.11.2020_spm_vs-24k/sentencepiece.bpe.model \
    --ext txt \
    --train_dir $BASE_DIR/dataset/split/thwiki-for-ddp_4.11.2020/train/ \
    --eval_dir $BASE_DIR/dataset/split/thwiki-for-ddp_4.11.2020/val/ \
    --block_size 416 \
    --learning_rate 3e-4 --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_steps 10000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --warmup_steps 500 \
    --seed 2020 \
    --save_steps 2500 \
    --logging_steps 1 \
    --save_total_limit 50 \
    --evaluate_during_training \
    --eval_steps 1000 \
    --logging_dir $BASE_DIR/logs/exp011_thwiki-for-ddp_4.11.2020_spm_vs-24k_fp16_bz32_maxstep-10k_ngpus-8_maxseqlen-416_mlmdataset \
    --output_dir $BASE_DIR/checkpoints/exp011_thwiki-for-ddp_4.11.2020_spm_vs-24k_fp16_bz32_maxstep-10k_ngpus-8_maxseqlen-416_mlmdataset \
    --add_space_token \
    --datasets_cache_dir $BASE_DIR/binarized/thwiki-for-ddp_4.11.2020 \
    --preprocessing_num_workers 20