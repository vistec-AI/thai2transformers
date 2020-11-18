# master
#ist-gpu-10
export GLUE_DIR=~/playgroud/glue
export MASTER_HOSTNAME=10.0.0.20
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --nnodes=2 \
    --master_addr=$MASTER_HOSTNAME \
    --master_port=2222 \
    --node_rank=0 \
    ~/playgroud/transformers/examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name MRPC \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/MRPC/ \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ~/tmp/mrpc_output_fp16/ \
    --fp16

# worker
#ist-gpu-10
export MASTER_HOSTNAME=10.0.0.20
export GLUE_DIR=~/playgroud/glue
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --nnodes=2 \
    --master_addr=$MASTER_HOSTNAME \
    --master_port=2222 \
    --node_rank=1 \
    ~/playgroud/transformers/examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name MRPC \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/MRPC/ \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ~/tmp/mrpc_output/\
    --fp16
