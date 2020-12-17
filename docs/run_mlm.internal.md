# Training mask language model

This step train a mask language model using trained tokenizer and datasets. The tokenizer is implemented derived from a class `transformers.tokenization_utils.PreTrainedTokenizer`. There are a lot of arguments that can be pass in this step. The script is derived from huggingface transformers `examples/language_modeling/run_mlm.py` with decent amount of new arguments and modifications.

## How to 

The following command can be use to train a mask language model. We can also use `--help` to get more information.

```bash
python3 run_mlm.py \
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
 --tokenizer_type "$PROJECT_TOKENIZER_TYPE"
```

The command above will load tokenizer from `$PROJECT_TOKENIZER_PATH` with tokenizer type `$PROJECT_TOKENIZER_TYPE`. Create train dataset from `*.txt` file in `$PROJECT_TRAIN_DATASET_DIR` and valiadation dataset from `*.txt` file in `$PROJECT_EVAL_DATASET_DIR` with dataset type `MemmapConcatFullSentenceTextDataset`. The datasets created will be cached at `$PROJECT_CACHE_DIR`, right now there are now mechanism to detect if the cache is actually corresponding to the same datasets specified in `train_dir` or `eval_dir` (if the cache already exits it will skip reading from those text files).

Due to the fact that most of the the datasets creation does not use gpus. So to only build datasets and cache it without training we can also use `--build_dataset_only` flags to trigger script to quit before training step.

The above command is suitable for non distributed training, to use distributed training we will to add distributed command in front as following.

```bash
NODE_RANK=0
MASTER_ADDR=node-1-gpu
MASTER_PORT=23333
N_GPUS_PER_NODE=4
N_NODES=4

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

$distributed_command run_mlm.py \
...
```

Variable explanation:
 - `$NODE_RANK` need to be unique for each node and the master node need to be `0`.
 - `$MASTER_ADDR` is an adress of master node.
 - `$MASTER_PORT` is an open and avaliable port on master node for communication.
 - `$N_GPUS_PER_NODE` amount to how many gpus each node has or want to be used to part of training.
 - `$N_NODES` is number of nodes.
 
The above command will run the master node which is the main node for distributed training and has a hostname, `node-1-gpu`. We will need another 3 nodes to join the master node so the other node will run the same command with one change to the command by chaging `NODE_RANK=0` to `NODE_RANK=1`, `NODE_RANK=2`, `NODE_RANK=3`. The master node need to ran first then the worker nodes can join usually the master node need ~15s to be up and runing before worker nodes.
