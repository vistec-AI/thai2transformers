# Training mask language model

This step train a mask language model using trained tokenizer and datasets. The tokenizer is implemented derived from a class `transformers.tokenization_utils.PreTrainedTokenizer`. There are a lot of arguments that can be pass in this step. The script is derived from huggingface transformers `examples/language_modeling/run_mlm.py` with decent amount of new arguments and modifications.

## Instruction

The following command can be use to train a mask language model (Append `--help` after the `run_mlm.py` to get more information).

```bash
python run_mlm.py \
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

The command above will load tokenizer from `$PROJECT_TOKENIZER_PATH` with tokenizer type `$PROJECT_TOKENIZER_TYPE` (including __ThaiRobertaTokenizer__ (Subword-level, SentencePiece), __ThaiWordsNewmmTokenizer__ (Word-level, PyThaiNLP's newmm), __ThaiWordsSyllableTokenizer__ (syllable-level, CRF-based syllable segmentor), __FakeSefrCutTokenizer__ (Word-level, ML-based word tokenizer),). Then, create binarized dataset from `*.txt` file in `$PROJECT_TRAIN_DATASET_DIR` and validation dataset from `*.txt` file in `$PROJECT_EVAL_DATASET_DIR` with dataset type `MemmapConcatFullSentenceTextDataset`. The datasets created will be cached at `$PROJECT_CACHE_DIR`, right now there are now mechanism to detect if the cache is actually corresponding to the same datasets specified in `train_dir` or `eval_dir` (if the cache already exits it will skip reading from those text files).

Due to the fact that most of the the datasets creation does not use gpus. So to only build datasets and cache it without training we can also use `--build_dataset_only` flags to trigger script to quit before training step.

<br>

For instance, the following command will train `roberta-base` model on 1 GPU (GPU ID: 3) with FP16 mixed-precision training on the `thwiki-20200820` dataset. The maximum sequence length is set to 64. The batch size is 1 with gradient accumulation of 16 step. The maximum trianing step is set to 100 steps with 10 warmup steps in which the learning rate is increased linearly to the peak value of `5e-4` and linearly decayed to zero.

```
cd scripts

CUDA_VISIBLE_DEVIDES="3" python run_mlm.py \
 --architecture roberta-base \
 --tokenizer_name_or_path /workspace/thai2transformers/data/tokenizers/thwiki-20200820/newmm/min-freq-4/newmm.json  \
 --ext txt \
 --train_dir /workspace/thai2transformers/data/dataset/thwiki-20200820/5_split/train \
 --eval_dir /workspace/thai2transformers/data/dataset/thwiki-20200820/5_split/val \
 --fp16 \
 --max_seq_length 64 \
 --learning_rate 5e-4 --weight_decay 0.01 \
 --adam_beta2 0.98 \
 --adam_epsilon 1e-6 \
 --max_steps 100 \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 16 \
 --warmup_steps 10 \
 --seed 1234 \
 --save_steps 1000 \
 --logging_steps 5 \
 --save_total_limit 10 \
 --evaluation_strategy steps \
 --eval_steps 10 \
 --prediction_loss_only \
 --logging_dir /workspace/thai2transformers/logs/roberta/thwiki-20200820/v1 \
 --output_dir /workspace/thai2transformers/checkpoint/roberta/thwiki-20200820/v1 \
 --add_space_token \
 --datasets_cache_dir /workspace/thai2transformers/data/dataset/thwiki-20200820/6_cache \
 --datasets_type MemmapConcatFullSentenceTextDataset \
 --tokenizer_type newmm
```

<details>
<summary>Example output:</summary>

```
```

</details>



