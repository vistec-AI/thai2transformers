# Masked Language Model (MLM) Pretraining

This page describes the steps required to train Thai RoBERTa based on our custom Tokenizer class. The tokenizers we implemented are inherited from `transformers.tokenization_utils.PreTrainedTokenizer`. There are a lot of arguments that can be pass in this step. Our MLM training script is adapted from Huggingface's transformers respository (`examples/language_modeling/run_mlm.py`) and we add a number of new arguments.

## Instruction

The following command can be used to train a masked language model (Append `--help` after the `run_mlm.py` to get more information).

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

Due to the fact that most of the dataset creation does not use GPUs. So to only build datasets and cache it without training we can also use `--build_dataset_only` flags to trigger script to quit before training step.

<br>

For instance, the following command will train `roberta-base` model on 1 GPU (GPU ID: 3) with FP16 mixed-precision training on the `thwiki-20200820` dataset. Maximum sequence length is set to 64. The batch size is 1 with gradient accumulation of 16 steps.  Maximum trianing step is set to 100 steps with 10 warmup steps in which the learning rate is increased linearly to the peak value of `5e-4` and linearly decayed to zero.

```
cd scripts

CUDA_VISIBLE_DEVIDES="3" python run_mlm.py \
 --architecture roberta-base \
 --tokenizer_name_or_path /workspace/thai2transformers/data/tokenizers/thwiki-20200820/newmm/min-freq-4/newmm.json \
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
12/25/2020 10:54:12 - WARNING - __main__ -   Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: True
12/25/2020 10:54:12 - INFO - __main__ -   Training/evaluation parameters TrainingArguments(output_dir='/workspace/thai2transformers/checkpoint/roberta/thwiki-20200820/v1', overwrite_output_dir=False, do_train=False, do_eval=True, do_predict=False, evaluate_during_training=False, evaluation_strategy=<EvaluationStrategy.STEPS: 'steps'>, prediction_loss_only=True, per_device_train_batch_size=1, per_device_eval_batch_size=1, per_gpu_train_batch_size=None, per_gpu_eval_batch_size=None, gradient_accumulation_steps=16, eval_accumulation_steps=None, learning_rate=0.0005, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.98, adam_epsilon=1e-06, max_grad_norm=1.0, num_train_epochs=3.0, max_steps=100, warmup_steps=10, logging_dir='/workspace/thai2transformers/logs/roberta/thwiki-20200820/v1', logging_first_step=False, logging_steps=5, save_steps=1000, save_total_limit=10, no_cuda=False, seed=1234, fp16=True, fp16_opt_level='O1', local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=10, dataloader_num_workers=0, past_index=-1, run_name='/workspace/thai2transformers/checkpoint/roberta/thwiki-20200820/v1', disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None)
Model name '/workspace/thai2transformers/data/tokenizers/thwiki-20200820/newmm/min-freq-4/newmm.json' not found in model shortcut name list (). Assuming '/workspace/thai2transformers/data/tokenizers/thwiki-20200820/newmm/min-freq-4/newmm.json' is a path, a model identifier, or url to a directory containing tokenizer files.
Calling ThaiWordsNewmmTokenizer.from_pretrained() with the path to a single file or url is deprecated
loading file /workspace/thai2transformers/data/tokenizers/thwiki-20200820/newmm/min-freq-4/newmm.json
12/25/2020 10:54:12 - INFO - data_loader -   Creating features from dataset file at /workspace/thai2transformers/data/dataset/thwiki-20200820/5_split/train/train.txt
Processed 100.00% 
12/25/2020 11:28:49 - INFO - data_loader -   Skipped 320284
12/25/2020 11:28:49 - INFO - data_loader -   Creating features from dataset file at /workspace/thai2transformers/data/dataset/thwiki-20200820/5_split/val/val.txt
Processed 100.00% 
12/25/2020 11:29:35 - INFO - data_loader -   Skipped 7921
loading configuration file ../roberta_config/th-roberta-base-config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "eos_token_id": 2,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "mask_token_id": 4,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "type_vocab_size": 1,
  "unk_token_id": 3,
  "vocab_size": 97982
}

max_steps is given, it will override any value given in num_train_epochs
Selected optimization level O1:  Insert automatic casts around Pytorch functions and Tensor methods.

Defaults for this optimization level are:
enabled                : True
opt_level              : O1
cast_model_type        : None
patch_torch_functions  : True
keep_batchnorm_fp32    : None
master_weights         : None
loss_scale             : dynamic
Processing user overrides (additional kwargs that are not None)...
After processing overrides, optimization options are:
enabled                : True
opt_level              : O1
cast_model_type        : None
patch_torch_functions  : True
keep_batchnorm_fp32    : None
master_weights         : None
loss_scale             : dynamic
***** Running training *****
  Num examples = 345348
  Num Epochs = 1
  Instantaneous batch size per device = 1
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 16
  Total optimization steps = 100
  0%|                                                                                                                                             | 0/100 [00:00<?, ?it/s]/opt/conda/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:114: UserWarning: Seems like `optimizer.step()` has been overridden after learning rate scheduler initialization. Please, make sure to call `optimizer.step()` before `lr_scheduler.step()`. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
{'loss': 9.954816436767578, 'learning_rate': 0.00025, 'epoch': 0.0002316503932265425}                                                                                     
{'loss': 9.293991851806641, 'learning_rate': 0.0005, 'epoch': 0.000463300786453085}                                                                                       
 10%|█████████████▏                                                                                                                      | 10/100 [01:09<05:05,  3.39s/it]***** Running Evaluation *****
  Num examples = 8331
  Batch size = 1
{'eval_loss': 8.826310157775879, 'epoch': 0.000463300786453085}                                                                                                           
{'loss': 9.041633605957031, 'learning_rate': 0.00047222222222222224, 'epoch': 0.0006949511796796275}                                                                      
{'loss': 8.147613525390625, 'learning_rate': 0.0004444444444444444, 'epoch': 0.00092660157290617}                                                                         
 20%|██████████████████████████▍                                                                                                         | 20/100 [16:35<17:56, 13.46s/it]***** Running Evaluation *****
  Num examples = 8331
  Batch size = 1
{'eval_loss': 7.698431015014648, 'epoch': 0.00092660157290617}                                                                                                            
{'loss': 7.112191772460937, 'learning_rate': 0.0004166666666666667, 'epoch': 0.0011582519661327126}                                                                       
{'loss': 7.994699096679687, 'learning_rate': 0.0003888888888888889, 'epoch': 0.001389902359359255}                                                                        
 30%|███████████████████████████████████████▌                                                                                            | 30/100 [31:54<15:48, 13.56s/it]***** Running Evaluation *****
  Num examples = 8331
  Batch size = 1
{'eval_loss': 7.177613735198975, 'epoch': 0.001389902359359255}                                                                                                           
{'loss': 7.21585693359375, 'learning_rate': 0.0003611111111111111, 'epoch': 0.0016215527525857976}                                                                        
{'loss': 6.712384033203125, 'learning_rate': 0.0003333333333333333, 'epoch': 0.00185320314581234}                                                                         
 40%|████████████████████████████████████████████████████▊                                                                               | 40/100 [47:23<13:50, 13.84s/it
```

</details>
