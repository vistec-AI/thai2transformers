
## Language Model Finetuning on Token Classification Task

<br>

--------

<br>


We provide a finetuning script (`./scripts/downstream/train_token_classification_lm_finetuning.py`) to finetune our pretrained language model on 3 multiclass classification tasks ( `wisesight_sentiment`, `wongnai_reviews`, `generated_reviews_enth` : review_star ) and 1 multilabel classification task (`prachathai67k`).


The arguements for the `train_sequence_classification_lm_finetuning.py` are as follows:

<br>

**Arguments:**

- `--model_name_or_path` : 

    The pretrained model checkpoint for weights initialization.
    
    Otherwise, specify other public language model (Currently, we support `mbert` and `xlmr` )

- `--tokenizer_name_or_path` : 

    The directory of tokenizer's vocab. Otherwise, 

- `--dataset_name` : 

    Specify the target labels for the token classification datasets. The target labels include `ner_tags` for Named-entity tagging and `pos_tags` for Part-of-Speech tagging.

- `--tokenizer_type` : 

    Specify the type of tokenizer including `ThaiRobertaTokenizer` ,`ThaiWordsNewmmTokenizer`, `ThaiWordsSyllableTokenizer`,
    
    `FakeSefrCutTokenizer`, and `CamembertTokenizer` (for `roberthai-95g-spm`).
    
    Otherwise, use `AutoTokenizer` for public model.

- `--output_dir` : 

    The directory to store the finetuned model checkpoints.

- `--lst20_data_dir` : 

    The directory to the LST20 dataset as `lst20` is required to download manually.

- `--per_device_train_batch_size` :  The train batch size

- `--per_device_eval_batch_size` :  The train batch size

- `--space_token`   :  The custom token that will replace a space token in the texts. As some models use custom space token (default: `"<_>"`). For `mbert` and `xlmr` specify the space token as `" "`.

- `--max_length`: Specify the max length of text inputs to be passed to the model, The max length should be less than the **max positional embedding** or the max sequence length that langauge model was pretrained on.

- `--num_train_epochs`: Number of epochs to finetune model (default: `5`)

- `--learning_rate`: The value of peak learning rate (default: `1e-05`)

- `--weight_decay` : The value of weight decay (default: `0.01`)

- `--warmup_steps`: The number of steps to warmup learning rate (default: `0`)

- `--no_cuda`: Append "--no_cuda" to use only CPUs during finetuning (default: `False`)

- `--fp16`: Append "--fp16" to use FP16 mixed-precision trianing (default: `False`)

- `--metric_for_best_model`: The metric to select the best model based on validation set (default: `loss`)

- `--greater_is_better`: The criteria to select the best model according to the specified metric either by expecting the greater value or lower value (default: `False` if the `metric_for_best_model` is not `"loss"`)

- `--logging_steps` : In interval of training steps to perform logging  (default: `10`)

- `--seed` : The seed value (default: `2020`)

- `--fp16_opt_level` : The OPT level for FP16 mixed-precision training (default: `O1`)

- `--gradient_accumulation_steps` : The number of steps to accumulate gradients (default: `1`, no gradient accumulation)

- `--adam_epsilon` : Value of Adam epsilon (default: `1e-05`)

- `--max_grad_norm` : Value of gradient norm (default: `1.0`)

- `--lowercase`     :  Append "--lowercase" to convert all input texts to lowercase as some model may support only uncased texts (default: 
`False`)

- `--run_name`     :  Specify the **run_name** for logging experiment to wandb.com (default: `False`)

<br>

### Example 

<br>

1. Finetuning `roberthai-95g-spm` model on NER tagging task of `thainer` dataset.

    The following script will finetune the `roberthai-thwiki-spm` pretrained model from checkpoint:7000. 
     
    The script will finetune model with FP16 mixed-precision training on 2 GPUs (ID: 1,2). The train and validation batch size is 16 with no gradient accumulation. The model checkpoint will be save every 250 steps and select the best model by validation loss. During finetuning, the learning rate will be warmed up linearly until `5e-05` for 100 steps, then linearly decay to zero. The maximum sequence length that the model will be passed (from the resuling number of tokens according to the tokenizer specified). Otherwise, it will truncate the sequence to `max_length`. Note that, `--lowercase` is appened to the arugment list as `roberthai-95g-spm` only support uncased text (all lowercase text). Space token is set to `"<th_roberta_space_token>"` as the model use this token for space token.

    ```
    cd ./scripts/downstream
    CUDA_VISIBLE_DEVICES=1,2 python train_token_classification_lm_finetuning.py \
    --tokenizer_type CamembertTokenizer \
    --tokenizer_name_or_path /workspace/checkpoints/roberthai-95g-spm/tokenizer_folder \
    --model_name_or_path /workspace/checkpoints/roberthai-95g-spm/model/checkpoint-320000 \
    --dataset_name thainer \
    --label_name ner_tags \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --warmup_steps 100  \
    --logging_steps 50 \
    --eval_steps 250 \
    --max_steps 1000  \
    --evaluation_strategy steps \
    --output_dir /workspacex/checkpoints/roberthai-95g-spm/finetuned/thainer/ner/v1 \
    --do_train \
    --do_eval \
    --max_length 510 \
    --fp16 \
    --space_token "<th_roberta_space_token>" \
    --lowercase
    ```

    <details>
    <summary>
    Log output:
    </summary>
    
    ```

    01/15/2021 09:45:09 - WARNING - __main__ -   Process rank: -1, device: cuda:0, n_gpu: 2distributed training: False, 16-bits training: True
    01/15/2021 09:45:09 - INFO - __main__ -   Training/evaluation parameters TrainingArguments(output_dir='/workspacex/checkpoints/roberthai-95g-spm/finetuned/thainer/ner/v1', overwrite_output_dir=False, do_train=True, do_eval=True, do_predict=False, evaluate_during_training=False, evaluation_strategy=<EvaluationStrategy.STEPS: 'steps'>, prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=16, per_gpu_train_batch_size=None, per_gpu_eval_batch_size=None, gradient_accumulation_steps=1, eval_accumulation_steps=None, learning_rate=5e-05, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=3.0, max_steps=1000, warmup_steps=100, logging_dir='runs/Jan15_09-45-09_IST-DGX01', logging_first_step=False, logging_steps=50, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=True, fp16_opt_level='O1', local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False, eval_steps=250, dataloader_num_workers=0, past_index=-1, run_name='/workspacex/checkpoints/roberthai-95g-spm/finetuned/thainer/ner/v1', disable_tqdm=False, remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None, greater_is_better=None)
    01/15/2021 09:45:09 - INFO - __main__ -   Data parameters DataTrainingArguments(dataset_name='thainer', label_name='ner_tags', max_length=510)
    01/15/2021 09:45:09 - INFO - __main__ -   Model parameters ModelArguments(model_name_or_path='/workspace/checkpoints/roberthai-95g-spm/model/checkpoint-320000', tokenizer_name_or_path='/workspace/checkpoints/roberthai-95g-spm/tokenizer_folder', tokenizer_type='CamembertTokenizer')
    01/15/2021 09:45:09 - INFO - __main__ -   Custom args CustomArguments(no_train_report=False, no_eval_report=False, no_test_report=False, lst20_data_dir=None, space_token='<th_roberta_space_token>', lowercase=True)
    Model name '/workspace/checkpoints/roberthai-95g-spm/tokenizer_folder' not found in model shortcut name list (camembert-base). Assuming '/workspace/checkpoints/roberthai-95g-spm/tokenizer_folder' is a path, a model identifier, or url to a directory containing tokenizer files.
    Didn't find file /workspace/checkpoints/roberthai-95g-spm/tokenizer_folder/added_tokens.json. We won't load it.
    Didn't find file /workspace/checkpoints/roberthai-95g-spm/tokenizer_folder/special_tokens_map.json. We won't load it.
    Didn't find file /workspace/checkpoints/roberthai-95g-spm/tokenizer_folder/tokenizer_config.json. We won't load it.
    Didn't find file /workspace/checkpoints/roberthai-95g-spm/tokenizer_folder/tokenizer.json. We won't load it.
    loading file /workspace/checkpoints/roberthai-95g-spm/tokenizer_folder/sentencepiece.bpe.model
    loading file None
    loading file None
    loading file None
    loading file None
    01/15/2021 09:45:09 - INFO - __main__ -   [INFO] space_token = `<th_roberta_space_token>`
    Reusing dataset thainer (/root/.cache/huggingface/datasets/thainer/thainer/1.3.0/e0a86672e5ad057c1093708597cdda3671a76e9b053d210a32205406726cca92)
    Loading cached processed dataset at /root/.cache/huggingface/datasets/thainer/thainer/1.3.0/e0a86672e5ad057c1093708597cdda3671a76e9b053d210a32205406726cca92/cache-fac20625c90fe862.arrow
    Loading cached split indices for dataset at /root/.cache/huggingface/datasets/thainer/thainer/1.3.0/e0a86672e5ad057c1093708597cdda3671a76e9b053d210a32205406726cca92/cache-e1c5648ecd5c184a.arrow and /root/.cache/huggingface/datasets/thainer/thainer/1.3.0/e0a86672e5ad057c1093708597cdda3671a76e9b053d210a32205406726cca92/cache-cf0c77b9ce362f6d.arrow
    Loading cached split indices for dataset at /root/.cache/huggingface/datasets/thainer/thainer/1.3.0/e0a86672e5ad057c1093708597cdda3671a76e9b053d210a32205406726cca92/cache-e1f36698c1dabb82.arrow and /root/.cache/huggingface/datasets/thainer/thainer/1.3.0/e0a86672e5ad057c1093708597cdda3671a76e9b053d210a32205406726cca92/cache-0132859955c1ebe7.arrow
    Loading cached split indices for dataset at /root/.cache/huggingface/datasets/thainer/thainer/1.3.0/e0a86672e5ad057c1093708597cdda3671a76e9b053d210a32205406726cca92/cache-6556fccfbcd0cbf4.arrow and /root/.cache/huggingface/datasets/thainer/thainer/1.3.0/e0a86672e5ad057c1093708597cdda3671a76e9b053d210a32205406726cca92/cache-eb99b34850b9ceb8.arrow
    loading configuration file /workspace/checkpoints/roberthai-95g-spm/model/checkpoint-320000/config.json
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
    "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1",
        "2": "LABEL_2",
        "3": "LABEL_3",
        "4": "LABEL_4",
        "5": "LABEL_5",
        "6": "LABEL_6",
        "7": "LABEL_7",
        "8": "LABEL_8",
        "9": "LABEL_9",
        "10": "LABEL_10",
        "11": "LABEL_11",
        "12": "LABEL_12",
        "13": "LABEL_13",
        "14": "LABEL_14",
        "15": "LABEL_15",
        "16": "LABEL_16",
        "17": "LABEL_17",
        "18": "LABEL_18",
        "19": "LABEL_19",
        "20": "LABEL_20",
        "21": "LABEL_21",
        "22": "LABEL_22",
        "23": "LABEL_23",
        "24": "LABEL_24",
        "25": "LABEL_25",
        "26": "LABEL_26",
        "27": "LABEL_27"
    },
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1,
        "LABEL_10": 10,
        "LABEL_11": 11,
        "LABEL_12": 12,
        "LABEL_13": 13,
        "LABEL_14": 14,
        "LABEL_15": 15,
        "LABEL_16": 16,
        "LABEL_17": 17,
        "LABEL_18": 18,
        "LABEL_19": 19,
        "LABEL_2": 2,
        "LABEL_20": 20,
        "LABEL_21": 21,
        "LABEL_22": 22,
        "LABEL_23": 23,
        "LABEL_24": 24,
        "LABEL_25": 25,
        "LABEL_26": 26,
        "LABEL_27": 27,
        "LABEL_3": 3,
        "LABEL_4": 4,
        "LABEL_5": 5,
        "LABEL_6": 6,
        "LABEL_7": 7,
        "LABEL_8": 8,
        "LABEL_9": 9
    },
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "roberta",
    "num_attention_head": 12,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 1,
    "type_vocab_size": 1,
    "vocab_size": 25005
    }

    loading weights file /workspace/checkpoints/roberthai-95g-spm/model/checkpoint-320000/pytorch_model.bin
    Some weights of the model checkpoint at /workspace/checkpoints/roberthai-95g-spm/model/checkpoint-320000 were not used when initializing RobertaForTokenClassification: ['lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.decoder.bias']
    - This IS expected if you are initializing RobertaForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing RobertaForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of RobertaForTokenClassification were not initialized from the model checkpoint at /workspace/checkpoints/roberthai-95g-spm/model/checkpoint-320000 and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    max_steps is given, it will override any value given in num_train_epochs
    The following columns in the training set don't have a corresponding argument in `RobertaForTokenClassification.forward` and have been ignored: old_positions.
    The following columns in the evaluation set don't have a corresponding argument in `RobertaForTokenClassification.forward` and have been ignored: old_positions.
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
    Num examples = 5077
    Num Epochs = 7
    Instantaneous batch size per device = 16
    Total train batch size (w. parallel, distributed & accumulation) = 32
    Gradient Accumulation steps = 1
    Total optimization steps = 1000
    Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"
    wandb: Offline run mode, not syncing to the cloud.
    wandb: W&B syncing is set to `offline` in this directory.  Run `wandb online` to enable cloud syncing.
    0%|          | 0/1000 [00:00<?, ?it/s]
    0%|          | 1/1000 [01:26<23:55:50, 86.24s/it]Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 32768.0
    0%|          | 4/1000 [01:27<8:14:06, 29.77s/it] Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 16384.0
    1%|          | 8/1000 [01:28<2:01:03,  7.32s/it]Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 8192.0
    2%|▎         | 25/1000 [01:44<14:56,  1.09it/s]Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0
    4%|▍         | 44/1000 [01:51<04:05,  3.89it/s]Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 2048.0
    5%|▌         | 50/1000 [01:52<03:41,  4.29it/s]
    {'loss': 1.863900146484375, 'learning_rate': 2.5e-05, 'epoch': 0.31446540880503143}
    10%|█         | 100/1000 [02:18<38:19,  2.55s/it]
    {'loss': 0.46619827270507813, 'learning_rate': 5e-05, 'epoch': 0.6289308176100629}
    {'loss': 0.18838241577148437, 'learning_rate': 4.722222222222222e-05, 'epoch': 0.943396226}
    15%|█▌        | 150/1000 [02:31<03:53,  3.64it/s]
    20%|██        | 200/1000 [02:52<33:10,  2.49s/it]
    {'loss': 0.12642303466796875, 'learning_rate': 4.4444444444444447e-05, 'epoch': 1.25786163 
    20%|██        | 200/1000 [02:52<33:10,  2.49s/it]
    {'loss': 0.1191162109375, 'learning_rate': 4.166666666666667e-05, 'epoch': 1.5723270440251573}
    25%|██▌       | 250/1000 [03:05<03:03,  4.08it/s]
    
    ***** Running Evaluation *****
    Num examples = 635
    Batch size = 32

    01/15/2021 09:48:36 - INFO - /opt/conda/lib/python3.6/site-packages/datasets/metric.py -   Removing /root/.cache/huggingface/metrics/seqeval/default/default_experiment-1-0.arrow
    {'eval_loss': 0.10173556208610535, 'eval_precision': 0.8637927080944737, 'eval_recall': 0.8817883895131086, 'eval_f1': 0.8726977875593652, 'eval_accuracy': 0.9725851004174542, 'epoch': 1.5723270440251573}
    30%|███       | 300/1000 [03:27<16:32,  1.42s/it]{'loss': 0.10812286376953124, 'learning_rate': 3.888888888888889e-05, 'epoch': 1.8867924528301887}                                       
    34%|███▍      | 341/1000 [03:37<02:52,  3.82it/s]
    35%|███▌      | 350/1000 [03:40<03:06,  3.49it/s]
    40%|███▉      | 399/1000 [03:53<02:34,  3.89it/s]
    50%|█████     | 500/1000 [05:12<23:33,  2.83s/it]
    
    ***** Running Evaluation *****
    Num examples = 635
    Batch size = 32

    01/15/2021 09:50:43 - INFO - /opt/conda/lib/python3.6/site-packages/datasets/metric.py -   Removing /root/.cache/huggingface/metrics/seqeval/default/default_experiment-1-0.arrow
    {'eval_loss': 0.08590535074472427, 'eval_precision': 0.878561736770692, 'eval_recall': 0.9094101123595506, 'eval_f1': 0.8937198067632851, 'e 50%|█████     | 500/1000 [05:17<23:33,  2.83s
    /Saving model checkpoint to /workspacex/checkpoints/roberthai-95g-spm/finetuned/thainer/ner/v1/checkpoint-500                                

    Configuration saved in /workspacex/checkpoints/roberthai-95g-spm/finetuned/thainer/ner/v1/checkpoint-500/config.json
    Model weights saved in /workspacex/checkpoints/roberthai-95g-spm/finetuned/thainer/ner/v1/checkpoint-500/pytorch_model.bin
    /opt/conda/lib/python3.6/site-packages/torch/nn/parallel/_functions.py:61: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
    warnings.warn('Was asked to gather along dimension 0, but all '
    53%|█████▎    | 526/1000 [05:32<02:03,  3.83it/s]
    {'loss': 0.0573553466796875, 'learning_rate': 2.5e-05, 'epoch': 3.459119496855346}
    {'loss': 0.05275115966796875, 'learning_rate': 2.2222222222222223e-05, 'epoch': 3.77358490 60%|██████    | 600/1000 [06:05<28:12,  4.23s/it]
    65%|██████▌   | 650/1000 [06:16<01:21,  4.30it/s]{'loss': 0.05139984130859375, 'learning_rate': 1.9444444444444445e-05, 'epoch': 4.08805031                                                  
    {'loss': 0.0428802490234375, 'learning_rate': 1.6666666666666667e-05, 'epoch': 4.40251572327044}
    71%|███████   | 710/1000 [06:43<01:43,  2.81it/s]
    ***** Running Evaluation *****
      Num examples = 635
    Batch size = 32

    {'eval_loss': 0.08580297976732254, 'eval_precision': 0.8899543378995434, 'eval_recall': 0.9124531835205992, 'eval_f1': 0.9010633379565418, 'eval_accuracy': 0.9763442646783942, 'epoch': 4.716981132075472}

                                                    {'loss': 0.037841796875, 'learning_rate': 1.1111111111111112e-05, 'epoch': 5.031446540880503}                                            
    85%|████████▌ | 850/1000 [07:27<00:40,  3.69it/s]3333333333334e-06, 'epoch': 5.345911949685535}
    90%|████████▉ | 899/1000 [07:38<00:25,  3.93it/s]
    95%|█████████▌| 950/1000 [08:09<00:12,  3.89it/s]
    100%|█████████▉| 999/1000 [08:21<00:00,  4.36it/s]

    ```
    </details>


    <details>
    <summary>
    Chunk-level per-class precision, recall, and F1-score on test set.
    </summary>
    
    ```

        Processed: 635 / 635 [ Test Result ]

        {
            'accuracy': 0.980321583662611,
            'f1_macro': 0.9132072525127524,
            'f1_micro': 0.8947951273532668,
            'nb_samples': 635,
            'precision_macro': 0.8956733587500255,
            'precision_micro': 0.8749323226854359,
            'recall_macro': 0.9329612501419587,
            'recall_micro': 0.9155807365439094
        }

                        precision    recall  f1-score   support

                 DATE     0.8955    0.9231    0.9091       195
                EMAIL     1.0000    1.0000    1.0000         1
                  LAW     0.8667    0.8667    0.8667        15
                  LEN     0.8095    0.9444    0.8718        18
             LOCATION     0.8384    0.8913    0.8641       460
                MONEY     0.9804    0.9804    0.9804        51
         ORGANIZATION     0.8731    0.9075    0.8900       584
              PERCENT     0.9333    0.8750    0.9032        16
               PERSON     0.9403    0.9708    0.9553       308
                PHONE     0.8462    0.9167    0.8800        12
                 TIME     0.7714    0.8526    0.8100        95
                  URL     0.8889    1.0000    0.9412         8
                  ZIP     1.0000    1.0000    1.0000         2

        micro avg         0.8749    0.9156    0.8948      1765
        macro avg         0.8957    0.9330    0.9132      1765
        weighted avg      0.8759    0.9156    0.8951      1765
    ```

    </details>