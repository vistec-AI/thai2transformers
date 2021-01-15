
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

- `--metric_for_best_model`: The metric to select the best model based on validation set (default: `f1_micro`)

- `--greater_is_better`: The criteria to select the best model according to the specified metric either by expecting the greater value or lower value (default: `True`)

- `--logging_steps` : In interval of training steps to perform logging  (default: `10`)

- `--seed` : The seed value (default: `2020`)

- `--fp16_opt_level` : The OPT level for FP16 mixed-precision training (default: `O1`)

- `--gradient_accumulation_steps` : The number of steps to accumulate gradients (default: `1`, no gradient accumulation)

- `--adam_epsilon` : Value of Adam epsilon (default: `1e-05`)

- `--max_grad_norm` : Value of gradient norm (default: `1.0`)

- `--lowercase`     :  Append "--lowercase" to convert all input texts to lowercase as some model may support only uncased texts (default: 
`False`)

- `--run_name`     :  Specify the **run_name** for logging experiment to wandb.com (default: `False`)