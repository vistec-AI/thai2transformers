
## Language Model Finetuning on Sequence Classification Task

<br>

--------

<br>
<!-- Currently, the sequence classification finetuning script supports 4 Thai datasets published in Huggingface's datasets including -->
<!-- `wisesight_sentiment`, `wongnai_reviews`, `generated_reviews_enth` and `prachathai67k`. -->

We provide a finetuning script (`./scripts/downstream/train_sequence_classification_lm_finetuning.py`) to finetune our pretrained language model on 3 multiclass classification tasks ( `wisesight_sentiment`, `wongnai_reviews`, `generated_reviews_enth` : review_star ) and 1 multilabel classification task (`prachathai67k`).


The arguements for the `train_sequence_classification_lm_finetuning.py` are as follows:

<br>

Required arguments:

- **tokenizer_type_or_public_model_name** : 

    The token type that RoBERThai used (`spm`, `spm_camembert` (for roberthai-95g-spm), `newmm`, `syllable`, `sefr_cut`). 
    
    If the token type is specified, it is required to specify the directory to model checkpoint and tokenizer via `--model_dir` and `--tokenizer_dir`.
    
    Otherwise, specify other public language model (Currently, we support `mbert` and `xlmr` )

- **dataset_name** : 

    Specify the dataset name to finetune. Currently, sequence classification datasets including `wisesight_sentiment`, `generated_reviews_enth-review_star`, and`wongnai_reviews`.

- **output_dir** : 

    The directory to store finetuned model

- **output_dir** : 

    The directory to logging output including Huggingface's Trainer log, Tensorboard log, and `wandb` log (optional)

<br>

Optional arguments:



- `--model_dir`     :  The directory of pretrained model checkpoint

- `--tokenizer_dir` :  The directory of tokenizer's vocab

- `--space_token`   :  The custom token that will replace a space token in the texts. As some models use custom space token (default: `"<_>"`). For `mbert` and `xlmr` specify the space token as `" "`.

- `--max_length`: Specify the max length of text inputs to be passed to the model, The max length should be less than the **max positional embedding** or the max sequence length that langauge model was pretrained on.

- `--num_train_epochs`: Number of epochs to finetune model (default: `5`)
- `--learning_rate`: The value of peak learning rate (default: `1e-05`)
- `--weight_decay` : The value of weight decay (default: `0.01`)
- `--warmup_ratio`: The ratio of steps / max_steps to warmup learning rate (default: `0.1`; in other word, warm up the learning until the peak valye for the first 10% of the total steps)
- `--batch_size`: The batch size (default: `16`)
- `--batch_size`: The batch size (default: `16`)
- `--batch_size`: The batch size (default: `16`)

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
- `--lowercase`     :  Append "--lowercase" to convert all input texts to lowercase as some model may 
support only uncased texts (default: `False`)
- `--run_name`     :  Specify the **run_name** for logging experiment to wandb.com (default: `False`)