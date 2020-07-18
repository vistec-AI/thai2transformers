# `thai2transformers`
Pretraining transformers in Thai and English

## Pretraining Datasets

Developing. See this [spreadsheet](https://docs.google.com/spreadsheets/d/1lQ06FT2RvBE8twKzvXeSe4w5CHnU29f8ZWMUcJdmRks/edit?usp=sharing). Download current version of cleaned datasets [here](https://drive.google.com/file/d/1oF7_COZJqGdIaDGMNI1rKdDCOEzVoZHq/view?usp=sharing).

## Tran MLM

1. Download and process wikipedia dump

```
bash prepare_wiki_th.sh
```

2. Train tokenizer

```
python train_tokenizer_roberthai.py \
--output_dir path_to_tokenizer_folder \
--vocab_size 52000 --min_frequency 2 \
--train_dir path_to_train_folder \
--ext .tokens

```

3. Train MLM; the paper has 8k batch size and spend 6% of all steps for warmup steps with a total of 500k steps. We can try replicating the settings with gradient accumulation. Refer to [this](https://arxiv.org/pdf/1907.11692.pdf).

```
python train_mlm_roberthai.py\
--tokenizer_name_or_path path_to_tokenizer_folder\
--ext .tokens\
--vocab_size 52000\
--train_dir path_to_train_folder --eval_dir path_to_eval_folder\
--train_max_length 512 --eval_max_length 512\
--learning_rate 6e-4 --weight_decay 0.01\
--adam_epsilon 1e-6\
--fp16 False\
--num_train_epochs 5\
--per_device_train_batch_size 32 --per_device_eval_batch_size 64\
--gradient_accumulation_steps 1 --warmup_steps 500
```