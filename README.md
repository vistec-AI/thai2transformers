# `thai2transformers`
Pretraining transformers in Thai and English



## Installation 

### 1) Manual installation

1) PyTorch

In this repository, we use PyTorch as a framework to train langauage model. The version of PyTorch that we used is 1.7.0 with CUDA 10.2.

```
pip install torch==1.7.0
```

2) SentencePiece

In order to manually build SentencePiece model from raw text files, it is required to install SentencePiece from source. (ref: https://github.com/google/sentencepiece#c-from-source)

```
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev

git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
```
On OSX/macOS, replace the last command with sudo update_dyld_shared_cache


To use trained SentencePiece model, you can only install sentencepice via pip.

```
pip install sentencepiece==0.1.94
```


3) Huggingface's `transformers` 


Currently, we use the library from huggingface.co namely [transformers](https://github.com/huggingface/transformers) to pretrain our Thai language models.

`transformers` can be installed via pip. (the version of transformers we used is 3.4.0)

```
pip install transformers==3.4.0
```


For faster training on GPUs with PyTorch (0.4 or newer), install Nvidia's apex library (https://github.com/NVIDIA/apex). apex can be installed with CUDA and C++ extensions (for performance and full functionality).

```
git clone https://github.com/NVIDIA/apex.git
cd apex

pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```



## Pretraining Datasets

Developing. See this [spreadsheet](https://docs.google.com/spreadsheets/d/1lQ06FT2RvBE8twKzvXeSe4w5CHnU29f8ZWMUcJdmRks/edit?usp=sharing). Download current version of cleaned datasets [here](https://drive.google.com/file/d/1oF7_COZJqGdIaDGMNI1rKdDCOEzVoZHq/view?usp=sharing).


## Preprocess dataset:

### Text cleaning/filtering

The text preprocessing rules applied are as follows:

Filtering rules:
- Drop rows with NaN/Na
- Drop rows with no Thai character

Cleaning rules:
- Replace Non-breaking space with an space token
- Remove soft-hyphen
- Remove zero width non-breaking space

Example:
```
python clean_data.py \
../cleaned_data/thwiki.csv \
../dataset/cleaned/thwiki-for-ddp_6.11.2020/thwiki.csv \
--drop_na \
--break_long_sentence \
--max_sentence_length 300 \
--drop_no_thai_char \
--min_newmm_token_len 4 \
--max_newmm_token_len 300
```


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