# Training tokenizer

This step train a tokenizer (output a vocaulary file) so it can be use later. Currently, there are multiple tokenizer that can be trained using `scripts/train_tokenizer.py`. This all done by cutting up text using predefined model and make a vocabulary with specified constrain, such as minimum frequency of word found in dataset.

## Type of tokenizer

 1. newmm - Dictionary-based word-level maximal matching tokenizer from [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)
 2. syllable - Syllable-level tokenizer from CRF-basd syllable segmentor for Thai ([ssg](https://github.com/ponrawee/ssg))
 3. fake_sefr_cut - ML-based word-level tokenizer from "Stacked Ensemble Filter and Refine for Word Segmentation" ([seft-cut](https://github.com/mrpeerat/SEFR_CUT)). In this configuration, the texts are required to be pretokenized with SEFR tokenizer, and it will split tokens by `SEFR_SPLIT_TOKEN` which is equivalent to `<|>`.
 4. spm - Subword-level tokenizer trained from [SentencePiece](https://github.com/google/sentencepiece) library.

</br>

## Instruction

a) Syllable-level and word-level tokenizer (`newmm`, `syllable`, `fake_sefr_cut`)

The following command can be used to train a tokenizer. Append `--help` after the `train_tokenizer.py` to get more information.

```bash
python3 train_tokenizer.py \
 --ext txt \
 --train_dir "$PROJECT_TRAIN_DATASET_DIR" \
 --eval_dir "$PROJECT_EVAL_DATASET_DIR" \
 --output_file "$PROJECT_TOKENIZER_PATH/$PROJECT_PRE_TOKENIZER_TYPE.json" \
 --pre_tokenizer_type "$PROJECT_PRE_TOKENIZER_TYPE" \
 --overwrite_output_file \
 --vocab_min_freq "$PROJECT_VOCAB_MIN_FREQ"
```

The command above will read `*.txt` file in the directory `$PROJECT_TRAIN_DATASET_DIR` line by line, strip text, and ignore empty line. Then, it tokenizes each line and count word occurences. Finally, it filters out words which has word occurences less than the threshold `$PROJECT_VOCAB_MIN_FREQ` in the training corpus. After the filering process, it write the vocabulary and their coresponding ids to `$PROJECT_TOKENIZER_PATH/$PROJECT_PRE_TOKENIZER_TYPE.json`.

b) Subword-level tokenizer (`spm`)

If the sentencepiece library is already installed, SentencePiece model can be built by the following command.

```
mkdir -p ./data/dataset/tokenizers/thwiki-20200820/spm/vs-24000
cd ./data/dataset/tokenizers/thwiki-20200820/spm/vs-24000
spm_train \
--input=./data/dataset/thwiki-20200820/5_split/train/train.txt \
--model_prefix sentencepiece.bpe \
--vocab_size=24000 \
--character_coverage=0.9998 --user_defined_symbols="<mask>,<_>" \
--max_sentencepiece_length=10 \
--add_dummy_prefix False \
--bos_id=0 \
--pad_id=1 \
--eos_id=2 \
--unk_id=3 \
--max_sentence_length 10000
```

The script will train SentencePiece model based on training corpus located at `./data/dataset/thwiki-20200820/5_split/train/train.txt` with the parameters specified (e.g. vocabulary size, character coverage) and write two files (`sentencepiece.bpe.model` and `sentencepiece.bpe.model`) to the directory: `./data/dataset/tokenizers/thwiki-20200820/spm/vs-24000`