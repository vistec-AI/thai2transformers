# Tokenizer training

This step is for building the vocabulary for tokenizer. Currently, there are 4 tokenizers that can be trained with `scripts/train_tokenizer.py`. This all done by segmenting text using predefined model and make a vocabulary with specified constrain which is the minimum number of word occurences found in the training corpus.

## Type of tokenizers

 1. newmm - Dictionary-based word-level maximal matching tokenizer from [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)
 2. __syllable__: a dictionary-based Thai syllable tokenizer based on maximal matching from [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp). The list of syllables used is from [pythainlp/corpus/syllables_th.txt](https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/corpus/syllables_th.txt).
 3. fake_sefr_cut - ML-based word-level tokenizer from "Stacked Ensemble Filter and Refine for Word Segmentation" ([seft-cut](https://github.com/mrpeerat/SEFR_CUT)). In this configuration, the texts are required to be pretokenized with SEFR tokenizer, and it will split tokens by `SEFR_SPLIT_TOKEN` which is equivalent to `<|>`.
 4. spm - Subword-level tokenizer trained from [SentencePiece](https://github.com/google/sentencepiece) library.

</br>

## Instruction

a) Syllable-level and word-level tokenizer (`newmm`, `syllable`, `fake_sefr_cut`)

The following command can be used to train a tokenizer (Append `--help` after the `run_mlm.py` to get more information).

```bash
python ./scripts/train_tokenizer.py \
--ext txt \
--train_dir "$PROJECT_TRAIN_DATASET_DIR" \
--output_file "$PROJECT_TOKENIZER_PATH/$PROJECT_PRE_TOKENIZER_TYPE.json" \
--pre_tokenizer_type "$PROJECT_PRE_TOKENIZER_TYPE" \
--overwrite_output_file \
--vocab_min_freq "$PROJECT_VOCAB_MIN_FREQ"
```

The command above will read `*.txt` file in the directory `$PROJECT_TRAIN_DATASET_DIR` line by line, strip text, and ignore empty line. Then, it tokenizes each line and count word occurences. Finally, it filters out words which has word occurences less than the threshold `$PROJECT_VOCAB_MIN_FREQ` in the training corpus. After the filering process, it write the vocabulary and their coresponding ids to `$PROJECT_TOKENIZER_PATH/$PROJECT_PRE_TOKENIZER_TYPE.json`.

For instance:

```bash
python ./scripts/train_tokenizer.py \
--ext txt \
--train_dir /workspace/thai2transformers/data/dataset/thwiki-20200820/5_split/train/ \
--output_file /workspace/thai2transformers/data/dataset/tokenizers/thwiki-20200820/newmm/newmm.json \
--pre_tokenizer_type newmm \
--overwrite_output_file \
--vocab_min_freq 4
```

b) Subword-level tokenizer (`spm`)

If the sentencepiece library is already installed, SentencePiece model can be built by the following command.

```
mkdir -p /workspace/thai2transformers/data/dataset/tokenizers/thwiki-20200820/spm/vs-24000
cd /workspace/thai2transformers/data/dataset/tokenizers/thwiki-20200820/spm/vs-24000

spm_train \
--input=/workspace/thai2transformers/data/dataset/thwiki-20200820/5_split/train/train.txt \
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

The script will train SentencePiece model based on training corpus located at `/workspace/thai2transformers/data/dataset/thwiki-20200820/5_split/train/train.txt` with the parameters specified (e.g. vocabulary size, character coverage) and write two files (`sentencepiece.bpe.model` and `sentencepiece.bpe.model`) to the directory: `/workspace/thai2transformers/data/dataset/tokenizers/thwiki-20200820/spm/vs-24000`