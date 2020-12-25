import glob
import os
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser
)
from tokenizers import Tokenizer, pre_tokenizers, models


try:
    from thai2transformers.tokenizers import (
        CustomPreTokenizer, WordLevelTrainer,
        ADDITIONAL_SPECIAL_TOKENS,
        PRE_TOKENIZERS_MAP, FakeSefrCustomTokenizer)
except ModuleNotFoundError:
    import sys
    sys.path.append('..')  # path hacking
    from thai2transformers.tokenizers import (
        CustomPreTokenizer, WordLevelTrainer,
        ADDITIONAL_SPECIAL_TOKENS,
        PRE_TOKENIZERS_MAP, FakeSefrCustomTokenizer)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_dir: str = field(
        metadata={"help": "The input training data dir (dir that contain text files)."}
    )  # Non-standard
    eval_dir: str = field(
        metadata={"help": "The input evaluation data dir (dir that contain text files)."},
        default=""
    )  # Non-standard
    ext: str = field(
        default='.txt', metadata={'help': 'extension of training and evaluation files.'}
        )


@dataclass
class CustomOthersArguments:
    pre_tokenizer_type: str = field(
        metadata={'help': 'type of pre-tokenizer.'}
        )
    output_file: str = field(
        metadata={'help': 'tokenizer vocab output file'}
        )
    vocab_size: int = field(
        default=None, metadata={'help': 'size of vocabulary.'}
        )
    vocab_min_freq: int = field(
        default=None, metadata={'help': 'min freq of word in vocab.'}
        )
    overwrite_output_file: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
        )
    debug: bool = field(
        default=False
        )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((DataTrainingArguments, CustomOthersArguments))

    (data_args, custom_args) = parser.parse_args_into_dataclasses()

    train_files = list(sorted(glob.glob(f'{data_args.train_dir}/*.{data_args.ext}')))
    validation_files = list(sorted(glob.glob(f'{data_args.eval_dir}/*.{data_args.ext}')))

    additional_special_tokens = ADDITIONAL_SPECIAL_TOKENS

    pre_tokenizer_func = PRE_TOKENIZERS_MAP.get(custom_args.pre_tokenizer_type, None)
    if pre_tokenizer_func is None:
        raise NotImplementedError
    elif custom_args.pre_tokenizer_type == 'sefr_cut':
        raise ValueError('sefr_cut is slow use fake_sefr_cu with sefr_cut_pre_tokenizer instead')

    if not os.path.exists(custom_args.output_file) or custom_args.overwrite_output_file:
        trainer = WordLevelTrainer(pre_tokenize_func=pre_tokenizer_func,
                                   vocab_size=custom_args.vocab_size,
                                   vocab_min_freq=custom_args.vocab_min_freq,
                                   input_files=train_files,
                                   additional_special_tokens=additional_special_tokens)
        trainer.count_parallel()
        trainer.save_vocab(custom_args.output_file)
    if custom_args.pre_tokenizer_type == 'fake_sefr_cut':
        custom_pre_tokenizer = pre_tokenizers.PreTokenizer.custom(
            FakeSefrCustomTokenizer(PRE_TOKENIZERS_MAP['fake_sefr_cut_keep_split_token']))
    else:
        custom_pre_tokenizer = pre_tokenizers.PreTokenizer.custom(
            CustomPreTokenizer(pre_tokenizer_func))
    tokenizer = Tokenizer(models.WordLevel.from_file(custom_args.output_file, unk_token='<unk>'))
    tokenizer.pre_tokenizer = custom_pre_tokenizer

    if custom_args.debug:
        print('Tokenize following text.')
        texts = ['<s>โรนัลโดเขาได้เล่นกับทีม</s>', 'โปรตุเกสมีโรนัลโด',
                 'โรนัลโดเขาได้เล่นกับทีม\nโปรตุเกสมีโรนัลโด']
        ids = [e.ids for e in tokenizer.encode_batch(texts)]
        decoded_texts = tokenizer.decode_batch(ids)
        decoded_texts = [text.replace(' ', '') for text in decoded_texts]
        for text, i, decoded_text in zip(texts, ids, decoded_texts):
            print('Text: ', text, '>>', 'Tokenized: ', i, '>>', 'Decoded: ', decoded_text)
        with open(validation_files[0], 'r') as f:
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if len(line) > 0 and not line.isspace():
                        encoded = tokenizer.encode(line)
                        decoded = tokenizer.decode(encoded.ids).replace(' ', '')
                        print('Text: ', line, '>>', encoded.ids, '>>', decoded)
                else:
                    break


if __name__ == '__main__':
    main()
