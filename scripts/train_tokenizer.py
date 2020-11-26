import glob
import os
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
)
from tokenizers import Tokenizer, pre_tokenizers, models
from pythainlp.tokenize import word_tokenize, syllable_tokenize
from pythainlp.corpus import thai_syllables, thai_words
from pythainlp.util.trie import Trie
from functools import partial


try:
    from thai2transformers.tokenizers import (
        CustomPreTokenizer, WordLevelTrainer, sefr_pre_token)
except ModuleNotFoundError:
    import sys
    sys.path.append('..')  # path hacking
    from thai2transformers.tokenizers import (
        CustomPreTokenizer, WordLevelTrainer, sefr_pre_token)


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
    )  # Non-standard
    ext: str = field(
        default='.txt', metadata={'help': 'extension of training and evaluation files.'}
        )


@dataclass
class CustomOthersArguments:
    pre_tokenizer_type: str = field(
        metadata={'help': 'type of pre-tokenizer.'}
        )
    vocab_size: int = field(
        metadata={'help': 'size of vocabulary.'}
        )
    output_file: str = field(
        metadata={'help': 'tokenizer vocab output file'}
        )
    overwrite_output_file: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((DataTrainingArguments, CustomOthersArguments))

    (data_args, custom_args) = parser.parse_args_into_dataclasses()

    train_files = list(sorted(glob.glob(f'{data_args.train_dir}/*.{data_args.ext}')))
    validation_files = list(sorted(glob.glob(f'{data_args.eval_dir}/*.{data_args.ext}')))

    additional_special_tokens = ['<bos>', '<pad>', '<eos>', '<unk>', '<mask>', '<_>', '\n']
    pre_tokenizers_map = {'newmm': partial(
        word_tokenize,
        custom_dict=Trie(frozenset(set(thai_words()).union(set(additional_special_tokens))))
        ),
                          'syllable': partial(
        syllable_tokenize,
        custom_dict=Trie(frozenset(set(thai_syllables()).union(set(additional_special_tokens))))
        ),
                          'sefr_cut': sefr_pre_token}

    pre_tokenizer_func = pre_tokenizers_map.get(custom_args.pre_tokenizer_type, None)
    if pre_tokenizer_func is None:
        raise NotImplementedError

    if not os.path.exists(custom_args.output_file) or custom_args.overwrite_output_file:
        trainer = WordLevelTrainer(pre_tokenize_func=pre_tokenizer_func,
                                   vocab_size=custom_args.vocab_size,
                                   input_files=train_files,
                                   additional_special_tokens=additional_special_tokens)
        trainer.count_parallel()
        trainer.save_vocab(custom_args.output_file)

    custom_pre_tokenizer = pre_tokenizers.PreTokenizer.custom(
        CustomPreTokenizer(pre_tokenizer_func))
    tokenizer = Tokenizer(models.WordLevel.from_file(custom_args.output_file, unk_token='<unk>'))
    tokenizer.pre_tokenizer = custom_pre_tokenizer

    print('Tokenize following text.')
    texts = ['โรนัลโดเขาได้เล่นกับทีม', 'โปรตุเกสมีโรนัลโด', 'โรนัลโดเขาได้เล่นกับทีม\nโปรตุเกสมีโรนัลโด']
    ids = [e.ids for e in tokenizer.encode_batch(texts)]
    decoded_texts = tokenizer.decode_batch(ids)
    for text, i, decoded_text in zip(texts, ids, decoded_texts):
        print('Text: ', text, '>', 'Tokenized: ', i, '>', 'Decoded: ', decoded_text)
    breakpoint()


if __name__ == '__main__':
    main()
