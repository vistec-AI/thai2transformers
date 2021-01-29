import os
import glob
import argparse
import shutil
from thai2transformers.helper import _readline_clean_and_strip, get_file_size


try:
    from thai2transformers.tokenizers import sefr_cut_tokenize, SEFR_SPLIT_TOKEN
except ModuleNotFoundError:
    import sys
    sys.path.append('..')  # path hacking
    from thai2transformers.tokenizers import sefr_cut_tokenize, SEFR_SPLIT_TOKEN


def pre_tokenize_texts(texts, n_jobs=-1):
    """
    Pre tokenize list of string to list of pre_tokenized_string.

    Args:
        texts:
            list of string.
        n_jobs:
            number of multiprocessing cores.

    Returns:
        pre_tokenized_texts:
            list of pretokenized string.

    Examples::

        >>> pre_tokenize_texts(['hello world'])
        ['hello<|><_><|>world']
    """
    list_of_tokens = sefr_cut_tokenize(texts, n_jobs)
    pre_tokenized_texts = [SEFR_SPLIT_TOKEN.join(tokens) for tokens in list_of_tokens]
    return pre_tokenized_texts


def write_output(texts, path):
    """Append text to file at specifed path."""
    with open(path, 'a') as f:
        f.write('\n'.join(texts))


def read_pre_tokenizer_and_write(input_folder, output_folder,
                                 chunk_size, progress=True):
    """
    Read text from file in `input_folder` and pre tokenize texts in chunk and output to specified
    `output_folder`.

    Args:
        input_folder:
            path to input_folder.
        output_folder:
            path to output_folder.
        chunk_size:
            size for chunk.
        progress:
            show progress. Defaults to True.
    """
    files = glob.glob(os.path.join(input_folder, '*.txt'))
    for file in files:
        lines = []
        with open(file, 'r') as f:
            file_size = get_file_size(f)
            for line in _readline_clean_and_strip(f):
                lines.append(line)
                if len(lines) >= chunk_size:
                    write_path = os.path.join(output_folder, os.path.basename(file))
                    write_output(pre_tokenize_texts(lines), write_path)
                    lines = []
                    print(f'\rProcessed {f.tell() / file_size * 100:.2f}%',
                          flush=True, end=' ')
            if lines:
                write_path = os.path.join(output_folder, os.path.basename(file))
                write_output(pre_tokenize_texts(lines), write_path)
                print(f'\rProcessed {f.tell() / file_size * 100:.2f}%',
                      flush=True, end=' ')


def main():
    parser = argparse.ArgumentParser(
        description='pre_tokenize .txt files with sefr_cut into specify path')
    parser.add_argument('--input_folder', type=str,
                        help='input folder which include texts file (.txt)')
    parser.add_argument('--output_folder', type=str,
                        help='output folder for pre-tokenized texts')
    parser.add_argument('--chunk_size', type=int,
                        help='chunk_size for pre-tokenize function')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite pre-tokenized output folder')
    args = parser.parse_args()

    if os.path.exists(args.output_folder):
        print(f'output folder: {args.output_folder} already exists.')
        if args.overwrite:
            shutil.rmtree(args.output_folder)
            print('output folder cleared.')
    os.makedirs(args.output_folder, exist_ok=True)

    read_pre_tokenizer_and_write(args.input_folder, args.output_folder, args.chunk_size)


if __name__ == '__main__':
    main()
