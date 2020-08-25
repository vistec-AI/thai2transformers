import glob
from thai2transformers.preprocess import process_transformers
from pythainlp.tokenize import word_tokenize
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


# argparse
import argparse

# python3 preprocess_corpus.py --input_dir raw_data/wisesight-large --output_dir cleaned_data/wisesight-large-cleaned


def process_one(fname, min_seq_length, max_seq_length):
    with open(fname, "r") as f:
        line_count = 0
        word_counts = []
        texts = []
        for line in tqdm(f):
            text = line.split(",")[1].strip()
            texts.append(process_transformers(text))
            word_counts.append(len(word_tokenize(text)))
            line_count += 1
    df = pd.DataFrame({"text": texts[1:], "wc": word_counts[1:]})
    print(np.percentile(df.wc, 5), np.percentile(df.wc, 95))
    df = df[(df.wc >= min_seq_length) & (df.wc <= max_seq_length)]
    return list(df.text)


def main():
    # argparser
    parser = argparse.ArgumentParser(
        prog="preprocess_corpus.py",
        description="preprocess corpus to be ready to train mlm",
    )

    # required
    parser.add_argument(
        "--input_dir", type=str,
    )
    parser.add_argument(
        "--output_dir", type=str,
    )
    parser.add_argument("--min_seq_length", type=int, default=5)
    parser.add_argument("--max_seq_length", type=int, default=300)
    parser.add_argument("--ext", type=str, default="")

    args = parser.parse_args()
    print(f"{args.input_dir}/*{args.ext}")
    fnames = [str(x) for x in glob.glob(f"{args.input_dir}/*{args.ext}")]

    print(f"There are {len(fnames)} files.")
    for fname in tqdm(fnames):
        print(f"Processing {fname}")
        texts = process_one(
            fname,
            min_seq_length=args.min_seq_length,
            max_seq_length=args.max_seq_length,
        )
        output_fname = f"{args.output_dir}/{fname.split('/')[-1].split('.')[0]}.txt"
        print(f"Saving to {output_fname}")
        with open(output_fname, "w") as f:
            f.writelines([f"{i}\n" for i in texts])


if __name__ == "__main__":
    main()
