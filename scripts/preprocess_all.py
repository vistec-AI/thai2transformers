from functools import partial
import glob
import multiprocessing
nb_cores = multiprocessing.cpu_count()
from thai2transformers.preprocess import process_transformers
from pythainlp.tokenize import word_tokenize
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

# argparse
import argparse

# python3 preprocess_all.py --input_dir raw_data/wisesight-large --output_dir cleaned_data/wisesight-large-cleaned

def process_one(fname, min_seq_length, max_seq_length):
    with open(fname, "r") as f:
        line_count = 0
        word_counts = []
        texts = []
        for line in tqdm(f):
            text = line.strip()
            texts.append(process_transformers(text))
            word_counts.append(len(word_tokenize(text)))
            line_count += 1
    df = pd.DataFrame({"text": texts, "wc": word_counts})
    print(np.percentile(df.wc, 5), np.percentile(df.wc, 95))
    df = df[(df.wc >= min_seq_length) & (df.wc <= max_seq_length)]
    return list(df.text)

def preprocess_fname(fname, min_seq_length=5, max_seq_length=300, output_dir=None):
    print(f"Processing {fname}")
    texts = process_one(
        fname,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
    )
    output_fname = f"{output_dir}/{fname.split('/')[-1].split('.')[0]}.txt"
    print(f"Saving to {output_fname}")
    with open(output_fname, "w") as f:
        f.writelines([f"{i}\n" for i in texts])


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
    
    with multiprocessing.Pool(nb_cores) as pool:
        results = pool.map(partial(preprocess_fname,
                                   min_seq_length=args.min_seq_length,
                                   max_seq_length=args.max_seq_length,
                                   output_dir=args.output_dir), fnames)


if __name__ == "__main__":
    main()
