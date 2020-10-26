import glob
import multiprocessing
import json
from thai2transformers.preprocess import process_transformers
from pythainlp.tokenize import word_tokenize, sent_tokenize
nb_cores = multiprocessing.cpu_count()
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import logging
import sentencepiece as spm
from functools import partial

logging.basicConfig(level=logging.INFO)

# argparse
import argparse

# python3 preprocess_pantip_large.py --input_dir raw_data/pantip-large --output_dir cleaned_data/pantip-large-cleaned

_TOKENIZER = word_tokenize
_TOKENIZER_NAME = 'newmm'

def process_one_pantip(text_list, min_seq_length=5, max_seq_length=300, sep_func=sent_tokenize):
    word_counts = []
    texts = []
    for text in text_list:
        text = text.strip()
        word_count = len(word_tokenize(text))
        if word_count > max_seq_length:
            sub_text = [process_transformers(i) for i in sep_func(text)]
            sub_word_count = [len(word_tokenize(i)) for i in sub_text]
            texts+=sub_text
            word_counts+=sub_word_count
        else:
            texts.append(process_transformers(text))
            word_counts.append(word_count)
    return pd.DataFrame({"text": texts, "wc": word_counts})

def process_fname_pantip(fname, min_seq_length=5, max_seq_length=300, max_lines=1000000):
    line_count=0
    reses=[]
    with open(fname,'r') as f:
        for line in tqdm(f):
            if line_count > max_lines: break  
            try:
                res = process_one_pantip([json.loads(line)['comment']])
                reses.append(res)
                line_count+=1
            except:
#                 print('Not comment')
                try:
                    res = process_one_pantip([json.loads(line)['content']])
                    reses.append(res)
                    line_count+=1
                except:
#                     print('Neither comment nor content')
                    print(json.loads(line))
    df = pd.concat(reses)
    df['text'] = df.text.map(lambda x: x.split('แก้ไขข้อความเมื่อ')[0])
    df = df[(df.wc >= min_seq_length) & (df.wc <= max_seq_length)]
    return df




def main():
    # argparser
    parser = argparse.ArgumentParser(
        prog="preprocess_pantip_large.py",
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
    parser.add_argument("--tokenizer", type=str, default='newmm', help='Tokenizer for sentence length filtering, specify either "newmm" or "spm"')
    parser.add_argument("--spm_model_path", type=str, help='Specify path of SentencePiece model when arg:tokenizer is "spm"')
    parser.add_argument("--ext", type=str, default="")

    args = parser.parse_args()

    if args.tokenizer.lower() == "spm":
        sp = spm.SentencePieceProcessor(model_file=args.spm_model_path)
        repr_space_token_fn = lambda x: x.replace(' ', '<th_roberta_space_token>')
        _TOKENIZER =lambda x: sp.encode(repr_space_token_fn(x), out_tyoe=str)
        _TOKENIZER_NAME = 'spm'
    
    print(f"{args.input_dir}/*{args.ext}")
    fnames = [str(x) for x in glob.glob(f"{args.input_dir}/*{args.ext}")]

    with multiprocessing.Pool(nb_cores) as pool:
        results = pool.map(process_fname_pantip, fnames)

    result_df = pd.concat(results).drop_duplicates().reset_index(drop=True)
    pd.Series(result_df.text.unique()).to_csv(f'{args.output_dir}/pantip-large.tok-{_TOKENIZER_NAME}_min-{args.min_seq_length}_max-{args.max_seq_length}.txt',
                                                index=False,
                                                header=None)

if __name__ == "__main__":
    main()
