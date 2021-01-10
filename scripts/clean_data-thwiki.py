import argparse
import os
import re
from pathlib import Path

def replace_nbspace(text: str):
    nbspace = '\xa0'
    cleaned_text = re.sub(nbspace, ' ', text)
    return cleaned_text

def remove_soft_hyphen(text: str):
    soft_hyphen = '\u00ad' # discretionary hyphen 
    cleaned_text = re.sub(soft_hyphen, '', text)
    return cleaned_text

def remove_zero_width_nbspace(text: str):
    zero_width_nbspace = '\ufeff'
    cleaned_text = re.sub(zero_width_nbspace, '', text)
    return cleaned_text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    print(f'Begin reading file from {args.input_path}')

    with open(args.input_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    print('Done.\n')

    print('Apply text cleaning rule 1: Replace non-breaking space with space token.')

    cleaned_texts = list(map(replace_nbspace, texts))
    print('Done.\n')

    print('Apply text cleaning rule 2: Remove invisible characters.')
    cleaned_texts = list(map(remove_zero_width_nbspace, cleaned_texts))
    cleaned_texts = list(map(remove_soft_hyphen, cleaned_texts))
    print('Done.\n')


    print(f'Begin writing file to {args.output_path}')
    output_dir = Path(args.output_path).parent
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_texts)