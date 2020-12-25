import argparse
import numpy as np
import pandas as pd
import csv
import os
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--train_ratio', type=float, default=0.95)
    parser.add_argument('--val_ratio', type=float, default=0.025)
    parser.add_argument('--test_ratio', type=float, default=0.025)
    
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio + args.test_ratio  != 1.0:

        raise "Summation of train/val/test ration is not eqaul to 1.0"

    print(f'INFO: Load text file from {args.input_path}')

    with open(args.input_path, 'r', encoding='utf-8') as f:
        df = pd.DataFrame({ 'text': f.readlines() })

    print('INFO: Begin splitting data.')
    
    print(f'\ttrain_ratio: {args.train_ratio}')
    print(f'\tval_ratio: {args.val_ratio}')
    print(f'\ttest_ratio: {args.test_ratio}')
    
    train_df, valid_df, test_df = np.split(df, [math.ceil(args.train_ratio * len(df)), math.ceil((args.train_ratio + args.val_ratio )*len(df))])
    
    print(f'\nINFO: Train/val/test statistics.')
    print(f'\ttrain set: {train_df.shape[0]}')
    print(f'\tval set: {valid_df.shape[0]}')
    print(f'\ttest set: {test_df.shape[0]}')

    print('')
    split_df = { 'train': train_df, 'val': valid_df, 'test': test_df }
    for split in ['train', 'val', 'test']:

        print(f'INFO: Begin writing {split} split to "{args.output_dir}/{split}/{split}.txt".')
        if os.path.exists(f'{args.output_dir}/{split}') == False:
            os.makedirs(f'{args.output_dir}/{split}', exist_ok=True)

        with open(f'{args.output_dir}/{split}/{split}.txt', 'w', encoding='utf-8') as f:
            texts = split_df[split]['text'].tolist()
            f.writelines(texts)

    print('\nINFO: Done writing all split.')