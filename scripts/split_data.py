import argparse
import numpy as np
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--train_ratio', type=float, default=0.95)
    parser.add_argument('--val_ratio', type=float, default=0.025)
    parser.add_argument('--test_ratio', type=float, default=0.025)
    
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio + args.test_ratio  != 1.0:

        raise "Summation of train/val/test ration is not eqaul to 1.0"

    print(f'INFO: Load csv file from {args.input_path}')

    df = pd.read_csv(args.input_path, encoding='utf-8')

    print('INFO: Begin splitting data.')
    print(f'\tseed: {args.seed}')
    print(f'\ttrain_ratio: {args.train_ratio}')
    print(f'\tval_ratio: {args.val_ratio}')
    print(f'\ttest_ratio: {args.test_ratio}')
    
    train_df, valid_df, test_df = np.split(df.sample(frac=1, random_state=args.seed), [int(.95*len(df)), int(.975*len(df))])
    
    print(f'\nINFO: Train/val/test statistics.')
    print(f'\ttrain set: {train_df.shape[0]}')
    print(f'\tval set: {valid_df.shape[0]}')
    print(f'\ttest set: {test_df.shape[0]}')

    print('')
    split_df = { 'train': train_df, 'val': valid_df, 'test': test_df }
    for split in ['train', 'val', 'test']:

        print(f'INFO: Begin writing {split} split to "{args.output_dir}/train/train.txt".')
         
        split_df[split][['text']].dropna().sample(frac=1, random_state=args.seed).to_csv(f'{output_dir}/{split}/{split}.txt',
                                                             encoding='utf-8', sep="\t", index=False, header=None,
                                                             escapechar="", quotechar="", quoting=csv.QUOTE_NONE)

    print('\nINFO: Done writing all split.')