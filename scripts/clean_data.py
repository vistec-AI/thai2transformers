
import argparse
import pandas as pd
import swifter
import re
from typing import List, Optional, Dict
from pythainlp.tokenize import word_tokenize, sent_tokenize


def break_long_sentence(text:str, sent_tokenizer=sent_tokenize,
                        word_toknizer=word_tokenize,
                        max_sent_len=300) -> List[str]:
    
    sents = sent_tokenizer(text)
    
    sents_n_toks = [ len(word_toknizer(sent)) for sent in sents ]
    
    groupped_sents = []

    seq_len_counter = 0
    temp_groupped_sent = ''
    for i, sent in enumerate(sents):

        if seq_len_counter + sents_n_toks[i] >= max_sent_len:

            groupped_sents.append(temp_groupped_sent)
            seq_len_counter = 0
            temp_groupped_sent = sent
        else:
            temp_groupped_sent += sent
            seq_len_counter += sents_n_toks[i]

        if i == len(sents) - 1:
            groupped_sents.append(temp_groupped_sent)
            
    return groupped_sents

def drop_na(df):
    return df.dropna(subset=['text'])

def drop_no_thai_char(df):
    return df[df['text'].str.contains(r'[ก-๙]')]

def drop_by_min_max_newmm_tokens(df, min_tokens:int, max_tokens:int):
    return df[(df['nb_tokens'] >= min_tokens) & (df['nb_tokens'] <= max_tokens)]

def strip_text(text: str):
    if type(text) != str:
        return text
    return text.strip()

def replace_nbspace(text: str):
    if type(text) != str:
        return text
    nbspace = '\xa0'
    cleaned_text = re.sub(fr'{nbspace}', ' ', text)
    return cleaned_text

def remove_thwiki_section(text:str):
    if type(text) != str:
        return text
    search_obj = re.search(r'Section::::', text)
    cleaned_text = text
    if search_obj:
        cleaned_text = re.sub(r'^Section::::', '', text)
        cleaned_text = re.sub(r'Section::::', '', text)
        cleaned_text = re.sub(r'\.$', '', cleaned_text)

    return cleaned_text

def remove_soft_hyphen(text: str):
    if type(text) != str:
        return text
    soft_hyphen = '\u00ad' # discretionary hyphen 
    cleaned_text = re.sub(fr'{soft_hyphen}', '', text)
    return cleaned_text

def remove_zero_width_nbspace(text: str):
    if type(text) != str:
        return text
    zero_width_nbspace = '\ufeff'
    cleaned_text = re.sub(fr'{zero_width_nbspace}', '', text)
    return cleaned_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--drop_na', action='store_true', default=True)
    parser.add_argument('--remove_thwiki_section', action='store_true', default=True)
    parser.add_argument('--break_long_sentence', action='store_true', default=True)
    parser.add_argument('--max_sentence_length', type=int, default=300)

    parser.add_argument('--drop_no_thai_char', action='store_true', default=True)
    parser.add_argument('--min_newmm_token_len', type=int, default=4)
    parser.add_argument('--max_newmm_token_len', type=int, default=500)

    parser.add_argument('--space_token', type=str, default='<th_roberta_space_token>')

    args = parser.parse_args()

    print(f'INFO: Load csv file from {args.input_path}')

    df = pd.read_csv(args.input_path)
    
    TEXT_FILTERING_RULES = [drop_na, drop_no_thai_char]
    for fn in TEXT_FILTERING_RULES:
        print(f'INFO: Perform filtering rule: {fn.__name__}')
        print(f'INFO: df.shape (before): {df.shape}')
        df = fn(df)  
        print(f'INFO: df.shape (after): {df.shape}')
        print(f'INFO: Done.')

    print('\nDone all text filtering rules. \n')

    TEXT_CLEANING_RULES = [replace_nbspace, remove_soft_hyphen, remove_zero_width_nbspace, strip_text ]
    if args.remove_thwiki_section:
        TEXT_CLEANING_RULES.append(remove_thwiki_section)

    for fn in TEXT_CLEANING_RULES:
        print(f'INFO: Start cleaning rule: {fn.__name__}')
        print(f'INFO: df.shape (before): {df.shape}')
        df = fn(df)  
        print(f'INFO: df.shape (after): {df.shape}')
        print(f'INFO: Done.')


    print(f'INFO: Write cleaned dataset as csv file to {args.input_path}')
    print(f'INFO: df.columns : {df.columns}')

    print('\nINFO: Done all text cleaning rules. \n')


    print('INFO: Perform sentnece breakdown. ')
    print(f'    max_sentence_length: {args.max_sentence_length}')
    print(f'INFO: df.shape (before): {df.shape}')    

    # split short and long sentences:
    df_short = df[df['nb_tokens'] <= 450]
    long_segments = df[df['nb_tokens'] > 450]['text'].tolist()
    breaked_segments = []
    for s in long_segments:
        breaked_segments += break_long_sentence(s, max_sent_len=args.max_sentence_length)
    print(f'\n\tNumber of long segments: {len(long_segments)}\n\tNumber of new segments: {len(breaked_segments)}')
    nb_tokens = [ len(word_tokenize(s)) for s in breaked_segments ]
    breaked_segments_df = pd.DataFrame({'text': breaked_segments, 'nb_tokens': nb_tokens})
    breaked_segments_df = breaked_segments_df[breaked_segments_df['nb_tokens'] > 0]
    
    df = pd.concat([df, breaked_segments_df])
    print(f'INFO: df.shape (after): {df.shape}') 

    print('\nINFO: Recompute Number of tokens.')
    df['nb_tokens'] = df['text'].swifter.apply(lambda x: len(word_tokenize(x)))

    print('INFO: Done.')

    
    print('\nINFO: Perform sentence length filtering.')
    print(f'    Minimum number of newmm tokens: {args.min_newmm_token_len}')
    print(f'    Maximum number of newmm tokens: {args.max_newmm_token_len}\n')
    print(f'INFO: df.shape (before): {df.shape}')
    df = drop_by_min_max_newmm_tokens(df, min_tokens=args.min_newmm_token_len, max_tokens=args.max_newmm_token_len)  
    print(f'INFO: df.shape (after): {df.shape}')    
    print('\nINFO: Done sentence length filtering.\n')
    

    print(f'\nINFO: Replace space token with {args.space_token}')
    df['text'] = df['text'].apply(lambda x: x.replace(' ', args.space_token))
    print('INFO: Done.>')

    print('INFO: Done.')

    print(f'INFO: Write preprocessed data to {args.output_path}.\n')
    df.to_csv(args.output_path, index=False, encoding='utf-8')
    print(f'INFO: Done.')
