#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import glob
import os
from typing import List, Optional
from pydantic import BaseModel
from tqdm import tqdm
import jsonlines
import itertools
import time
import re
import copy
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize as en_sent_tokenize
from pythainlp.tokenize import word_tokenize, sent_tokenize as th_sent_tokenize


class WikiArticle(BaseModel):
    docid: int
    url: str
    title: str
    text: str
    segments: Optional[List[str]]

pattern_parenthesis = r'\s\(([\s,;]|)+\)'
pattern_th = r'[ก-๙]'

def get_list_of_extracted_files(input_dir: str) -> List[str]:
    list_of_extracted_files = []
      
    for sub_dir in glob.glob(os.path.join(input_dir,'*')):
        print(f'Sub directory: {sub_dir}')
        for target_file_path in glob.glob(os.path.join(sub_dir,'*')):
            list_of_extracted_files.append(target_file_path)
    print(f'Total number of files: {len(list_of_extracted_files)}') 
    return list_of_extracted_files
        
def extract_data(list_of_extracted_files: List[str]) -> List[WikiArticle]:
    list_of_thwiki_objs = []
    for file_path in tqdm(list_of_extracted_files):
        with jsonlines.open(file_path, mode='r') as reader:
            for obj in reader:
                wiki_article = WikiArticle(docid=int(obj['id']),
                                        url=obj['url'],
                                        title=obj['title'],
                                        text=obj['text'])
                list_of_thwiki_objs.append(wiki_article)
    return list_of_thwiki_objs

def _extract_segmetns(text: str) -> List[str]:
    newline = '\n'
    double_newlines = '\n\n'
    segments = []
    # split by \n\n
    segments = text.split(double_newlines)
    # Further split by \n
    nested_segments = [ s.split(newline) for s in segments]
    segments = list(itertools.chain(*nested_segments))
   
    segments = list(map(str.strip, segments))
    segments = list(filter(lambda x: len(x) != 0, segments))
    
    # skip fist segment (article title) if found duplicated text
    if len(segments) >=2 and segments[0] == segments[1][0:len(segments[0])]:
        segments = segments[1:]
    return segments

def extract_segments(objs: List[WikiArticle]) ->List[WikiArticle]:
    """Extract segments from WikiArticle.text and store to attribute: `segments`"""
    new_objs = copy.deepcopy(objs)
    for i, obj in tqdm(enumerate(new_objs)):
        new_objs[i].segments = _extract_segmetns(obj.text)
    return new_objs


def _remove_first_empty_parenthesis(text: str) -> str:
    
    """
    for example
    "กรุงเทพมหานคร (;)เป็นองค์กรปกครองส่วนท้องถิ่นรูปแบบพิเศษ ()(1)" -> "กรุงเทพมหานคร เป็นองค์กรปกครองส่วนท้องถิ่นรูปแบบพิเศษ ()(1)" 
    """

    return re.sub(pattern_parenthesis, '', text, 1)

def remove_first_empty_parenthesis(objs: List[WikiArticle]) -> List[WikiArticle]:

    """Remove first empty parenthesis from WikiArticle.segments[0]"""
    new_objs = copy.deepcopy(objs)
    for i, obj in tqdm(enumerate(new_objs)):
        if len(obj.segments) >= 1:
            new_objs[i].segments[0] = _remove_first_empty_parenthesis(obj.segments[0])
    return new_objs

def _group_splitted_segments(splitted_segments: List[str], max_group_seq_len) -> List[str]:
    splitted_segments_n_toks = [ len(word_tokenize(sent)) for sent in splitted_segments ]

    groupped_sents = []

    seq_len_counter = 0
    temp_groupped_sent = ''
    for i, sent in enumerate(splitted_segments):

        if seq_len_counter + splitted_segments_n_toks[i] >= max_group_seq_len:

            groupped_sents.append(temp_groupped_sent)
            seq_len_counter = 0
            temp_groupped_sent = sent
        else:
            temp_groupped_sent += sent
            seq_len_counter += splitted_segments_n_toks[i]

        if i == len(splitted_segments) - 1:
            groupped_sents.append(temp_groupped_sent)
            
    return groupped_sents

def _split_long_segment(segments: List[str],
                        max_seq_len = 400,
                        max_group_seq_len=300) -> List[str]:

    new_segments = []
    for i, segment in enumerate(segments):
        if len(word_tokenize(segment)) > max_seq_len:
            if re.search(pattern_th, segment) == None:
                # For English segment
                new_segments.extend(en_sent_tokenize(segment))
            else:
                # For Thao segment
                splitted_segments = th_sent_tokenize(segment)
                groupped_splitted_segments = _group_splitted_segments(splitted_segments,
                                                                      max_group_seq_len=max_group_seq_len)
                new_segments.extend(groupped_splitted_segments)
        else:
            new_segments.append(segment)
    
    return new_segments

def split_long_segment(objs: List[WikiArticle]) -> List[WikiArticle]:

    """Remove first empty parenthesis from WikiArticle.segments[0]"""
    new_objs = copy.deepcopy(objs)
    for i, obj in tqdm(enumerate(new_objs)):
        
        new_objs[i].segments = _split_long_segment(obj.segments) 
            
    return new_objs

def add_end_of_doc_token(objs: List[WikiArticle]) -> List[WikiArticle]:

    new_objs = copy.deepcopy(objs)
    for i, obj in tqdm(enumerate(new_objs)):
        if len(obj.segments) >= 1:
            new_objs[i].segments[-1] = obj.segments[-1] + '</s></s>'
    return new_objs

def replace_space_token(list_of_thwiki_objs, space_token):
    updated_list_of_thwiki_objs = copy.deepcopy(list_of_thwiki_objs)
    for i, doc in enumerate(updated_list_of_thwiki_objs):
        for j, s in enumerate(doc.segments):
            updated_list_of_thwiki_objs[i].segments[j] = s.replace(' ', space_token)
    return updated_list_of_thwiki_objs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', help="Directory storing extracted wikipedia dump in jsonl format.")
    parser.add_argument('output_dir', help="Output path to write the output")

    parser.add_argument('--remove_first_empty_parenthesis', action='store_true', default=True)
    parser.add_argument('--split_long_segment', action='store_true', default=True)
    parser.add_argument('--add_end_of_doc_token', action='store_true', default=True)
    parser.add_argument('--space_token', type=str, default='<_>')
    
    args = parser.parse_args()
    
    print(f'Begin loading files from {args.input_dir}')
    list_of_extracted_files = get_list_of_extracted_files(args.input_dir)
    print(f'Done.\n')
    print(f'Begin extracting data')
    list_of_thwiki_objs = extract_data(list_of_extracted_files)
    list_of_thwiki_objs_segment_extracted = extract_segments(list_of_thwiki_objs)
    print(f'Done.\n')

    print(f'Argumnet: remove_first_empty_parenthesis = {args.remove_first_empty_parenthesis}, Begin removing first empty parenthesis.')
    if args.remove_first_empty_parenthesis:
        list_of_thwiki_objs_segment_extracted = remove_first_empty_parenthesis(list_of_thwiki_objs_segment_extracted)
    print(f'Done.\n')
    print(f'Argumnet: split_long_segment = {args.remove_first_empty_parenthesis}, Begin spliting long segment.')
    if args.split_long_segment:
        list_of_thwiki_objs_segment_extracted = split_long_segment(list_of_thwiki_objs_segment_extracted)
    print(f'Done.\n')
    print(f'Argumnet: add_end_of_doc_token = {args.remove_first_empty_parenthesis}, Begin adding end of document token `</s></s>`.')
    if args.add_end_of_doc_token:
        list_of_thwiki_objs_segment_extracted = add_end_of_doc_token(list_of_thwiki_objs_segment_extracted)
    print(f'Done.\n')
    
    print(f'Begin replacing space with space token')
    print(f'Argument space_token = {args.space_token}')
    list_of_thwiki_objs_segment_extracted = replace_space_token(list_of_thwiki_objs_segment_extracted,
                                                                space_token=args.space_token)
    print(f'Done.\n')

    print(f'Begin writing result to the file: `{args.output_dir}`')
    lines = []
    for doc in list_of_thwiki_objs_segment_extracted:
        for s in doc.segments:
            lines.append(s)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'thwiki.txt'), 'w', encoding='utf-8') as f:
        for l in lines:
            f.write(l + '\n')
    print(f'Done.\n')
