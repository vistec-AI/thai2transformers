import re
from datasets import load_dataset, concatenate_datasets
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

#see if there are contextually similar sentences using USE
#code adapted from https://github.com/cstorm125/thxxwiki/blob/main/align_sentences.py

# !pip install tensorflow==2.3.0 tensorflow_text tensorflow_hub
import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf  # tensorflow 2.3.0

_model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    )

def match_sentences(lang1_sentences, lang2_sentences, model):
    embedding_1 = model(lang1_sentences)
    embedding_2 = model(lang2_sentences)
    distance_matrix_12 = tf.matmul(embedding_1, embedding_2, transpose_b=True)
    print(embedding_1.shape, embedding_2.shape, distance_matrix_12.shape)
    best_distances = tf.argmax(distance_matrix_12, axis=1).numpy()

    matched_sentences_lang2 = []
    scores = []
    for i, lang2_idx in enumerate(best_distances):
        score = distance_matrix_12[i][lang2_idx].numpy()
        scores.append(score)
        matched_sentences_lang2.append(lang2_sentences[lang2_idx])
    return matched_sentences_lang2, scores

def get_overlaps(sent_seed, sent_compare, bs=1000, use_thres=0.8):
    dfs = []
    for i in tqdm(range(len(sent_seed) // bs + 1)):
        for j in tqdm(range(len(sent_compare) // bs + 1)):
            matched_sentences, scores = match_sentences(
                sent_seed[i * bs : (i + 1) * bs],
                sent_compare[j * bs : (j + 1) * bs],
                _model,
            )
            df = pd.DataFrame(
                {
                    "iapp_context": sent_seed[i * bs : (i + 1) * bs],
                    "thaiqa_context": matched_sentences,
                    "use_score": scores,
                }
            )
            df = df[(df.use_score > use_thres)]
            dfs.append(df)
            print(
                f"{df.shape[0]} sentences above {use_thres} threshold"
            )
    df_cc = pd.concat(dfs).sort_values('use_score',ascending=False).reset_index(drop=True)
    overlaps = df_cc.thaiqa_context.tolist()
    print(f'{len(overlaps)} overlapping sentences')
    return overlaps

#filter out overlappers
def filter_overlaps(example, overlaps):
    return False if example['context'] in overlaps else True

#make thaiqa looks like iapp
def convert_thaiqa_to_iapp(example):
    extra_tag = re.match('<doc.*>', example['context'][:-7]).group(0)
    example['answers'] = {
        'text': example['answers']['answer'],
        'answer_start': [np.int32(example['answers']['answer_begin_position'][0] - len(extra_tag))],
        'answer_end': [np.int32(example['answers']['answer_end_position'][0] - len(extra_tag))],
    }
    example['context'] = example['context'][len(extra_tag):-7]
    example['article_id'] = str(example['article_id'])
    example['question_id'] = str(example['question_id'])
    example['title'] = ''
    return example

#make xquad looks like iapp
def convert_xquad_to_iapp(example):
    example['answers']['answer_start'] = [np.int32(example['answers']['answer_start'][0])]
    example['answers']['answer_end'] = [np.int32(example['answers']['answer_start'][0] + len(example['answers']['text'][0]))]
    example['article_id'] = str(example['context'][:30]) #no article id provided to using first 30 characters of context
    example['question_id'] = str(example['id'])
    example['title'] = ''
    example.pop('id', None)
    return example

#load and convert datasets
thaiqa = load_dataset('thaiqa_squad')
thaiqa2 = thaiqa.map(convert_thaiqa_to_iapp)

xquad = load_dataset('xquad','xquad.th')
xquad2 = xquad.map(convert_xquad_to_iapp)

iapp = load_dataset('iapp_wiki_qa_squad')

#see if there is an exact match
iapp_contexts = set(iapp['validation']['context'] + iapp['test']['context'])
thaiqa2_contexts = set(thaiqa2['train']['context'])
xquad2_contexts = set(xquad2['validation']['context'])
print('number of contexts each:',len(iapp_contexts), len(thaiqa2_contexts), len(xquad2_contexts))
print('number of contexts exact matches:',iapp_contexts.intersection(thaiqa2_contexts), iapp_contexts.intersection(xquad2_contexts))

sent_iapp = list(iapp_contexts)
sent_thaiqa = list(thaiqa2_contexts)
sent_xquad = list(xquad2_contexts)

overlaps_thaiqa = get_overlaps(sent_iapp, sent_thaiqa, bs=1000, use_thres=0.8)
overlaps_xquad = get_overlaps(sent_iapp, sent_xquad, bs=1000, use_thres=0.8)
print('number overlapping contexts: ', len(overlaps_thaiqa), len(overlaps_xquad))

#remove overlaps
thaiqa3 = thaiqa2.filter(lambda x: filter_overlaps(x,overlaps_thaiqa))
xquad3 = xquad2.filter(lambda x: filter_overlaps(x,overlaps_xquad))

#concatenate them together
datasets = iapp
datasets['train'] = concatenate_datasets([datasets['train'],thaiqa3['train'],xquad3['validation']])

#save to disk
datasets.save_to_disk('iapp_thaiqa_xquad')