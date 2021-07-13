# -*- coding: utf-8 -*-
"""
Preprocessing for thai2transformers
"""
from typing import Collection, Callable
from functools import partial
import re
import html
import emoji
from pythainlp.tokenize import word_tokenize


_TK_UNK, _TK_REP, _TK_WREP, _TK_URL, _TK_END = "<unk> <rep> <wrep> <url> </s>".split()

SPACE_SPECIAL_TOKEN = "<_>"

# str->str rules
def fix_html(text: str) -> str:
    """
        List of replacements from html strings in `test`. (code from `fastai`)
        :param str text: text to replace html string
        :return: text where html strings are replaced
        :rtype: str
        :Example:
            >>> fix_html("Anbsp;amp;nbsp;B @.@ ")
            A & B.
    """
    re1 = re.compile(r"  +")
    text = (
        text.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(text))


def replace_url(text: str) -> str:
    """
        Replace url in `text` with TK_URL (https://stackoverflow.com/a/6041965)
        :param str text: text to replace url
        :return: text where urls  are replaced
        :rtype: str
        :Example:
            >>> replace_url("go to https://github.com")
            go to <url>
    """
    URL_PATTERN = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    return re.sub(URL_PATTERN, _TK_URL, text)


def rm_brackets(text: str) -> str:
    """
        Remove all empty brackets and artifacts within brackets from `text`.
        :param str text: text to remove useless brackets
        :return: text where all useless brackets are removed
        :rtype: str
        :Example:
            >>> rm_brackets("hey() whats[;] up{*&} man(hey)")
            hey whats up man(hey)
    """
    # remove empty brackets
    new_line = re.sub(r"\(\)", "", text)
    new_line = re.sub(r"\{\}", "", new_line)
    new_line = re.sub(r"\[\]", "", new_line)
    # brakets with only punctuations
    new_line = re.sub(r"\([^a-zA-Z0-9ก-๙]+\)", "", new_line)
    new_line = re.sub(r"\{[^a-zA-Z0-9ก-๙]+\}", "", new_line)
    new_line = re.sub(r"\[[^a-zA-Z0-9ก-๙]+\]", "", new_line)
    # artifiacts after (
    new_line = re.sub(r"(?<=\()[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", new_line)
    new_line = re.sub(r"(?<=\{)[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", new_line)
    new_line = re.sub(r"(?<=\[)[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", new_line)
    # artifacts before )
    new_line = re.sub(r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\))", "", new_line)
    new_line = re.sub(r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\})", "", new_line)
    new_line = re.sub(r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\])", "", new_line)
    return new_line


def replace_newlines(text: str) -> str:
    """
        Replace newlines in `text` with spaces.
        :param str text: text to replace all newlines with spaces
        :return: text where all newlines are replaced with spaces
        :rtype: str
        :Example:
            >>> rm_useless_spaces("hey whats\n\nup")
            hey whats  up
    """

    return re.sub(r"[\n]", " ", text.strip())


def rm_useless_spaces(text: str) -> str:
    """
        Remove multiple spaces in `text`. (code from `fastai`)
        :param str text: text to replace useless spaces
        :return: text where all spaces are reduced to one
        :rtype: str
        :Example:
            >>> rm_useless_spaces("oh         no")
            oh no
    """
    return re.sub(" {2,}", " ", text)

def replace_spaces(text: str, space_token: str = SPACE_SPECIAL_TOKEN) -> str:
    """
        Replace spaces with _
        :param str text: text to replace spaces
        :return: text where all spaces replaced with _
        :rtype: str
        :Example:
            >>> replace_spaces("oh no")
            oh_no
    """
    return re.sub(" ", space_token, text)


def replace_rep_after(text: str) -> str:
    """
    Replace repetitions at the character level in `text`
    :param str text: input text to replace character repetition
    :return: text with repetitive tokens removed.
    :rtype: str
    :Example:
        >>> text = "กาาาาาาา"
        >>> replace_rep_after(text)
        'กา'
    """

    def _replace_rep(m):
        c, cc = m.groups()
        return f"{c}"

    re_rep = re.compile(r"(\S)(\1{3,})")
    return re_rep.sub(_replace_rep, text)


# List[str] -> List[str] rules
def ungroup_emoji(toks: Collection[str]) -> Collection[str]:
    """
    Ungroup Zero Width Joiner (ZVJ) Emojis
    See https://emojipedia.org/emoji-zwj-sequence/
    :param Collection[str] toks: list of tokens
    :return: list of tokens where emojis are ungrouped
    :rtype: Collection[str]
    :Example:
        >>> toks = []
        >>> ungroup_emoji(toks)
        []
    """
    res = []
    for tok in toks:
        if emoji.emoji_count(tok) == len(tok):
            res.extend(list(tok))
        else:
            res.append(tok)
    return res


def replace_wrep_post(toks: Collection[str]) -> Collection[str]:
    """
    Replace reptitive words post tokenization;
    fastai `replace_wrep` does not work well with Thai.
    :param Collection[str] toks: list of tokens
    :return: list of tokens where repetitive words are removed.
    :rtype: Collection[str]
    :Example:
        >>> toks = ["กา", "น้ำ", "น้ำ", "น้ำ", "น้ำ"]
        >>> replace_wrep_post(toks)
        ['กา', 'น้ำ']
    """
    previous_word = None
    rep_count = 0
    res = []
    for current_word in toks + [_TK_END]:
        if current_word == previous_word:
            rep_count += 1
        elif (current_word != previous_word) & (rep_count > 0):
            res += [previous_word]
            rep_count = 0
        else:
            res.append(previous_word)
        previous_word = current_word
    return res[1:]


def remove_space(toks: Collection[str]) -> Collection[str]:
    """
    Do not include space for bag-of-word models.
    :param Collection[str] toks: list of tokens
    :return: Collection of tokens where space tokens (" ") are filtered out
    :rtype: Collection[str]
    :Example:
        >>> toks = ['ฉัน','เดิน',' ','กลับ','บ้าน']
        >>> remove_space(toks)
        ['ฉัน','เดิน','กลับ','บ้าน']
    """
    res = []
    for t in toks:
        t = t.strip()
        if t:
            res.append(t)
    return res


# combine them together
def process_transformers(
    text: str,
    pre_rules: Collection[Callable] = [
        fix_html,
        rm_brackets,
        replace_newlines,
        rm_useless_spaces,
        replace_spaces,
        replace_rep_after,
    ],
    tok_func: Callable = word_tokenize,
    post_rules: Collection[Callable] = [ungroup_emoji, replace_wrep_post],
) -> str:
    text = text.lower()
    for rule in pre_rules:
        text = rule(text)
    toks = tok_func(text)
    for rule in post_rules:
        toks = rule(toks)
    return "".join(toks)

# prepare qa features
def _get_context_span(input_ids,
                     sequence_ids,
                     answers, 
                     start_col, 
                     text_col='text',
                     pad_on_right=True):
    
    # Start/end character index of the answer in the text.
    start_char = answers[start_col][0] 
    end_char = start_char + len(answers[text_col][0]) + 1

    # Start token index of the current span in the text.
    token_start_index = 0
    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
        token_start_index += 1

    # End token index of the current span in the text.
    token_end_index = len(input_ids) - 1
    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
        token_end_index -= 1
    
    return token_start_index, token_end_index, start_char, end_char

def _get_answer_span(tokenized_examples,
                    offsets,
                    start_char,
                    end_char,
                    token_start_index,
                    token_end_index,):

    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
        token_start_index += 1
    start_position = token_start_index - 1
    while offsets[token_end_index][1] >= end_char:
        token_end_index -= 1
    end_position = token_end_index + 1

    return start_position, end_position

def prepare_qa_train_features(examples, 
                           tokenizer,
                           question_col='question',
                           context_col='context',
                           answers_col='answers',
                           start_col='answer_start',
                           text_col='text',
                           pad_on_right=True,
                           max_length=416,
                           doc_stride=128):
    
    tokenized_examples = tokenizer(
        examples[question_col if pad_on_right else context_col],
        examples[context_col if pad_on_right else question_col],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride, #overlapping of overflowing tokens
        return_overflowing_tokens=True, #return multiple input ids if exceeding max_length
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping") #map overflowing examples to original examples
    offset_mapping = tokenized_examples.pop("offset_mapping") #offset map; character index not resetted across overflowing examples

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    #loop through all examples' offset_mapping
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i] #a list of tokens
        cls_index = input_ids.index(tokenizer.cls_token_id) #use cls as answer when there is no answer 
        sequence_ids = tokenized_examples.sequence_ids(i) #sequence_ids 0 for question and 1 for context
        sample_index = sample_mapping[i] #since many examples can point to the same question due to overflowing
        answers = examples[answers_col][sample_index] #answers of each example

        # If no answers are given, set the cls_index as answer
        if len(answers[text_col]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            token_start_index, token_end_index, start_char, end_char = _get_context_span(input_ids=input_ids,
                                                                sequence_ids=sequence_ids,
                                                                answers=answers, 
                                                                start_col=start_col, 
                                                                text_col=text_col,
                                                                pad_on_right=pad_on_right)
            # If answer is not in span, return cls_index as answer
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:                
                start_position, end_position = _get_answer_span(tokenized_examples,
                    offsets,
                    start_char,
                    end_char,
                    token_start_index,
                    token_end_index,)
                tokenized_examples["start_positions"].append(start_position)
                tokenized_examples["end_positions"].append(end_position)
                
    return tokenized_examples

def prepare_qa_validation_features(examples, 
                           tokenizer,
                           question_col='question',
                           context_col='context',
                           question_id_col = 'question_id',
                           pad_on_right=True,
                           max_length=416,
                           doc_stride=128):

    tokenized_examples = tokenizer(
        examples[question_col if pad_on_right else context_col],
        examples[context_col if pad_on_right else question_col],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []
    pass

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples[question_id_col][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples