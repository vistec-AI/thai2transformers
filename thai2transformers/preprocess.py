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
