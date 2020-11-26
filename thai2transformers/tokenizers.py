import glob
import json
import multiprocessing
from collections import Counter
from typing import Collection, Callable, Dict
from tokenizers import NormalizedString, PreTokenizedString

_nb_cores = multiprocessing.cpu_count()


class CustomPreTokenizer:
    def __init__(self, pre_tokenize_func: Callable):
        self.pre_tokenize_func = pre_tokenize_func

    def split(
        self, n: int, normalized_string: NormalizedString
    ) -> Collection[NormalizedString]:
        break_i = []
        total_i = 0
        for word in self.pre_tokenize_func(str(normalized_string)):
            total_i += len(word)
            break_i.append(total_i)
        splits = []
        last = 0
        for (i, char) in enumerate(str(normalized_string)):
            if i in break_i:
                splits.append(normalized_string[last:i])
                last = i
        splits.append(normalized_string[last:])
        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.split)


class WordLevelTrainer:
    def __init__(
        self,
        pre_tokenize_func: Callable,
        vocab_size: int,
        input_dir: str,
        additional_special_tokens: Collection[str],
        ext: str = ".txt",
    ):
        self.pre_tokenize_func = pre_tokenize_func
        self.vocab_size = vocab_size
        self.input_dir = input_dir
        self.special_tokens = additional_special_tokens
        self.ext = ext
        self.input_fnames = glob.glob(f"{self.input_dir}/*{self.ext}")
        self.vocab = None
        self.freq = None
        self.token_counter = None

    def count_one(self, fname: str) -> Counter:
        with open(fname, "r") as f:
            word_list = [self.pre_tokenize_func(line) for line in f]
            flat_list = [item for sublist in word_list for item in sublist]
        return Counter(flat_list)

    def count_parallel(self, nb_cores: int = _nb_cores) -> Dict[(str, int)]:
        with multiprocessing.Pool(nb_cores) as pool:
            counters = pool.map(self.count_one, self.input_fnames)
        self.token_counter = sum(counters, Counter())
        counter_all = self.token_counter.most_common(self.vocab_size)
        self.freq = [(i, 0) for i in self.special_tokens] + counter_all
        self.vocab = dict((c[0], i) for i, c in enumerate(self.freq))
        return self.vocab

    def save_vocab(self, output_path: str):
        with open(output_path, "w") as f:
            json.dump(self.vocab, f)
