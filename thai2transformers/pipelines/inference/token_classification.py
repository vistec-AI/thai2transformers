import torch

from typing import Callable, List, Tuple, Union
from functools import partial
import itertools

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from pythainlp.tokenize import word_tokenize as pythainlp_word_tokenize
newmm_word_tokenizer = partial(pythainlp_word_tokenize, keep_whitespace=True, engine='newmm')

from thai2transformers.preprocess import rm_useless_spaces

SPIECE = '▁'

class TokenClassificationPipeline:

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 pretokenizer: Callable[[str], List[str]] = newmm_word_tokenizer,
                 lowercase=False,
                 space_token='<_>',
                 device: int = -1,
                 group_entities: bool = False,
                 strict: bool = False):

        super().__init__()

        assert isinstance(tokenizer, PreTrainedTokenizer)
        assert isinstance(model, PreTrainedModel)
        
        self.model = model
        self.tokenizer = tokenizer
        self.pretokenizer = pretokenizer
        self.lowercase = lowercase
        self.space_token = space_token
        self.device = 'cpu' if device == -1 or not torch.cuda.is_available() else f'cuda:{device}'
        self.group_entities = group_entities
        self.strict = strict

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.model.to(self.device)

    def preprocess(self, inputs: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:

        if self.lowercase:
            inputs = inputs.lower() if type(inputs) == str else list(map(str.lower, inputs))
        
        inputs = rm_useless_spaces(inputs) if type(inputs) == str else list(map(rm_useless_spaces, inputs))

        tokens = self.pretokenizer(inputs) if type(inputs) == str else list(map(self.pretokenizer, inputs))

        tokens = list(map(lambda x: x.replace(' ', self.space_token), tokens)) if type(inputs) == str else \
                 list(map(lambda _tokens: list(map(lambda x: x.replace(' ', self.space_token), _tokens)), tokens))

        return tokens

    def _inference(self, input: str):

        tokens = [[self.tokenizer.bos_token]] + \
                    [self.tokenizer.tokenize(tok) if tok != SPIECE else [SPIECE] for tok in self.preprocess(input)] + \
                    [[self.tokenizer.eos_token]]
        ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        flatten_tokens = list(itertools.chain(*tokens))
        flatten_ids = list(itertools.chain(*ids))

        input_ids = torch.LongTensor([flatten_ids]).to(self.device)


        out = self.model(input_ids=input_ids, return_dict=True)
        probs = torch.softmax(out['logits'], dim=-1)
        vals, indices = probs.topk(1)
        indices_np = indices.detach().cpu().numpy().reshape(-1)

        list_of_token_label_tuple = list(zip(flatten_tokens, [ self.id2label[idx] for idx in indices_np] ))
        merged_preds = self._merged_pred(preds=list_of_token_label_tuple, ids=ids)
        merged_preds_removed_spiece = list(map(lambda x: (x[0].replace(SPIECE, ''), x[1]), merged_preds))

        # remove start and end tokens
        merged_preds_removed_bos_eos = merged_preds_removed_spiece[1:-1]
        # convert to list of Dict objects
        merged_preds_return_dict = [ {'word': word if word != self.space_token else ' ', 'entity': tag } for word, tag in merged_preds_removed_bos_eos ]
        if not self.group_entities:
            return merged_preds_return_dict
        else:
            return self._group_entities(merged_preds_removed_bos_eos)

    def __call__(self, inputs: Union[str, List[str]]):

        """     
            
        """
        if type(inputs) == str:
            return self._inference(inputs)
        
        if type(inputs) == list:
            results = [ self._inference(text) for text in inputs]
            return results
       

    def _merged_pred(self, preds: List[Tuple[str, str]], ids: List[List[int]]):
    
        token_mapping = [ ]
        for i in range(0, len(ids)):
            for j in range(0, len(ids[i])):
                token_mapping.append(i)

        grouped_subtokens = []
        _subtoken = []
        prev_idx = 0
    
        for i, (subtoken, label) in enumerate(preds):
            
            current_idx =  token_mapping[i]
            if prev_idx != current_idx:
                grouped_subtokens.append(_subtoken)
                _subtoken = [(subtoken, label)]
                if i == len(preds) -1:
                    _subtoken = [(subtoken, label)]
                    grouped_subtokens.append(_subtoken)
            elif i == len(preds) -1:
                _subtoken += [(subtoken, label)]
                grouped_subtokens.append(_subtoken)
            else:
                _subtoken += [(subtoken, label)]
            prev_idx = current_idx
        
        merged_subtokens = []
        _merged_subtoken = ''
        for subtoken_group in grouped_subtokens:
            
            first_token_pred = subtoken_group[0][1]
            _merged_subtoken = ''.join(list(map(lambda x: x[0], subtoken_group)))
            merged_subtokens.append((_merged_subtoken, first_token_pred))
        return merged_subtokens

    def _group_entities(self, ner_tags: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        
        B_PREFIX = tuple(('B-', 'B_'))
        I_PREFIX = tuple(('I-', 'I_'))
        tokens, tags = zip(*ner_tags)
        tokens, tags = list(tokens), list(tags)
        ne_position_mappings = []
        _ne_position_mapping = []
        for i, tag in enumerate(tags):
            current_ne = tag.split('-')[-1] if tag != 'O' else 'O'

            if tag.startswith('B-') and i-1 >= 0 and tags[i-1].startswith('B-') :
                ne_position_mappings.append(_ne_position_mapping)
                _ne_position_mapping = []
                _ne_position_mapping.append((i, current_ne))
            elif tag.startswith('I-') and i == 0 and not self.strict:
                _ne_position_mapping.append((i, current_ne))
            elif tag.startswith('I-') and i == 0 and  self.strict:
                _ne_position_mapping.append((i, 'O'))
                tags[i] = 'O'
            elif tag.startswith('B-') and i-1 >= 0 and tags[i-1] == 'O' :
                ne_position_mappings.append(_ne_position_mapping)
                _ne_position_mapping = []
                _ne_position_mapping.append((i, current_ne))
            elif tag.startswith('I-') and i-1 >= 0 and tags[i-1] == 'O' and not self.strict:
                ne_position_mappings.append(_ne_position_mapping)
                _ne_position_mapping = []
                _ne_position_mapping.append((i, current_ne))
            elif tag.startswith('I-') and i-1 >= 0 and tags[i-1] == 'O' and self.strict:
                _ne_position_mapping.append((i, 'O'))
                tags[i] = 'O'
            elif tag.startswith('B-') and i-1 >= 0 and tags[i-1].startswith('I-'):
                ne_position_mappings.append(_ne_position_mapping)
                _ne_position_mapping = []
                _ne_position_mapping.append((i, current_ne))
            elif tags[i].startswith('B-') :
                _ne_position_mapping.append((i, current_ne))
            elif tags[i].startswith('I-') and i -1 >= 0 \
              and ( tags[i-1] == 'O' or tags[i-1].startswith('I-') or tags[i-1].startswith('B-')) \
              and len(_ne_position_mapping) > 0 \
              and _ne_position_mapping[-1][1] == current_ne \
              and not self.strict:
                _ne_position_mapping.append((i, current_ne))
            elif tags[i].startswith('I-') and i -1 >= 0 and len(_ne_position_mapping) > 0 \
              and _ne_position_mapping[-1][1] == current_ne:
                _ne_position_mapping.append((i, current_ne))
      
            elif tags[i].startswith('I-') and i -1 >= 0 and len(_ne_position_mapping) > 0 \
              and _ne_position_mapping[-1][1] != current_ne and not self.strict:
                ne_position_mappings.append(_ne_position_mapping)
                _ne_position_mapping = []
                _ne_position_mapping.append((i, current_ne))
            elif tag == 'O'  and i -1 >= 0 and ( tags[i-1].startswith('I-') or tags[i-1].startswith('B-')):
                # end of I-
                ne_position_mappings.append(_ne_position_mapping)
                _ne_position_mapping = []
                _ne_position_mapping.append((i, current_ne))
            elif tag == 'O'  and i -1 >= 0 and tags[i-1] == 'O':
                _ne_position_mapping.append((i, current_ne))
            elif tag == 'O' and i == 0:
                _ne_position_mapping.append((i, current_ne))

            if i==len(tags) -1:
                if tag == 'O':
                    ne_position_mappings.append(_ne_position_mapping)
                    _ne_position_mapping = []
                elif tag.startswith('B-'):
                    ne_position_mappings.append(_ne_position_mapping)
                    _ne_position_mapping = []
                elif tag.startswith('I-') and tags[i-1].startswith('I-') \
                  and len(_ne_position_mapping) >= 3 \
                  and _ne_position_mapping[-1][1] == current_ne:
                    ne_position_mappings.append(_ne_position_mapping)
                    _ne_position_mapping = []
                elif tag.startswith('I-') and tags[i-1].startswith('B-') \
                  and len(_ne_position_mapping) == 2 \
                  and _ne_position_mapping[-1][1] == current_ne:
                    ne_position_mappings.append(_ne_position_mapping)
                    _ne_position_mapping = []
                elif tag.startswith('I-') and ( tags[i-1].startswith('B-') or tags[i-1].startswith('I-') ) \
                  and len(_ne_position_mapping) >= 1 \
                  and _ne_position_mapping[-1][1] == current_ne \
                  and not self.strict:
                    ne_position_mappings.append(_ne_position_mapping)
                    _ne_position_mapping = []
                elif tag.startswith('I-') and ( tags[i-1] == 'O') \
                  and len(_ne_position_mapping) == 1 \
                  and not self.strict:
                    ne_position_mappings.append(_ne_position_mapping)
                    _ne_position_mapping = []
        groups = []
        for i, ne_position_mapping in enumerate(ne_position_mappings):
        
            text = ''
            ne = ne_position_mapping[0][1]
            for ne_position in ne_position_mapping:
                _token = tokens[ne_position[0]]
                text += _token if _token != self.space_token else ' '
            groups.append({
                'entity_group': ne,
                'word': text
            })
    
        return groups

    