import argparse
import math
import os
import sys

sys.path.append('..')

from functools import partial
import urllib.request
from tqdm import tqdm
from pathlib import Path

import pandas as pd
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
    get_constant_schedule, 
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    RobertaConfig,
    CamembertTokenizer,
    BertTokenizer,
    XLMRobertaTokenizer
)

from datasets import load_dataset, list_metrics, load_dataset, Dataset
from thai2transformers.datasets import SequenceClassificationDataset
from thai2transformers.metrics import classification_metrics, multilabel_classification_metrics
from thai2transformers.finetuners import SequenceClassificationFinetuner
from thai2transformers.models import RobertaForMultiLabelSequenceClassification
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
)
from thai2transformers.utils import get_dict_val
from thai2transformers.conf import Task

CACHE_DIR = f'{str(Path.home())}/.cache/huggingface_datasets'

METRICS = {
    Task.MULTICLASS_CLS: classification_metrics,
    Task.MULTILABEL_CLS: multilabel_classification_metrics
}

TOKENIZER_CLS = {
    'mbert': BertTokenizer,
    'xlmr': XLMRobertaTokenizer,
    'spm_camembert': CamembertTokenizer,
    'spm': ThaiRobertaTokenizer,
    'newmm': ThaiWordsNewmmTokenizer,
    'syllable': ThaiWordsSyllableTokenizer,
    'sefr_cut': FakeSefrCutTokenizer,
    
}

DATASET_METADATA = {
    'wisesight_sentiment': {
        'task': Task.MULTICLASS_CLS,
        'text_input_col_name': 'texts',
        'label_col_name': 'category',
        'num_labels': 4,
        'split_names': ['train', 'validation', 'test']
    },
    'wongnai_reviews': {
        'task': Task.MULTICLASS_CLS,
        'text_input_col_name': 'review_body',
        'label_col_name': 'star_rating',
        'num_labels': 5,
        'split_names': ['train', 'test']
    },
    'generated_reviews_enth': { # th review rating , correct translation only
        'task': Task.MULTICLASS_CLS,
        'text_input_col_name': ['translation', 'th'],
        'label_col_name': 'review_star',
        'num_labels': 5,
        'split_names': ['train', 'validation', 'test']
    },
    'prachathai67k': {
        'url': 'https://archive.org/download/prachathai67k/data.zip',
        'task': Task.MULTILABEL_CLS,
        'text_input_col_name': 'body_text',
        'label_col_name': ['politics', 'human_rights', 'quality_of_life',
                           'international', 'social', 'environment',
                           'economics', 'culture', 'labor',
                           'national_security', 'ict', 'education'],
        'num_labels': 12,
        'split_names': ['train', 'validation', 'test']
    }
}


def init_model_tokenizer_for_seq_cls(model_dir, tokenizer_cls, tokenizer_dir, task, num_labels):
    
    config = AutoConfig.from_pretrained(
        model_dir,
        num_labels=num_labels
    )
 
    tokenizer = tokenizer_cls.from_pretrained(
        tokenizer_dir,
    )
    if task == Task.MULTICLASS_CLS:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            config=config,
        )
    if task == Task.MULTILABEL_CLS:
        model = RobertaForMultiLabelSequenceClassification.from_pretrained(
            model_dir,
            config=config,
        )

    print(f'\n[INFO] Model architecture: {model} \n\n')
    print(f'\n[INFO] tokenizer: {tokenizer} \n\n')

    return model, tokenizer, config

def init_trainer(task, model, train_dataset, val_dataset, warmup_steps, args): 
        
    training_args = TrainingArguments(
                        num_train_epochs=args.num_train_epochs,
                        per_device_train_batch_size=args.batch_size,
                        per_device_eval_batch_size=args.batch_size,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        learning_rate=args.learning_rate,
                        warmup_steps=warmup_steps,
                        weight_decay=args.weight_decay,
                        adam_epsilon=args.adam_epsilon,
                        max_grad_norm=args.max_grad_norm,
                        #checkpoint
                        output_dir=args.output_dir,
                        overwrite_output_dir=True,
                        #logs
                        logging_dir=args.log_dir,
                        logging_first_step=False,
                        logging_steps=args.logging_steps,
                        #eval
                        evaluation_strategy='steps',
                        eval_steps=args.eval_steps,
                        #others
                        seed=args.seed,
                        fp16=args.fp16,
                        fp16_opt_level=args.fp16_opt_level,
                        dataloader_drop_last=True,
                        no_cuda=args.no_cuda,
                        metric_for_best_model=args.metric_for_best_model,
                        prediction_loss_only=False
                    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=METRICS[task],
        train_dataset=train_dataset,
        eval_dataset=val_dataset    
    )
    return trainer, training_args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=str)
    parser.add_argument('tokenizer_dir', type=str)
    parser.add_argument('tokenizer_type', type=str, help='The type token model used. Specify the name of tokenizer either `spm`, `newmm`, `syllable`, or `sefr_cut`.')
    parser.add_argument('dataset_name', help='Specify the dataset name to finetune. Currently, sequence classification datasets include `wisesight_sentiment`, `generated_reviews_enth` and`wongnai_reviews`.')
    parser.add_argument('--prepare_for_tokenization', action='store_true', default=False, help='To replace space with a special token e.g. `<_>`. This may require for some pretrained models.')
    parser.add_argument('--space_token', type=str, default='<_>', help='The special token for space, specify if argumet: prepare_for_tokenization is applied')
    parser.add_argument('--max_seq_length', type=int, default=512)

    # Finetuning
    parser.add_argument('output_dir', type=str)
    parser.add_argument('log_dir', type=str)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-05)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--greater_is_better', action='store_true', default=False)
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss')
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    args = parser.parse_args()


    try:
        print(f'\n\n[INFO] Dataset: {args.dataset_name}')
        print(f'[INFO] Task: {DATASET_METADATA[args.dataset_name]["task"].value}')
        print(f'\n[INFO] space_token: {args.space_token}')
        print(f'[INFO] prepare_for_tokenization: {args.prepare_for_tokenization}\n')
        # Hotfix: As currently (Wed 6 Jan 2021), the `prachathai67k` dataset can't be download directly with `datasets.load_dataset` method
        if args.dataset_name == 'prachathai67k':
        

            import jsonlines
            import zipfile

            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)
                        
            # 1. Download prachathai67k dataset from Internet Archive direct link

            
            download_dir = f'{CACHE_DIR}/prachathai67k'
            out_zip_path = os.path.join(download_dir, 'data.zip')
            out_dir = f'{CACHE_DIR}/prachathai67k/'                 
            data_dir = f'{CACHE_DIR}/prachathai67k/data'  
            
            if not os.path.exists(out_zip_path):
                os.makedirs(download_dir, exist_ok=True)
                print(f'\n[INFO] Start downloading `prachathai67k` dataset.')
                with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=DATASET_METADATA[args.dataset_name]['url'].split('/')[-1]) as t:
                    urllib.request.urlretrieve(DATASET_METADATA[args.dataset_name]['url'], filename=out_zip_path, reporthook=t.update_to)
                print(f'\n[INFO] Done.')

                print(f'\n[INFO] Start extracting zipped file from {out_zip_path}')  
                with zipfile.ZipFile(out_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(out_dir)
                print(f'\n[INFO] Done.')

            # 2. Read jsonlines object (`train.jsonl`, `val.jsonl`, and `test.jsonl`)
            print(f'\n[INFO] Loading dataset from local files stored in `{data_dir}`.')
            _dataset = dict()
            for split_name in ['train', 'valid', 'test']:
                with jsonlines.open(os.path.join(data_dir, f'{split_name}.jsonl')) as f:
                    _dataset[split_name] = list(iter(f))[:1000]

            # 3. Convert list of objects into DataFrame
            _dataset_df = { split_name: pd.DataFrame(_dataset[split_name]) for split_name in ['train', 'valid', 'test']}

            # 4. Convert DataFrame into datasets.Dataset instance
            dataset = { split_name: Dataset.from_pandas(_dataset_df[split_name]) for split_name in ['train', 'valid', 'test']}
            dataset['validation'] = dataset.pop('valid') # rename key
            print(f'dataset: {dataset}')
            print(f'\nDone.')

        else:
            dataset = load_dataset(args.dataset_name)

        if args.tokenizer_type == 'sefr_cut':
            print(f'Apply `sefr_cut` tokenizer to the text inputs of {args.dataset_name} dataset')
            import sefr_cut
            sefr_cut.load_model('best')
            sefr_tokenize = lambda x: sefr_cut.tokenize(x)
            
            text_input_col_name = DATASET_METADATA[args.dataset_name]['text_input_col_name']

            for split_name in ['train', 'validation', 'test']:

                dataset[split_name] = dataset[split_name].map(lambda batch: { 
                                        text_input_col_name: '<|>'.join([ '<|>'.join(tok_text + ['<_>']) for tok_text in sefr_tokenize(get_dict_val(batch, text_input_col_name).split()) ]) 
                                    }, batched=False, batch_size=1)
            
    except Exception as e:
        raise e

    if args.tokenizer_type not in list(TOKENIZER_CLS.keys()):
        raise f"The tokenizer type `{args.tokenizer_type}`` is not supported"
    
    tokenizer_cls = TOKENIZER_CLS[args.tokenizer_type]
    task = DATASET_METADATA[args.dataset_name]['task']
    
    model, tokenizer, config = init_model_tokenizer_for_seq_cls(args.model_dir,
                                                        tokenizer_cls,
                                                        args.tokenizer_dir,
                                                        task=task,
                                                        num_labels=DATASET_METADATA[args.dataset_name]['num_labels'])
    if args.tokenizer_type == 'spm_camembert':
        tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', args.space_token]

    print('\n[INFO] Preprocess texts in datasets')
    # 
    print('[INFO] Done.')

    print('\n[INFO] Tokenizing texts in datasets')
    dataset_split = { split_name: SequenceClassificationDataset.from_dataset(
                        task,
                        tokenizer,
                        dataset[split_name],
                        DATASET_METADATA[args.dataset_name]['text_input_col_name'],
                        DATASET_METADATA[args.dataset_name]['label_col_name'],
                        max_length=args.max_seq_length - 2,
                        space_token=args.space_token,
                        prepare_for_tokenization=args.prepare_for_tokenization) for split_name in ['train', 'validation', 'test']
                    }
    print('[INFO] Done.')
        
    warmup_steps = math.ceil(len(dataset_split['train']) / args.batch_size * args.warmup_ratio * args.num_train_epochs)

    print(f'\n[INFO] Number of train examples = {len(dataset["train"])}')
    print(f'[INFO] Number of validation examples = {len(dataset["validation"])}')
    print(f'[INFO] Number of batches per epoch (training set) = {math.ceil(len(dataset_split["train"]) / args.batch_size)}')
    print(f'[INFO] Number of batches per epoch (validation set) = {math.ceil(len(dataset_split["validation"]))}')
    print(f'[INFO] Warmup ratio = {args.warmup_ratio}')
    print(f'[INFO] Warmup steps = {warmup_steps}')
    print(f'[INFO] Learning rate: {args.learning_rate}')
    print(f'[INFO] Logging steps: {args.logging_steps}')
    print(f'[INFO] Saving steps: {args.save_steps}')
    print(f'[INFO] FP16 training: {args.fp16}\n')
    


    trainer, training_args = init_trainer(task=task,
                                model=model,
                                train_dataset=dataset_split['train'],
                                val_dataset=dataset_split['validation'],
                                warmup_steps=warmup_steps,
                                args=args)

    print('[INFO] TrainingArguments:')
    print(training_args)
    print('\n')

    print('\nBegin model finetuning.')
    trainer.train()
    print('Done.\n')


    print('\nBegin model evaluation on test set.')
    result = trainer.evaluate(eval_dataset=dataset_split['test'])
    print(f'Evaluation on test set (dataset: {args.dataset_name})')    
    for key, value in result.items():
        print(f'{key} : {value:.4f}')

    print('Done.\n')
