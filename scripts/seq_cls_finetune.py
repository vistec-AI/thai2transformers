import argparse
import math
import os
import sys

sys.path.append('..')

from functools import partial
import urllib.request
from tqdm import tqdm
from pathlib import Path
from sklearn import preprocessing
import pandas as pd
import numpy as np
import wandb
import torch
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
    CamembertTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    BertConfig,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    XLMRobertaConfig,
    DataCollatorWithPadding,
    default_data_collator
)

from datasets import load_dataset, list_metrics, load_dataset, Dataset
from thai2transformers.datasets import SequenceClassificationDataset
from thai2transformers.metrics import classification_metrics, multilabel_classification_metrics
from thai2transformers.finetuners import SequenceClassificationFinetuner
from thai2transformers.auto import AutoModelForMultiLabelSequenceClassification
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
)
from thai2transformers.utils import get_dict_val
from thai2transformers.conf import Task
from thai2transformers.preprocess import process_transformers

CACHE_DIR = f'{str(Path.home())}/.cache/huggingface_datasets'

METRICS = {
    Task.MULTICLASS_CLS: classification_metrics,
    Task.MULTILABEL_CLS: multilabel_classification_metrics
}

PUBLIC_MODEL = {
    'mbert': {
        'name': 'bert-base-multilingual-cased',
        'tokenizer': BertTokenizerFast.from_pretrained('bert-base-multilingual-cased'),
        'config': BertConfig.from_pretrained('bert-base-multilingual-cased'),
    },
    'xlmr': {
        'name': 'xlm-roberta-base',
        'tokenizer': XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base'),
        'config': XLMRobertaConfig.from_pretrained('xlm-roberta-base'),
    },
    'xlmr-large': {
        'name': 'xlm-roberta-large',
        'tokenizer': XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large'),
        'config': XLMRobertaConfig.from_pretrained('xlm-roberta-base'),
    },
}

TOKENIZER_CLS = {
    'spm_camembert': CamembertTokenizer,
    'spm': ThaiRobertaTokenizer,
    'newmm': ThaiWordsNewmmTokenizer,
    'syllable': ThaiWordsSyllableTokenizer,
    'sefr_cut': FakeSefrCutTokenizer,
}

DATASET_METADATA = {
    'wisesight_sentiment': {
        'huggingface_dataset_name': 'wisesight_sentiment',
        'task': Task.MULTICLASS_CLS,
        'text_input_col_name': 'texts',
        'label_col_name': 'category',
        'num_labels': 4,
        'split_names': ['train', 'validation', 'test']
    },
    'wongnai_reviews': {
        'huggingface_dataset_name': 'wongnai_reviews',
        'task': Task.MULTICLASS_CLS,
        'text_input_col_name': 'review_body',
        'label_col_name': 'star_rating',
        'num_labels': 5,
        'split_names': ['train', 'validation', 'test']
    },
    'generated_reviews_enth-correct_translation': { 
        'huggingface_dataset_name': 'generated_reviews_enth',
        'task': Task.MULTICLASS_CLS,
        'text_input_col_name': ['translation', 'th'],
        'label_col_name': 'correct',
        'num_labels': 2,
        'split_names': ['train', 'validation', 'test']
    },
    'generated_reviews_enth-review_star': { 
        'huggingface_dataset_name': 'generated_reviews_enth',
        'task': Task.MULTICLASS_CLS,
        'text_input_col_name': ['translation', 'th'],
        'label_col_name': 'review_star',
        'num_labels': 5,
        'split_names': ['train', 'validation', 'test']
    },
    'prachathai67k': {
        'huggingface_dataset_name': 'prachathai67k',
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

def init_public_model_tokenizer_for_seq_cls(public_model_name, task, num_labels):
    
    config = PUBLIC_MODEL[public_model_name]['config']
    config.num_labels = num_labels
    tokenizer = PUBLIC_MODEL[public_model_name]['tokenizer']
    model_name = PUBLIC_MODEL[public_model_name]['name']
    if task == Task.MULTICLASS_CLS:
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   config=config)
    if task == Task.MULTILABEL_CLS:
        model = AutoModelForMultiLabelSequenceClassification.from_pretrained(model_name,
                                                                             config=config)

    print(f'\n[INFO] Model architecture: {model} \n\n')
    print(f'\n[INFO] tokenizer: {tokenizer} \n\n')

    return model, tokenizer, config

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
        model = AutoModelForMultiLabelSequenceClassification.from_pretrained(
            model_dir,
            config=config,
        )

    print(f'\n[INFO] Model architecture: {model} \n\n')
    print(f'\n[INFO] tokenizer: {tokenizer} \n\n')

    return model, tokenizer, config

def init_trainer(task, model, train_dataset, val_dataset, warmup_steps, args, data_collator=default_data_collator): 
        
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
                        evaluation_strategy='epoch' if 'validation' in DATASET_METADATA[args.dataset_name]['split_names'] else 'no',
                        load_best_model_at_end=True,
                        #others
                        seed=args.seed,
                        fp16=args.fp16,
                        fp16_opt_level=args.fp16_opt_level,
                        dataloader_drop_last=True,
                        no_cuda=args.no_cuda,
                        metric_for_best_model=args.metric_for_best_model,
                        prediction_loss_only=False,
                        run_name=args.wandb_run_name
                    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=METRICS[task],
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    return trainer, training_args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('tokenizer_type_or_public_model_name', type=str, help='The type token model used. Specify the name of tokenizer either `spm`, `newmm`, `syllable`, or `sefr_cut`.')
    parser.add_argument('dataset_name', help='Specify the dataset name to finetune. Currently, sequence classification datasets include `wisesight_sentiment`, `generated_reviews_enth-correct_translation`, `generated_reviews_enth-review_star` and`wongnai_reviews`.')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('log_dir', type=str)

    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--tokenizer_dir', type=str)
    parser.add_argument('--prepare_for_tokenization', action='store_true', default=False, help='To replace space with a special token e.g. `<_>`. This may require for some pretrained models.')
    parser.add_argument('--space_token', type=str, default='<_>', help='The special token for space, specify if argumet: prepare_for_tokenization is applied')
    parser.add_argument('--max_seq_length', type=int, default=None)

    # Finetuning
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--greater_is_better', action='store_true', default=True)
    parser.add_argument('--metric_for_best_model', type=str, default='f1_micro')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # wandb
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    # Set seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        print(f'\n\n[INFO] Dataset: {args.dataset_name}')
        print(f'\n\n[INFO] Huggingface\'s dataset name: {DATASET_METADATA[args.dataset_name]["huggingface_dataset_name"]} ')
        print(f'[INFO] Task: {DATASET_METADATA[args.dataset_name]["task"].value}')
        print(f'\n[INFO] space_token: {args.space_token}')
        print(f'[INFO] prepare_for_tokenization: {args.prepare_for_tokenization}\n')

        if args.dataset_name == 'wongnai_reviews':
            print(f'\n\n[INFO] For Wongnai reviews dataset, perform train-val set splitting (0.9,0.1)')
            dataset = load_dataset(DATASET_METADATA[args.dataset_name]["huggingface_dataset_name"])
            print(f'\n\n[INFO] Perform dataset splitting')
            train_val_split = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=args.seed)
            dataset['train'] = train_val_split['train']
            dataset['validation'] = train_val_split['test']
            print(f'\n\n[INFO] Done')
            print(f'dataset: {dataset}')
        elif args.dataset_name == 'prachathai67k':
            # Hotfix: As currently (Wed 6 Jan 2021), the `prachathai67k` dataset can't be download directly with `datasets.load_dataset` method

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
            for split_name in DATASET_METADATA[args.dataset_name]['split_names']:
                with jsonlines.open(os.path.join(data_dir, f'{split_name}.jsonl')) as f:
                    _dataset[split_name] = list(iter(f))

            # 3. Convert list of objects into DataFrame
            _dataset_df = { split_name: pd.DataFrame(_dataset[split_name]) for split_name in DATASET_METADATA[args.dataset_name]['split_names']}

            # 4. Convert DataFrame into datasets.Dataset instance
            dataset = { split_name: Dataset.from_pandas(_dataset_df[split_name]) for split_name in DATASET_METADATA[args.dataset_name]['split_names']}
            dataset['validation'] = dataset.pop('valid') # rename key
            print(f'dataset: {dataset}')
            print(f'\nDone.')
        else:
            dataset = load_dataset(DATASET_METADATA[args.dataset_name]["huggingface_dataset_name"])

        if DATASET_METADATA[args.dataset_name]['task'] == Task.MULTICLASS_CLS:

            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(get_dict_val(dataset['train'], keys=DATASET_METADATA[args.dataset_name]['label_col_name']))

        if args.tokenizer_type_or_public_model_name == 'sefr_cut':
            print(f'Apply `sefr_cut` tokenizer to the text inputs of {args.dataset_name} dataset')
            import sefr_cut
            sefr_cut.load_model('best')
            sefr_tokenize = lambda x: sefr_cut.tokenize(x)
            
            text_input_col_name = DATASET_METADATA[args.dataset_name]['text_input_col_name']

            for split_name in DATASET_METADATA[args.dataset_name]['split_names']:

                dataset[split_name] = dataset[split_name].map(lambda batch: { 
                                        text_input_col_name: '<|>'.join([ '<|>'.join(tok_text + ['<_>']) for tok_text in sefr_tokenize(get_dict_val(batch, text_input_col_name).split()) ]) 
                                    }, batched=False, batch_size=1)
    except Exception as e:
        raise e

    if args.tokenizer_type_or_public_model_name not in list(TOKENIZER_CLS.keys()) \
       and args.tokenizer_type_or_public_model_name not in list(PUBLIC_MODEL.keys()):
        raise f"The tokenizer type or public model name `{args.tokenizer_type_or_public_model_name}`` is not supported"

    if args.tokenizer_type_or_public_model_name in list(TOKENIZER_CLS.keys()):
        tokenizer_cls = TOKENIZER_CLS[args.tokenizer_type_or_public_model_name]

    task = DATASET_METADATA[args.dataset_name]['task']
    
    if args.tokenizer_type_or_public_model_name in PUBLIC_MODEL.keys():
        model, tokenizer, config = init_public_model_tokenizer_for_seq_cls(args.tokenizer_type_or_public_model_name,
                                                            task=task,
                                                            num_labels=DATASET_METADATA[args.dataset_name]['num_labels'])
    else:
        model, tokenizer, config = init_model_tokenizer_for_seq_cls(args.model_dir,
                                                            tokenizer_cls,
                                                            args.tokenizer_dir,
                                                            task=task,
                                                            num_labels=DATASET_METADATA[args.dataset_name]['num_labels'])
    if args.tokenizer_type_or_public_model_name == 'spm_camembert':
        tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', args.space_token]

    print('\n[INFO] Preprocess and tokenizing texts in datasets')
    max_length = args.max_seq_length if args.max_seq_length else config.max_position_embeddings
    print(f'[INFO] max_length = {max_length} \n')
    
    dataset_split = { split_name: SequenceClassificationDataset.from_dataset(
                        task,
                        tokenizer,
                        dataset[split_name],
                        DATASET_METADATA[args.dataset_name]['text_input_col_name'],
                        DATASET_METADATA[args.dataset_name]['label_col_name'],
                        max_length=max_length,
                        space_token=args.space_token,
                        prepare_for_tokenization=args.prepare_for_tokenization,
                        preprocessor=partial(process_transformers, space_token=args.space_token),
                        label_encoder=label_encoder) for split_name in DATASET_METADATA[args.dataset_name]['split_names']
                    }
    print('[INFO] Done.')
        
    warmup_steps = math.ceil(len(dataset_split['train']) / args.batch_size * args.warmup_ratio * args.num_train_epochs)

    print(f'\n[INFO] Number of train examples = {len(dataset["train"])}')
    print(f'[INFO] Number of batches per epoch (training set) = {math.ceil(len(dataset_split["train"]) / args.batch_size)}')

    if 'validation' in DATASET_METADATA[args.dataset_name]['split_names']:
        print(f'[INFO] Number of validation examples = {len(dataset["validation"])}')
        print(f'[INFO] Number of batches per epoch (validation set) = {math.ceil(len(dataset_split["validation"]))}')
    print(f'[INFO] Warmup ratio = {args.warmup_ratio}')
    print(f'[INFO] Warmup steps = {warmup_steps}')
    print(f'[INFO] Learning rate: {args.learning_rate}')
    print(f'[INFO] Logging steps: {args.logging_steps}')
    print(f'[INFO] FP16 training: {args.fp16}\n')
    
    data_collator = DataCollatorWithPadding(tokenizer,
                                            padding=True,
                                            pad_to_multiple_of=8 if args.fp16 else None)

    trainer, training_args = init_trainer(task=task,
                                model=model,
                                train_dataset=dataset_split['train'],
                                val_dataset=dataset_split['validation'] if 'validation' in DATASET_METADATA[args.dataset_name]['split_names'] else None,
                                warmup_steps=warmup_steps,
                                args=args,
                                data_collator=data_collator)

    print('[INFO] TrainingArguments:')
    print(training_args)
    print('\n')

    print('\nBegin model finetuning.')
    trainer.train()
    print('Done.\n')


    print('\nBegin model evaluation on test set.')
    result = trainer.evaluate(
                eval_dataset=dataset_split['test'])
    print(f'Evaluation on test set (dataset: {args.dataset_name})')    
    
    for key, value in result.items():
        print(f'{key} : {value:.4f}')
        wandb.run.summary[f'test-set_{key}'] = value

    print('[INFO] Done.\n')
    print('[INDO] Begin saving best checkpoint.')
    trainer.save_model(os.path.join(args.output_dir, 'checkpoint-best'))
    print('[INFO] Done.\n')