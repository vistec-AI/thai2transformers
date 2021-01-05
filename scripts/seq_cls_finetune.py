import argparse
import math
import sys
sys.path.append('..')

from functools import partial

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
    CamembertTokenizer
)

from datasets import load_dataset, list_metrics, load_dataset
from thai2transformers.datasets import SequenceClassificationDataset
from thai2transformers.metrics import classification_metrics
from thai2transformers.finetuners import SequenceClassificationFinetuner

from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
)



TOKENIZER_CLS = {
    'spm_camembert': CamembertTokenizer,
    'spm': ThaiRobertaTokenizer,
    'newmm': ThaiWordsNewmmTokenizer,
    'syllable': ThaiWordsSyllableTokenizer,
    'sefr_cut': FakeSefrCutTokenizer,
    
}

DATASET_METADATA = {
    'wisesight_sentiment': {
        'task': 'multiclass_classification',
        'text_input_col_name': 'texts',
        'label_col_name': 'category',
        'num_labels': 4,
        'split_names': ['train', 'validation', 'test']
    },
    'wongnai_reviews': {
        'task': 'multiclass_classification',
        'text_input_col_name': 'review_body',
        'label_col_name': 'star_rating',
        'num_labels': 5,
        'split_names': ['train', 'test']
    },
    'generated_reviews_enth': { # th review rating , correct translation only
        'task': 'multiclass_classification',
        'text_input_col_name': 'translation.th',
        'label_col_name': 'review_star',
        'num_labels': 5,
        'split_names': ['train', 'validation', 'test']
    }
    'prachathai67k': {
        'task': 'multilabel_classification',
        'text_input_col_name': 'body_text',
        'label_col_name': '..',
        'num_labels': 12
    }
}

def init_model_tokenizer_for_seq_cls(model_dir, tokenizer_cls, tokenizer_dir, num_labels):
    
    config = AutoConfig.from_pretrained(
        model_dir,
        num_labels=num_labels
    )
 
    tokenizer = tokenizer_cls.from_pretrained(
        tokenizer_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
    )

    print(f'\n[INFO] Model architecute: {model} \n\n')
    print(f'\n[INFO] tokenizer: {tokenizer} \n\n')

    return model, tokenizer, config

def init_trainer(model, train_dataset, val_dataset,
                 output_dir, log_dir,
                 num_train_epochs=1,
                 learning_rate=1e-05,
                 weight_decay=0.1,
                 warmup_steps=0,
                 batch_size=16,
                 eval_steps=250,
                 no_cuda=True,
                 save_steps=500,
                 seed=2020,
                 logging_steps=10,
                 greater_is_better=False,
                 fp16=False,
                 metric_for_best_model='eval_loss'):
        
    training_args = TrainingArguments(
                        num_train_epochs=num_train_epochs,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        gradient_accumulation_steps=1,
                        learning_rate=learning_rate,
                        warmup_steps=warmup_steps,
                        weight_decay=weight_decay,
                        adam_epsilon=1e-08,
                        max_grad_norm=1.0,
                        #checkpoint
                        output_dir=output_dir,
                        overwrite_output_dir=True,
                        #logs
                        logging_dir=log_dir,
                        logging_first_step=True,
                        logging_steps=logging_steps,
                        #eval
                        evaluation_strategy='steps',
                        eval_steps=eval_steps,
                        #others
                        seed=seed,
                        fp16=fp16,
                        fp16_opt_level="O1",
                        dataloader_drop_last=True,
                        no_cuda=no_cuda,
                        metric_for_best_model=metric_for_best_model,
                        prediction_loss_only=False
                    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=classification_metrics,
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


    parser.add_argument('--target_class', type=str, default='multiclass', help='Specify the type for target class either `multilabel` or `multiclass`.')
    
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
    
    args = parser.parse_args()


    try:
        print(f'\n\n[INFO] Dataset: {args.dataset_name}\n')
        dataset = load_dataset(args.dataset_name)

        print(f'\n[INFO] space_token: {args.space_token}')
        print(f'[INFO] prepare_for_tokenization: {args.prepare_for_tokenization}\n')

        if args.tokenizer_type == 'sefr_cut':
            print(f'Apply `sefr_cut` tokenizer to the text inputs of {args.dataset_name} dataset')
            import sefr_cut
            sefr_cut.load_model('best')
            sefr_tokenize = lambda x: sefr_cut.tokenize(x)
            
            text_input_col_name = DATASET_METADATA[args.dataset_name]['text_input_col_name']

            for split_name in ['train', 'validation', 'test']:

                dataset[split_name] = dataset[split_name].map(lambda batch: { 
                                        text_input_col_name: '<|>'.join([ '<|>'.join(tok_text + ['<_>']) for tok_text in sefr_tokenize(batch[text_input_col_name].split()) ]) 
                                    }, batched=False, batch_size=1)
            
    except Exception as e:
        raise e

    if args.tokenizer_type not in list(TOKENIZER_CLS.keys()):
        raise f"The tokenizer type `{args.tokenizer_type}`` is not supported"
    
    tokenizer_cls = TOKENIZER_CLS[args.tokenizer_type]

    
    model, tokenizer, config = init_model_tokenizer_for_seq_cls(args.model_dir,
                                                        tokenizer_cls,
                                                        args.tokenizer_dir,
                                                        num_labels=DATASET_METADATA[args.dataset_name]['num_labels'])
    if args.tokenizer_type == 'spm_camembert':
        tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', args.space_token]

    dataset_split = { split_name: SequenceClassificationDataset.from_dataset(tokenizer, dataset[split_name],
                        DATASET_METADATA[args.dataset_name]['text_input_col_name'],
                        DATASET_METADATA[args.dataset_name]['label_col_name'],
                        max_length=args.max_seq_length - 2,
                        space_token=args.space_token,
                        prepare_for_tokenization=args.prepare_for_tokenization) for split_name in ['train', 'validation', 'test']
                    }

        
    warmup_steps = math.ceil(len(dataset_split['train']) / args.batch_size * args.warmup_ratio * args.num_train_epochs)

    print(f'[INFO] Number of train examples = {len(dataset["train"])}')
    print(f'[INFO] Number of validation examples = {len(dataset["validation"])}')

    print(f'[INFO] Number of batches per epoch (training set) = {math.ceil(len(dataset_split["train"]) / args.batch_size)}')
    print(f'[INFO] Number of batches per epoch (validation set) = {math.ceil(len(dataset_split["validation"]))}')
    print(f'[INFO] Warmup steps = {warmup_steps}')
    print(f'[INFO] Learning rate: {args.learning_rate}')
    print(f'[INFO] Logging steps: {args.logging_steps}')
    print(f'[INFO] Saving steps: {args.save_steps}')
    print(f'[INFO] FP16 training: {args.fp16}\n')
    


    trainer, training_args = init_trainer(model=model,
                                train_dataset=dataset_split['train'],
                                val_dataset=dataset_split['validation'],
                                output_dir=args.output_dir,
                                log_dir=args.log_dir,
                                num_train_epochs=args.num_train_epochs,
                                learning_rate=args.learning_rate,
                                weight_decay=args.weight_decay,
                                warmup_steps=warmup_steps,
                                batch_size=args.batch_size,
                                no_cuda=args.no_cuda,
                                fp16=args.fp16)

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
