import logging
logging.basicConfig(level=logging.INFO)

from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification, 
    AutoConfig,
    Trainer, 
    TrainingArguments
)
from transformers.data.processors.utils import InputFeatures

#thai2transformers
from thai2transformers.datasets import SequenceClassificationDataset
from thai2transformers.metrics import sequence_classification_metrics

#argparse
import argparse
# python train_sequence_classification_huggingface.py --model_name_or_path xlm-roberta-base \
# --num_labels 5 --train_dir data/train_th --eval_dir data/valid_th --num_train_epochs 3

def main():
    #argparser
    parser = argparse.ArgumentParser(
        prog="train_sequence_classification_huggingface",
        description="train sequence classification with huggingface Trainer",
    )
    
    #required
    parser.add_argument("--model_name_or_path", type=str,)
    parser.add_argument("--num_labels", type=int,)
    parser.add_argument("--train_dir", type=str,)
    parser.add_argument("--eval_dir", type=str,)
    parser.add_argument("--num_train_epochs", type=int,)

    #checkpoint
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument('--overwrite_output_dir', default=True, type=lambda x: (str(x).lower() in ['true','True','T']))
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    
    #logs
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--logging_steps", type=int, default=200)
    
    #eval
    parser.add_argument('--evaluate_during_training', default=True, type=lambda x: (str(x).lower() in ['true','True','T']))
    parser.add_argument("--eval_steps", type=int, default=500)
    
    #train hyperparameters
    parser.add_argument("--train_max_length", type=int, default=128)
    parser.add_argument("--eval_max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument('--dataloader_drop_last', default=False, type=lambda x: (str(x).lower() in ['true','True','T']))
    
    #others
    parser.add_argument("--seed", type=int, default=1412)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() in ['true','True','T']))
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    
    args = parser.parse_args()

    
    #initialize models and tokenizers
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    
    #datasets
    train_dataset = SequenceClassificationDataset(tokenizer,args.train_dir,args.train_max_length)
    eval_dataset = SequenceClassificationDataset(tokenizer,args.eval_dir,args.eval_max_length)
    
    #training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        #checkpoint
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        #logs
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        #eval
        evaluate_during_training=args.evaluate_during_training,
        eval_steps=args.eval_steps,
        #others
        seed=args.seed,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        dataloader_drop_last=args.dataloader_drop_last,
    )

    #initiate trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=sequence_classification_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
    )
    
    #train
    trainer.train()
    
    #evaluate
    trainer.evaluate()

if __name__ == "__main__":
    main()