import logging
logging.basicConfig(level=logging.INFO)

#lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

#transformers
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
from thai2transformers.finetuners import SequenceClassificationFinetuner

#argparse
import argparse
# python train_sequence_classification_lightning.py --model_name_or_path xlm-roberta-base \
# --num_labels 7 --num_hidden 768 --train_dir data/train_mari --valid_dir data/test_mari \
# --test_dir data/test_mari --num_train_epochs 10

def main():
    #argparser
    parser = argparse.ArgumentParser(
        prog="train_sequence_classification_lightning",
        description="train sequence classification with pytorch-lightning",
    )
    
    #required
    parser.add_argument("--model_name_or_path", type=str,)
    parser.add_argument("--num_labels", type=int,)
    parser.add_argument("--num_hidden", type=int,)
    parser.add_argument("--train_dir", type=str,)
    parser.add_argument("--valid_dir", type=str,)
    parser.add_argument("--test_dir", type=str,)
    parser.add_argument("--num_train_epochs", type=int,)

    #checkpoint
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--save_total_limit", type=int, default=1)

    #train hyperparameters
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    #callbacks
    parser.add_argument('--early_stopping', default=True, type=lambda x: (str(x).lower() in ['true','True','T']))
    parser.add_argument("--patience", type=int, default=3)

    #others
    parser.add_argument("--seed", type=int, default=1412)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument('--fp16', default=False, type=lambda x: (str(x).lower() in ['true','True','T']))

    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    
    args = parser.parse_args()

    #callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=args.output_dir,
        save_top_k=args.save_total_limit,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       min_delta=0.00,
       patience=args.patience,
       verbose=False,
       mode='min'
    )

    #training args
    train_args = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp16 else 32,
        amp_level=args.fp16_opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback = checkpoint_callback,
        early_stop_callback=early_stop_callback if args.early_stopping else None
    )

    #initiate trainer
    model = SequenceClassificationFinetuner(args)
    trainer = pl.Trainer(**train_args)
    
    #train
    trainer.fit(model)
    
    #evaluate
    trainer.test()

if __name__ == "__main__":
    main()