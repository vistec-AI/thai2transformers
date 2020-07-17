import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
from thai2transformers.metrics import sequence_classification_metrics
from thai2transformers.datasets import SequenceClassificationDataset
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
    get_constant_schedule, 
    AutoTokenizer, 
    AutoModel,
)

class SequenceClassificationFinetuner(pl.LightningModule):
    def __init__(self, hparams):
        super(SequenceClassificationFinetuner, self).__init__()
        
        self.hparams = hparams
        self.model = AutoModel.from_pretrained(
            self.hparams.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(self.hparams.num_hidden, self.hparams.num_labels),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):
        _, o2 = self.model(input_ids,
                           attention_mask=attention_mask,
                           position_ids=position_ids,
                           head_mask=head_mask,
                           inputs_embeds=inputs_embeds)
        preds = self.head(o2)
        return preds

    def _step(self, batch):
        preds = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        loss = self.loss_fn(preds, batch['label'])

        return loss, preds

    def training_step(self, batch, batch_nb):
        loss, _ = self._step(batch)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, preds = self._step(batch)
        labels = batch['label']
        pred = argparse.Namespace(label_ids=labels.cpu(),predictions=preds.cpu())
        results = sequence_classification_metrics(pred)
        results['loss'] = loss.cpu()
        return results
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = np.stack([x['loss'] for x in outputs]).mean()
        avg_val_acc = np.stack([x['accuracy'] for x in outputs]).mean()
        avg_val_f1_micro = np.stack([x['f1_micro'] for x in outputs]).mean()
        avg_val_f1_macro = np.stack([x['f1_macro'] for x in outputs]).mean()
        avg_val_precision_micro = np.stack([x['precision_micro'] for x in outputs]).mean()
        avg_val_precision_macro = np.stack([x['precision_macro'] for x in outputs]).mean()
        avg_val_recall_micro = np.stack([x['recall_micro'] for x in outputs]).mean()
        avg_val_recall_macro = np.stack([x['recall_macro'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_val_loss, 
                            'avg_val_acc': avg_val_acc,
                            'avg_val_f1_micro': avg_val_f1_micro,
                            'avg_val_precision_micro': avg_val_precision_micro,
                            'avg_val_recall_micro': avg_val_recall_micro,
                            'avg_val_f1_macro': avg_val_f1_macro,
                            'avg_val_precision_macro': avg_val_precision_macro,
                            'avg_val_recall_macro': avg_val_recall_macro,
                           }
        return {'val_loss': avg_val_loss,
                'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        loss, preds = self._step(batch)
        labels = batch['label']
        pred = argparse.Namespace(label_ids=labels.cpu(),predictions=preds.cpu())
        results = sequence_classification_metrics(pred)
        results['loss'] = loss.cpu()
        return results

    def test_epoch_end(self, outputs):
        avg_test_loss = np.stack([x['loss'] for x in outputs]).mean()
        avg_test_acc = np.stack([x['accuracy'] for x in outputs]).mean()
        avg_test_f1_micro = np.stack([x['f1_micro'] for x in outputs]).mean()
        avg_test_f1_macro = np.stack([x['f1_macro'] for x in outputs]).mean()
        avg_test_precision_micro = np.stack([x['precision_micro'] for x in outputs]).mean()
        avg_test_precision_macro = np.stack([x['precision_macro'] for x in outputs]).mean()
        avg_test_recall_micro = np.stack([x['recall_micro'] for x in outputs]).mean()
        avg_test_recall_macro = np.stack([x['recall_macro'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_test_loss, 
                            'avg_test_acc': avg_test_acc,
                            'avg_test_f1_micro': avg_test_f1_micro,
                            'avg_test_precision_micro': avg_test_precision_micro,
                            'avg_test_recall_micro': avg_test_recall_micro,
                            'avg_test_f1_macro': avg_test_f1_macro,
                            'avg_test_precision_macro': avg_test_precision_macro,
                            'avg_test_recall_macro': avg_test_recall_macro,
                           }
        return {'test_loss': avg_test_loss,
                'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
        on_tpu=False, using_native_amp=False, using_lbfgs=False
    ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        train_dataset = SequenceClassificationDataset(tokenizer=self.tokenizer,
            data_dir = self.hparams.train_dir,
            max_length=self.hparams.max_length,
            )
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparams.per_device_train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=4,
        )
        
        #calculate total timesteps
        t_total = (
            (
                len(dataloader.dataset)
                // (self.hparams.per_device_train_batch_size * max(1, self.hparams.n_gpu))
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        #create scheduler
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = SequenceClassificationDataset(tokenizer=self.tokenizer,
            data_dir = self.hparams.valid_dir,
            max_length=self.hparams.max_length,
            )
        
        return DataLoader(
            val_dataset, batch_size=self.hparams.per_device_eval_batch_size, num_workers=4
        )

    def test_dataloader(self):
        test_dataset = SequenceClassificationDataset(tokenizer=self.tokenizer,
            data_dir = self.hparams.test_dir,
            max_len=self.hparams.max_length,
            )
        
        return DataLoader(
            test_dataset, batch_size=self.hparams.per_device_eval_batch_size, num_workers=4
        )
