import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
from thai2transformers.metrics import classification_metrics
from thai2transformers.datasets import (
    SequenceClassificationDataset,
    TokenClassificationDataset,
)
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModel,
)

class TokenClassificationFinetuner(pl.LightningModule):
    def __init__(self, hparams):
        super(TokenClassificationFinetuner, self).__init__()

        self.hparams = hparams
        self.model = AutoModel.from_pretrained(self.hparams.model_name_or_path,)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path,)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden),
            torch.nn.Dropout(p=self.hparams.drop_p),
            torch.nn.Linear(self.hparams.num_hidden, self.hparams.num_labels),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        o1, o2 = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        preds = self.head(o1)
        return preds

    def _flatten_batch(self, batch):
        preds = self.forward(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        )

        active_mask = batch["attention_mask"].view(-1) == 1
        active_preds = preds.view(-1, self.hparams.num_labels)
        active_labels = torch.where(
            active_mask,
            batch["label"].view(-1),
            torch.tensor(self.loss_fn.ignore_index).type_as(batch["label"]),
        )
        return active_preds, active_labels, preds

    def _step(self, batch):
        active_preds, active_labels, preds = self._flatten_batch(batch)
        loss = self.loss_fn(active_preds, active_labels)
        return loss, preds

    def training_step(self, batch, batch_nb):
        loss, _ = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, preds = self._step(batch)
        pred_labs = preds.argmax(2).cpu().numpy()
        
        #get batch labels and pred_labs
        dfs = []
        for i in range(batch['input_ids'].shape[0]):
            df = pd.DataFrame({'word_ids':batch['word_ids'][i].cpu().numpy(),
                               'label':batch['label'][i].cpu().numpy(),
                               'pred_labs':pred_labs[i]})
            df = df[df.label!=0].groupby('word_ids').max().reset_index(drop=True)
            dfs.append(df)
        df_batch = pd.concat(dfs)
        
        #get metrics
        pred = argparse.Namespace(
            label_ids=df_batch.label, predictions=df_batch.pred_labs,
        )
        results = classification_metrics(pred, pred_labs=True)
        results["loss"] = loss.cpu()
        return results

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)
    
    def _avg_epoch_end(self, outputs):
        total_samples = np.sum([x["nb_samples"] for x in outputs])
        avg_loss = np.sum([x["loss"] * x["nb_samples"] for x in outputs]) / total_samples
        avg_acc = np.sum([x["accuracy"] * x["nb_samples"] for x in outputs]) / total_samples
        #micro
        avg_f1_micro = np.sum([x["f1_micro"] * x["nb_samples"] for x in outputs]) / total_samples
        avg_precision_micro = np.sum(
            [x["precision_micro"] * x["nb_samples"] for x in outputs]
        ) / total_samples
        avg_recall_micro = np.sum([x["recall_micro"] * x["nb_samples"] for x in outputs]) / total_samples
        #macro
        avg_f1_macro = np.sum([x["f1_macro"] * x["nb_samples"] for x in outputs]) / total_samples
        avg_precision_macro = np.sum(
            [x["precision_macro"] for x in outputs]
        ) / total_samples
        avg_recall_macro = np.sum([x["recall_macro"] * x["nb_samples"] for x in outputs]) / total_samples
        
        return {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "avg_f1_micro": avg_f1_micro,
            "avg_precision_micro": avg_precision_micro,
            "avg_recall_micro": avg_recall_micro,
            "avg_f1_macro": avg_f1_macro,
            "avg_precision_macro": avg_precision_macro,
            "avg_recall_macro": avg_recall_macro,
            "total_samples": total_samples,
        }
    
    def validation_epoch_end(self, outputs):
        results = self._avg_epoch_end(outputs)
        tensorboard_logs = {
            "val_loss": results["avg_loss"],
            "avg_val_acc": results["avg_acc"],
            "avg_val_f1_micro": results["avg_f1_micro"],
            "avg_val_precision_micro": results["avg_precision_micro"],
            "avg_val_recall_micro": results["avg_recall_micro"],
            "avg_val_f1_macro": results["avg_f1_macro"],
            "avg_val_precision_macro": results["avg_precision_macro"],
            "avg_val_recall_macro": results["avg_recall_macro"],
            "total_samples": results["total_samples"]
        }
        return {
            "val_loss": results["avg_loss"],
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def test_epoch_end(self, outputs):
        results = self._avg_epoch_end(outputs)
        tensorboard_logs = {
            "test_loss": results["avg_loss"],
            "avg_test_acc": results["avg_acc"],
            "avg_test_f1_micro": results["avg_f1_micro"],
            "avg_test_precision_micro": results["avg_precision_micro"],
            "avg_test_recall_micro": results["avg_recall_micro"],
            "avg_test_f1_macro": results["avg_f1_macro"],
            "avg_test_precision_macro": results["avg_precision_macro"],
            "avg_test_recall_macro": results["avg_recall_macro"],
            "total_samples": results["total_samples"]
        }
        return {
            "test_loss": results["avg_loss"],
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.weight"]
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
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        train_dataset = TokenClassificationDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.train_dir,
            max_length=self.hparams.max_length,
            label_pad_token=self.hparams.label_pad_token,
            label_first_subword=self.hparams.label_first_subword,
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparams.per_device_train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=4,
        )

        # calculate total timesteps
        t_total = (
            (
                len(dataloader.dataset)
                // (
                    self.hparams.per_device_train_batch_size
                    * max(1, self.hparams.n_gpu)
                )
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        # create scheduler
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = TokenClassificationDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.valid_dir,
            max_length=self.hparams.max_length,
            label_pad_token=self.hparams.label_pad_token,
            label_first_subword=self.hparams.label_first_subword,
        )

        return DataLoader(
            val_dataset,
            batch_size=self.hparams.per_device_eval_batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        test_dataset = TokenClassificationDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.test_dir,
            max_length=self.hparams.max_length,
            label_pad_token=self.hparams.label_pad_token,
            label_first_subword=self.hparams.label_first_subword,
        )

        return DataLoader(
            test_dataset,
            batch_size=self.hparams.per_device_eval_batch_size,
            num_workers=4,
        )


class SequenceClassificationFinetuner(pl.LightningModule):
    def __init__(self, hparams):
        super(SequenceClassificationFinetuner, self).__init__()

        self.hparams = hparams
        self.model = AutoModel.from_pretrained(self.hparams.model_name_or_path,)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path,)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.num_hidden, self.hparams.num_hidden),
            torch.nn.Dropout(p=self.hparams.drop_p),
            torch.nn.Linear(self.hparams.num_hidden, self.hparams.num_labels),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        o1, o2 = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        preds = self.head(o2)
        return preds

    def _step(self, batch):
        preds = self.forward(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        )

        loss = self.loss_fn(preds, batch["label"])

        return loss, preds

    def training_step(self, batch, batch_nb):
        loss, _ = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, preds = self._step(batch)
        labels = batch["label"]
        pred = argparse.Namespace(label_ids=labels.cpu(), predictions=preds.cpu())
        results = classification_metrics(pred)
        results["loss"] = loss.cpu()
        return results

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def _avg_epoch_end(self, outputs):
        total_samples = np.sum([x["nb_samples"] for x in outputs])
        avg_loss = np.sum([x["loss"] * x["nb_samples"] for x in outputs]) / total_samples
        avg_acc = np.sum([x["accuracy"] * x["nb_samples"] for x in outputs]) / total_samples
        #micro
        avg_f1_micro = np.sum([x["f1_micro"] * x["nb_samples"] for x in outputs]) / total_samples
        avg_precision_micro = np.sum(
            [x["precision_micro"] * x["nb_samples"] for x in outputs]
        ) / total_samples
        avg_recall_micro = np.sum([x["recall_micro"] * x["nb_samples"] for x in outputs]) / total_samples
        #macro
        avg_f1_macro = np.sum([x["f1_macro"] * x["nb_samples"] for x in outputs]) / total_samples
        avg_precision_macro = np.sum(
            [x["precision_macro"] for x in outputs]
        ) / total_samples
        avg_recall_macro = np.sum([x["recall_macro"] * x["nb_samples"] for x in outputs]) / total_samples
        
        return {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "avg_f1_micro": avg_f1_micro,
            "avg_precision_micro": avg_precision_micro,
            "avg_recall_micro": avg_recall_micro,
            "avg_f1_macro": avg_f1_macro,
            "avg_precision_macro": avg_precision_macro,
            "avg_recall_macro": avg_recall_macro,
            "total_samples": total_samples,
        }
    
    def validation_epoch_end(self, outputs):
        results = self._avg_epoch_end(outputs)
        tensorboard_logs = {
            "val_loss": results["avg_loss"],
            "avg_val_acc": results["avg_acc"],
            "avg_val_f1_micro": results["avg_f1_micro"],
            "avg_val_precision_micro": results["avg_precision_micro"],
            "avg_val_recall_micro": results["avg_recall_micro"],
            "avg_val_f1_macro": results["avg_f1_macro"],
            "avg_val_precision_macro": results["avg_precision_macro"],
            "avg_val_recall_macro": results["avg_recall_macro"],
            "total_samples": results["total_samples"]
        }
        return {
            "val_loss": results["avg_loss"],
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def test_epoch_end(self, outputs):
        results = self._avg_epoch_end(outputs)
        tensorboard_logs = {
            "test_loss": results["avg_loss"],
            "avg_test_acc": results["avg_acc"],
            "avg_test_f1_micro": results["avg_f1_micro"],
            "avg_test_precision_micro": results["avg_precision_micro"],
            "avg_test_recall_micro": results["avg_recall_micro"],
            "avg_test_f1_macro": results["avg_f1_macro"],
            "avg_test_precision_macro": results["avg_precision_macro"],
            "avg_test_recall_macro": results["avg_recall_macro"],
            "total_samples": results["total_samples"]
        }
        return {
            "test_loss": results["avg_loss"],
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.weight"]
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
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        train_dataset = SequenceClassificationDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.train_dir,
            max_length=self.hparams.max_length,
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparams.per_device_train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=4,
        )

        # calculate total timesteps
        t_total = (
            (
                len(dataloader.dataset)
                // (
                    self.hparams.per_device_train_batch_size
                    * max(1, self.hparams.n_gpu)
                )
            )
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        # create scheduler
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = SequenceClassificationDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.valid_dir,
            max_length=self.hparams.max_length,
        )

        return DataLoader(
            val_dataset,
            batch_size=self.hparams.per_device_eval_batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        test_dataset = SequenceClassificationDataset(
            tokenizer=self.tokenizer,
            data_dir=self.hparams.test_dir,
            max_length=self.hparams.max_length,
        )

        return DataLoader(
            test_dataset,
            batch_size=self.hparams.per_device_eval_batch_size,
            num_workers=4,
        )
