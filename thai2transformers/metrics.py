import numpy as np
from seqeval.metrics import (accuracy_score as seqeval_accuracy_score, 
                             classification_report as seqeval_classification_report, 
                             f1_score as seqeval_f1_score,
                             precision_score as seqeval_precision_score, 
                             recall_score as seqeval_recall_score)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(pred, pred_labs=False):
    labels = pred.label_ids
    preds = pred.predictions if pred_labs else pred.predictions.argmax(-1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro")
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1_micro": f1_micro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "nb_samples": len(labels),
    }


def seqeval_classification_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    precision_macro = seqeval_precision_score(labels, preds, average="macro")
    recall_macro = seqeval_recall_score(labels, preds, average="macro")
    f1_macro = seqeval_f1_score(labels, preds, average="macro")
    precision_micro = seqeval_precision_score(labels, preds, average="micro")
    recall_micro = seqeval_recall_score(labels, preds, average="micro")
    f1_micro = seqeval_f1_score(labels, preds, average="micro")
    acc = seqeval_accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1_micro": f1_micro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "nb_samples": len(labels),
        "classification_report": seqeval_classification_report(labels, preds, digits=4),
    }
