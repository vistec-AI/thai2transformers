from typing import List, Dict, Union

import pandas as pd
import numpy as np
from datasets import load_metric
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report
from seqeval.metrics import (accuracy_score as seqeval_accuracy_score, 
                             classification_report as seqeval_classification_report, 
                             f1_score as seqeval_f1_score,
                             precision_score as seqeval_precision_score, 
                             recall_score as seqeval_recall_score)

from .conf import Task


def sk_classification_metrics(pred, pred_labs=False):
    result = classification_metrics(pred, pred_labs=pred_labs)
    labels = pred.label_ids
    preds = pred.predictions if pred_labs else pred.predictions.argmax(-1) 
    result['classification_report'] = classification_report(labels, preds, digits=4)
    return result


def classification_metrics(pred, pred_labs=False):
    labels = pred.label_ids
    preds = pred.predictions if pred_labs else pred.predictions.argmax(-1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro")
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'nb_samples': len(labels)
    }


def seqeval_classification_metrics(pred):
    from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    labels = pred.label_ids
    preds = pred.predictions
    precision_macro = precision_score(labels, preds, average='macro')
    recall_macro = recall_score(labels, preds, average='macro')
    f1_macro = f1_score(labels, preds, average='macro')
    precision_micro = precision_score(labels, preds, average='micro')
    recall_micro = recall_score(labels, preds, average='micro')
    f1_micro = f1_score(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'nb_samples': len(labels),
        'classification_report': classification_report(labels, preds, digits=4)
    }

def _compute_best_threshold(targets, probs):
    f1s = []
    for threshold in range(1,100):
        preds = (probs > (threshold / 100)).astype(int)
        f1s.append((
            threshold/100,
            f1_score(targets,
                     preds,
                     average='binary')
        ))

    f1s_df = pd.DataFrame(f1s).sort_values(1,ascending=False).reset_index(drop=True)
    f1s_df.columns = ['threshold_label','f1_label']

    return f1s_df.threshold_label[0], f1s_df.f1_label[0]

def _select_best_thresholds(targets, probs, n_labels):
    best_thresholds = dict()
    for i in range(0, n_labels):
        
        best_thresholds[f'label-{i}'] = _compute_best_threshold(targets[:,i], probs[:,i])
        
    return best_thresholds

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def multilabel_classification_metrics(pred, n_labels):

    labels = pred.label_ids
    logits = pred.predictions
    probs = sigmoid(logits)

    best_threshold_mapping = _select_best_thresholds(labels, probs, n_labels)
    best_thresholds = [ v[0] for k,v in best_threshold_mapping.items() ]

    preds = np.array(probs > best_thresholds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    accuracy_micro = (labels == preds).mean()

    return {
        'accuracy': acc,
        'accuracy_micro': accuracy_micro,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'nb_samples': len(labels)
    }


def chunk_level_classification_metrics(agg_chunk_labels, agg_chunk_preds):
    
    class LabelsPreds:
        label_ids = agg_chunk_labels
        predictions = agg_chunk_preds

    return seqeval_classification_metrics(LabelsPreds)


def token_level_classification_metrics(agg_chunk_labels, agg_chunk_preds):
    
    class LabelsPreds:
        label_ids = agg_chunk_labels
        predictions = agg_chunk_preds

    return sk_classification_metrics(LabelsPreds, pred_labs=True)

seq_metric = load_metric("seqeval")
def token_classification_metrics(pred, task: str, id2label: Dict[int, str]):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    if task == Task.CHUNK_LEVEL_CLS.value:
        results = seq_metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    else:
        result = token_level_classification_metrics(sum(true_labels, []),
                                                    sum(true_predictions, []))
        result = {k: v for k, v in result.items() if k != 'classification_report'}
        return result
