import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def sequence_classification_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
    }