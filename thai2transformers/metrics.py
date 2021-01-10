import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


def classification_metrics(pred, pred_labs=False):
    labels = pred.label_ids
    preds = pred.predictions if pred_labs else pred.predictions.argmax(-1) 
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
        'nb_samples': len(labels)
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
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'nb_samples': len(labels),
    }
