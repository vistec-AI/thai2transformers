import os
import pandas as pd
import numpy as np
import collections
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report
from seqeval.metrics import (accuracy_score as seqeval_accuracy_score, 
                             classification_report as seqeval_classification_report, 
                             f1_score as seqeval_f1_score,
                             precision_score as seqeval_precision_score, 
                             recall_score as seqeval_recall_score)
from datasets import load_metric
from thai2transformers.preprocess import prepare_qa_validation_features
from thai2transformers.utils import get_thai2transformers_path

#tokenize
from pythainlp.tokenize import word_tokenize, syllable_tokenize
def character_tokenize(word): return [i for i in word]

def sk_classification_metrics(pred, pred_labs=False):
    result = classification_metrics(pred)
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

squad_newmm_metric = load_metric(os.path.join(get_thai2transformers_path(), 'squad_newmm'))


def _postprocess_qa_predictions(examples,
                               features, 
                               raw_predictions,
                               tokenizer,
                               n_best_size = 20, 
                               max_answer_length = 30,
                               allow_no_answer=False,
                               question_id_col='question_id'):
    
    #get start_logits and end_logits
    all_start_logits, all_end_logits = raw_predictions

    #get `offset_mapping` and `example_id` back
    features.set_format(type=features.format["type"], columns=list(features.features.keys()))

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples[question_id_col])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        context = example["context"]

        min_null_score = None
        valid_answers = []
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            #debug
            input_ids = features[feature_index]['input_ids'] 

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index-1] is None #end_index is the exclusive upperbound
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index-1][1] #end_index is the exclusive upperbound
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text_decode": tokenizer.decode(input_ids[start_index:end_index]),
                            "text": context[start_char:end_char],
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid failure.
            best_answer = {"text": "", "score": 0.0}
            
        if not allow_no_answer:
            predictions[example[question_id_col]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example[question_id_col]] = answer

    return predictions

def question_answering_metrics(datasets,
                               trainer,
                               metric=squad_newmm_metric,
                               tok_func=word_tokenize,
                               n_best_size=20,
                               max_answer_length=100,
                               allow_no_answer=False,
                               question_col='question',
                               context_col='context',
                               question_id_col='question_id',
                               answers_col='answers',
                               text_col='text',
                               start_col='answer_start',
                               pad_on_right=True,
                               max_length=416,
                               doc_stride=128):
    
    validation_features = datasets.map(
        lambda examples: prepare_qa_validation_features(examples=datasets, 
                           tokenizer=trainer.tokenizer,
                           question_col=question_col,
                           context_col=context_col,
                           question_id_col = question_id_col,
                           pad_on_right=pad_on_right,
                           max_length=max_length,
                           doc_stride=doc_stride),
        batched=True,
        remove_columns=datasets.column_names
    )
    
    #logits for start and end
    pred = trainer.predict(validation_features)
    
    #question_id, answer text format
    question_id_predictions = _postprocess_qa_predictions(examples=datasets,
                               features=validation_features, 
                               raw_predictions=pred.predictions,
                               tokenizer=trainer.tokenizer,
                               n_best_size = n_best_size, 
                               max_answer_length = max_answer_length,
                               allow_no_answer = allow_no_answer,
                               question_id_col=question_id_col)


    #format to have same field names as squad
    formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in question_id_predictions.items()]

    #format to have same field names as squad
    references = [{"id": str(ex[question_id_col]), 
                   "answers": {'text': ex[answers_col][text_col],
                               'answer_start':ex[answers_col][start_col]}} for ex in datasets]

    return metric.compute(predictions=formatted_predictions, references=references, tok_func=tok_func), formatted_predictions, references
