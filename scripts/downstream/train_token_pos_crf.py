# call example
# python train_token_pos_crf.py --dataset_name_or_path thainer --feature_col tokens\
#     --label_col ner_tags --metric_for_best_model f1_macro --c1 1.0 --c2 0.0

import argparse
import random

import numpy as np
import pandas as pd
import pycrfsuite
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from thai2transformers.metrics import classification_metrics
from tqdm.auto import tqdm


def extract_features(doc, window=3, max_n_gram=3):
    # padding for words
    doc = ["xxpad" for i in range(window)] + doc + ["xxpad" for i in range(window)]
    doc_features = []

    # for each word
    for i in range(window, len(doc) - window):
        # bias term
        word_features = ["bias"]

        # ngram features
        for n_gram in range(1, min(max_n_gram + 1, 2 + window * 2)):
            for j in range(i - window, i + window + 2 - n_gram):
                feature_position = f"{n_gram}_{j-i}_{j-i+n_gram}"

                # word
                word_ = f'{"|".join(doc[j:(j+n_gram)])}'
                word_features += [f"word_{feature_position}={word_}"]

        # append to feature per word
        doc_features.append(word_features)
    return doc_features


def generate_xy(all_tuples):
    # target
    y = [[str(l) for (w, l) in t] for t in all_tuples]
    # features
    x_pre = [[w for (w, l) in t] for t in all_tuples]
    x = [extract_features(x_, window=2, max_n_gram=2) for x_ in tqdm(x_pre)]
    return x, y


def train_crf(model_name, c1, c2, x_train, y_train, max_iterations=500):
    # Train model
    trainer = pycrfsuite.Trainer(verbose=True)

    for xseq, yseq in tqdm(zip(x_train, y_train)):
        trainer.append(xseq, yseq)

    trainer.set_params(
        {
            "c1": c1,
            "c2": c2,
            "max_iterations": max_iterations,
            "feature.possible_transitions": True,
            "feature.minfreq": 3.0,
        }
    )

    trainer.train(f"{model_name}_{c1}_{c2}.model")


def evaluate_crf(model_path, features, labels, tag_labels):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)
    y_pred = []
    for xseq in tqdm(features, total=len(features)):
        y_pred.append(tagger.tag(xseq))
    preds = [int(tag) for row in y_pred for tag in row]
    labs = [int(tag) for row in labels for tag in row]
    return (
        classification_report(labs, preds, target_names=tag_labels, digits=4),
        f1_score(labs, preds, average="micro"),
        f1_score(labs, preds, average="macro"),
    )


def main():
    parser = argparse.ArgumentParser(
        prog="train_token_classification_crf.py", description="train crf for token classification",
    )
    parser.add_argument("--dataset_name_or_path", type=str, default="thainer")
    parser.add_argument("--feature_col", type=str, default="tokens")
    parser.add_argument("--label_col", type=str, default="ner_tags")
    parser.add_argument("--metric_for_best_model", type=str, default="f1_macro")
    parser.add_argument("--c1", type=float, default=1.0)
    parser.add_argument("--c2", type=float, default=0.0)
    parser.add_argument("--data_dir", type=str, default="path/to/local_folder")
    parser.add_argument(
        "--tune_hyperparameters", default=False, type=lambda x: (str(x).lower() in ["true", "t"]),
    )
    parser.add_argument("--seed", type=int, default=2020)

    args = parser.parse_args()

    # load dataset
    print("Loading dataset...")
    if args.dataset_name_or_path == "lst20":
        dataset = load_dataset(args.dataset_name_or_path, data_dir=args.data_dir)
    else:
        dataset = load_dataset(args.dataset_name_or_path)

    if args.dataset_name_or_path == "thainer" and args.label_col == "ner_tags":
        dataset = dataset.map(
            lambda examples: {"ner_tags": [i if i not in [13, 26] else 27 for i in examples[args.label_col]]}
        )
        train_valtest_split = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=args.seed)
        dataset["train"] = train_valtest_split["train"]
        dataset["validation"] = train_valtest_split["test"]
        val_test_split = dataset["validation"].train_test_split(test_size=0.5, shuffle=True, seed=args.seed)
        dataset["validation"] = val_test_split["train"]
        dataset["test"] = val_test_split["test"]
        tag_labels = dataset["train"].features[args.label_col].feature.names
        tag_labels = [tag_labels[i] for i in range(len(tag_labels)) if i not in [13, 26]]
    elif args.dataset_name_or_path == "thainer" and args.label_col == "pos_tags":
        train_valtest_split = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=args.seed)
        dataset["train"] = train_valtest_split["train"]
        dataset["validation"] = train_valtest_split["test"]
        val_test_split = dataset["validation"].train_test_split(test_size=0.5, shuffle=True, seed=args.seed)
        dataset["validation"] = val_test_split["train"]
        dataset["test"] = val_test_split["test"]
        tag_labels = dataset["train"].features[args.label_col].feature.names
    else:
        tag_labels = dataset["train"].features[args.label_col].feature.names

    if args.dataset_name_or_path == 'thainer':
        from transformers import AutoTokenizer
        mbert_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        def pre_tokenize(token, space_token):
            token = token.replace(' ', space_token)
            return token
        def is_not_too_long(example,
                            max_length=510):
            tokens = sum([mbert_tokenizer.tokenize(
                pre_tokenize(token, space_token='<_>'))
                          for token in example[args.feature_col]], [])
            return len(tokens) < max_length
        dataset['test'] = dataset['test'].filter(is_not_too_long)
    print("Dataset loaded")

    # extract features
    print("Extracting features and labels...")

    def generate_sents(dataset, idx):
        features = dataset[idx][args.feature_col]
        labels = dataset[idx][args.label_col]
        return [(features[i], labels[i]) for i in range(len(features))]

    train_sents = [generate_sents(dataset["train"], i) for i in range(len(dataset["train"]))]
    valid_sents = [generate_sents(dataset["validation"], i) for i in range(len(dataset["validation"]))]
    test_sents = [generate_sents(dataset["test"], i) for i in range(len(dataset["test"]))]

    x_train, y_train = generate_xy(train_sents)
    if args.dataset_name_or_path == "lst20":
        random.seed(args.seed)
        x_train_small = random.sample(x_train, 10000)
        random.seed(args.seed)
        y_train_small = random.sample(y_train, 10000)
    else:
        x_train_small = x_train
        y_train_small = y_train
    x_valid, y_valid = generate_xy(valid_sents)
    x_test, y_test = generate_xy(test_sents)
    print(f"train features/labels: {len(x_train)}/{len(y_train)}")
    print(f"valid features/labels: {len(x_valid)}/{len(y_valid)}")
    print(f"test features/labels: {len(x_test)}/{len(y_test)}")

    # hyperparameter tuning
    if args.tune_hyperparameters:
        hyperparams = []
        for c1 in tqdm([0.0, 0.5, 1.0]):
            for c2 in tqdm([0.0, 0.5, 1.0]):
                train_crf(args.dataset_name_or_path, c1, c2, x_train_small, y_train_small)
                report, f1_micro, f1_macro = evaluate_crf(
                    f"{args.dataset_name_or_path}_{c1}_{c2}.model", x_valid, y_valid, tag_labels
                )
                print(report)
                d = {"c1": c1, "c2": c2, "f1_micro": f1_micro, "f1_macro": f1_macro, "report": report}
                hyperparams.append(d)
        hyperparams_df = pd.DataFrame(hyperparams).sort_values("f1_macro", ascending=False).reset_index(drop=True)
        best_hyperparams = hyperparams_df.iloc[0, :].to_dict()
    else:
        best_hyperparams = {"c1": args.c1, "c2": args.c2}

    # test
    c1, c2 = best_hyperparams["c1"], best_hyperparams["c2"]
    train_crf(f"{args.dataset_name_or_path}_{args.label_col}_best", c1, c2, x_train, y_train)
    if args.dataset_name_or_path == "lst20" and args.label_col == "ner_tags":
        report, f1_micro, f1_macro = evaluate_crf(
            f"{args.dataset_name_or_path}_{args.label_col}_best_{c1}_{c2}.model", x_test, y_test, tag_labels[:-1]
        )  # test set of lst20 does not have E_TTL
        print(report)
    else:
        report, f1_micro, f1_macro = evaluate_crf(
            f"{args.dataset_name_or_path}_{args.label_col}_best_{c1}_{c2}.model", x_test, y_test, tag_labels
        )
        print(report)


if __name__ == "__main__":
    main()
