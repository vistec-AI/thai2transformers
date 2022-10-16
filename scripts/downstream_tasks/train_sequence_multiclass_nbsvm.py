# #call example
# python train_sequence_multiclass_nbsvm.py --dataset_name_or_path wisesight_sentiment\
#     --feature_col texts --label_col category --metric_for_best_model f1_micro\
#     --penalty l2 --C 2.0 --tune_hyperparameters false --seed 1412

import argparse

import numpy as np
import pandas as pd
from datasets import load_dataset
from pythainlp.ulmfit import process_thai
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted, check_X_y
from thai2transformers.metrics import classification_metrics


# model selection
def validation_f1(penalty, C, seed):
    probs = np.zeros((x_valid.shape[0], y_valid.shape[1]))
    for i in range(len(enc.categories_[0])):
        if penalty == "l1":
            model = NbSvmClassifier(penalty="l1", C=C, dual=False, seed=seed).fit(x_train, y_train[:, i])
        else:
            model = NbSvmClassifier(penalty="l2", C=C, dual=True, seed=seed).fit(x_train, y_train[:, i])
        probs[:, i] = model.predict_proba(x_valid)[:, 1]

        preds = probs.argmax(1)
    return f1_score(labels_valid, preds, average="micro")


# Implementation by Jeremy Howard (https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline)
# Class from AlexSánchez (https://www.kaggle.com/kniren)
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty="l2", C=1.0, dual=False, seed=1412):
        self.penalty = penalty
        self.C = C
        self.dual = dual
        self.seed = seed

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.toarray().ravel() if type(y) != np.ndarray else y.ravel()
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(
            penalty=self.penalty, C=self.C, dual=self.dual, solver="liblinear", random_state=self.seed,
        ).fit(x_nb, y)
        return self


def main():
    parser = argparse.ArgumentParser(
        prog="train_sequence_multiclass_nbsvm.py", description="train nbsvm for multi-class sequence classification",
    )
    parser.add_argument("--dataset_name_or_path", type=str, default="wisesight_sentiment")
    parser.add_argument("--feature_col", type=str, default="texts")
    parser.add_argument("--label_col", type=str, default="category")
    parser.add_argument("--metric_for_best_model", type=str, default="f1_micro")
    parser.add_argument("--penalty", type=str, default="l2")
    parser.add_argument("--C", type=float, default=3.0)
    parser.add_argument(
        "--tune_hyperparameters", default=False, type=lambda x: (str(x).lower() in ["true", "t"]),
    )
    parser.add_argument("--seed", type=int, default="1412")

    args = parser.parse_args()

    # load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name_or_path)

    # split for wongnai_reviews
    if args.dataset_name_or_path == "wongnai_reviews":
        train_val_split = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=2020)
        dataset["train"] = train_val_split["train"]
        dataset["validation"] = train_val_split["test"]

    # feature extraction for generated_reviews_enth
    if args.dataset_name_or_path == "generated_reviews_enth":
        texts_train = [i["th"] for i in dataset["train"][args.feature_col]]
        texts_valid = [i["th"] for i in dataset["validation"][args.feature_col]]
        texts_test = [i["th"] for i in dataset["test"][args.feature_col]]
    else:
        texts_train = dataset["train"][args.feature_col]
        texts_valid = dataset["validation"][args.feature_col]
        texts_test = dataset["test"][args.feature_col]
    print("Dataset loaded")

    # extract features
    print("Extracting features...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        tokenizer=process_thai,
        min_df=3,
        max_df=0.9,
        strip_accents="unicode",
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
    )

    x_train = tfidf.fit_transform(texts_train)
    x_valid = tfidf.transform(texts_valid)
    x_test = tfidf.transform(texts_test)
    print(f"train/valid/test features: {x_train.shape}/{x_valid.shape}/{x_test.shape}")

    # extract labels
    print("Extracting labels...")
    # feature extraction for review_star of generated_reviews_enth
    if args.dataset_name_or_path == "generated_reviews_enth" and args.label_col == "review_star":
        labels_train = [i - 1 for i in dataset["train"][args.label_col]]
        labels_valid = [i - 1 for i in dataset["validation"][args.label_col]]
        labels_test = [i - 1 for i in dataset["test"][args.label_col]]
    else:
        labels_train = dataset["train"][args.label_col]
        labels_valid = dataset["validation"][args.label_col]
        labels_test = dataset["test"][args.label_col]
    enc = OneHotEncoder(handle_unknown="ignore")
    y_train = enc.fit_transform(np.array(labels_train)[:, None])
    y_valid = enc.transform(np.array(labels_valid)[:, None])
    y_test = enc.transform(np.array(labels_test)[:, None])
    print(f"train/valid/test labels: {y_train.shape}/{y_valid.shape}/{y_test.shape}")

    # hyperparameter tuning
    if args.tune_hyperparameters:
        print("Tuning hyperparameters")
        hyperparams = []
        for p in ["l1", "l2"]:
            for c in range(1, 5):
                hyp = {
                    "dataset": args.dataset_name_or_path,
                    "penalty": p,
                    "C": c,
                    "f1_micro": validation_f1(p, c, seed=args.seed),
                }
                hyp["dual"] = True if p == "l2" else False
                hyperparams.append(hyp)
        hyperparams_df = pd.DataFrame(hyperparams).sort_values("f1_micro", ascending=False).reset_index(drop=True)
        best_hyperparams = hyperparams_df.drop(["f1_micro", "dataset"], 1).iloc[0, :].to_dict()
        print(f"Best hyperparameters: {best_hyperparams}")
    else:
        best_hyperparams = {"penalty": args.penalty, "C": args.C}
        best_hyperparams["dual"] = True if args.penalty == "l2" else False

    # test
    print("Testing...")
    probs = np.zeros((x_test.shape[0], y_test.shape[1]))
    for i in range(len(enc.categories_[0])):
        model = NbSvmClassifier(**best_hyperparams).fit(x_train, y_train[:, i])
        probs[:, i] = model.predict_proba(x_test)[:, 1]

    class Preds:
        label_ids = labels_test
        predictions = probs

    print(f"Test results: {classification_metrics(Preds)}")


if __name__ == "__main__":
    main()
