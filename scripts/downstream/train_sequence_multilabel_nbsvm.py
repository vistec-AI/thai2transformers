# #call example
# python train_sequence_multilabel_nbsvm.py --dataset_name_or_path prachathai67k\
#     --feature_col title --label_cols politics human_rights quality_of_life\
#     --metric_for_best_model f1_macro\
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
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted, check_X_y
from thai2transformers.metrics import classification_metrics


# thresholding
def best_threshold(y, probs):
    f1s = []
    for th in range(1, 100):
        f1s.append((th / 100, f1_score(y, (probs > (th / 100)).astype(int))))
    f1s_df = pd.DataFrame(f1s).sort_values(1, ascending=False).reset_index(drop=True)
    f1s_df.columns = ["th_label", "f1_label"]
    return f1s_df.th_label[0], f1s_df.f1_label[0]


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
        prog="train_sequence_multilabel_nbsvm.py", description="train nbsvm for multi-label sequence classification",
    )
    parser.add_argument("--dataset_name_or_path", type=str, default="prachathai67k")
    parser.add_argument("--feature_col", type=str, default="title")
    parser.add_argument(
        "--label_cols",
        nargs="+",
        default=[
            "politics",
            "human_rights",
            "quality_of_life",
            "international",
            "social",
            "environment",
            "economics",
            "culture",
            "labor",
            "national_security",
            "ict",
            "education",
        ],
    )
    parser.add_argument("--metric_for_best_model", type=str, default="f1_macro")
    parser.add_argument("--penalty", type=str, default="l2")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--tune_hyperparameters", default=False, type=lambda x: (str(x).lower() in ["true", "t"]),
    )
    parser.add_argument("--seed", type=int, default="1412")

    args = parser.parse_args()

    # load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name_or_path)
    print("Dataset loaded")

    # extract features
    print("Extracting features...")
    texts_train = dataset["train"][args.feature_col]
    texts_valid = dataset["validation"][args.feature_col]
    texts_test = dataset["test"][args.feature_col]

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
    y_train = np.array([dataset["train"][col] for col in args.label_cols]).transpose()
    y_valid = np.array([dataset["validation"][col] for col in args.label_cols]).transpose()
    y_test = np.array([dataset["test"][col] for col in args.label_cols]).transpose()
    print(f"train/valid/test labels: {y_train.shape}/{y_valid.shape}/{y_test.shape}")

    # hyperparameter tuning
    if args.tune_hyperparameters:
        hyperparams = []
        for p in ["l1", "l2"]:
            for c in range(1, 5):
                d = {"penalty": p, "C": c, "seed": seed}
                for i in range(y_valid.shape[1]):
                    if p == "l1":
                        model = NbSvmClassifier(penalty="l1", C=c, dual=False, seed=seed).fit(x_train, y_train[:, i])
                    else:
                        model = NbSvmClassifier(penalty="l2", C=c, dual=True, seed=seed).fit(x_train, y_train[:, i])
                    probs = model.predict_proba(x_valid)[:, 1]
                    d[f"th_label_{i}"], d[f"f1_label_{i}"] = best_threshold(y_valid[:, i], probs)
                d["f1_macro"] = np.mean([d[f"f1_label_{i}"] for i in range(y_valid.shape[1])])
                hyperparams.append(d)

        hyperparams_df = pd.DataFrame(hyperparams).sort_values("f1_macro", ascending=False).reset_index(drop=True)
        best_hyperparams = (
            hyperparams_df[["penalty", "C", "seed"] + [f"th_label_{i}" for i in range(y_valid.shape[1])]]
            .iloc[0, :]
            .to_dict()
        )
        print(f"Best hyperparameters: {best_hyperparams}")
    else:
        probs = np.zeros((x_test.shape[0], y_test.shape[1]))
        preds = np.zeros((x_test.shape[0], y_test.shape[1]))
        best_hyperparams = {"penalty": args.penalty, "C": args.C}
        best_hyperparams["dual"] = True if args.penalty == "l2" else False
        for i in range(y_valid.shape[1]):
            model = NbSvmClassifier(
                penalty=best_hyperparams["penalty"],
                C=best_hyperparams["C"],
                dual=best_hyperparams["dual"],
                seed=args.seed,
            ).fit(x_train, y_train[:, i])
            probs = model.predict_proba(x_valid)[:, 1]
            best_hyperparams[f"th_label_{i}"], _ = best_threshold(y_valid[:, i], probs)

    # test
    print("Testing...")
    probs = np.zeros((x_test.shape[0], y_test.shape[1]))
    preds = np.zeros((x_test.shape[0], y_test.shape[1]))
    for i in range(y_test.shape[1]):
        model = NbSvmClassifier(penalty=best_hyperparams["penalty"], C=best_hyperparams["C"], seed=args.seed).fit(
            x_train, y_train[:, i]
        )
        probs[:, i] = model.predict_proba(x_test)[:, 1]
        preds[:, i] = (probs[:, i] > best_hyperparams[f"th_label_{i}"]).astype(int)

    # micro
    micro_df = pd.DataFrame.from_dict(
        {
            "accuracy": (preds == y_test).mean(),
            "f1_micro": f1_score(y_test.reshape(-1), preds.reshape(-1)),
            "precision_micro": precision_score(y_test.reshape(-1), preds.reshape(-1)),
            "recall_micro": recall_score(y_test.reshape(-1), preds.reshape(-1)),
        },
        orient="index",
    )

    # macro
    test_performances = []
    for i in range(y_test.shape[1]):
        d = {}
        d["f1_macro"] = f1_score(y_test[:, i], preds[:, i])
        d["precision_macro"] = precision_score(y_test[:, i], preds[:, i])
        d["recall_macro"] = recall_score(y_test[:, i], preds[:, i])
        test_performances.append(d)
    macro_df = pd.DataFrame(pd.DataFrame(test_performances).mean())

    # merged
    test_df = pd.concat([micro_df, macro_df], 0)
    print(f"Test results: {test_df.to_dict()[0]}")


if __name__ == "__main__":
    main()
