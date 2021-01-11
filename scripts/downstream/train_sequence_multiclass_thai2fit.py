# #call example
# python train_sequence_multiclass_thai2fit.py --dataset_name_or_path wisesight_sentiment\
#     --feature_col texts --label_col category

import argparse

import numpy as np
import pandas as pd
from datasets import load_dataset
from fastai.callbacks import CSVLogger, SaveModelCallback
from fastai.text import *
from pythainlp.ulmfit import (THWIKI_LSTM, ThaiTokenizer, post_rules_th,
                              pre_rules_th, process_thai)
from thai2transformers.metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser(
        prog="train_sequence_multiclass_nbsvm.py", description="train nbsvm for multi-class sequence classification",
    )
    parser.add_argument("--dataset_name_or_path", type=str, default="wisesight_sentiment")
    parser.add_argument("--feature_col", type=str, default="texts")
    parser.add_argument("--label_col", type=str, default="category")
    parser.add_argument("--batch_size", type=int, default=64)

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

    # extract features and labels
    print("Extracting features and labels...")
    # feature extraction for review_star of generated_reviews_enth
    if args.dataset_name_or_path == "generated_reviews_enth" and args.label_col == "review_star":
        labels_train = [i - 1 for i in dataset["train"][args.label_col]]
        labels_valid = [i - 1 for i in dataset["validation"][args.label_col]]
        labels_test = [i - 1 for i in dataset["test"][args.label_col]]
    else:
        labels_train = dataset["train"][args.label_col]
        labels_valid = dataset["validation"][args.label_col]
        labels_test = dataset["test"][args.label_col]

    # df
    train_df = pd.DataFrame({"texts": texts_train, "labels": labels_train})
    valid_df = pd.DataFrame({"texts": texts_valid, "labels": labels_valid})
    test_df = pd.DataFrame({"texts": texts_test, "labels": labels_test})
    print(f"train/valid/test dfs: {train_df.shape}, {valid_df.shape}, {test_df.shape}")

    # LM databunch
    print("Creating LM databunch...")
    tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)

    processor = [
        TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
        NumericalizeProcessor(vocab=None, max_vocab=60000, min_freq=3),
    ]

    data_lm = (
        ItemLists(
            args.output_dir,
            train=TextList.from_df(train_df, args.output_dir, cols=["texts"], processor=processor),
            valid=TextList.from_df(valid_df, args.output_dir, cols=["texts"], processor=processor),
        )
        .label_for_lm()
        .databunch(bs=args.batch_size)
    )
    data_lm.sanity_check()
    data_lm.save(f"{args.dataset_name_or_path}_lm.pkl")
    print("Saved LM databunch to {args.dataset_name_or_path}_lm.pkl")

    # finetune LM
    print("Finetuning LM...")
    config = dict(
        emb_sz=400,
        n_hid=1550,
        n_layers=4,
        pad_token=1,
        qrnn=False,
        tie_weights=True,
        out_bias=True,
        output_p=0.25,
        hidden_p=0.1,
        input_p=0.2,
        embed_p=0.02,
        weight_p=0.15,
    )
    trn_args = dict(drop_mult=1.0, clip=0.12, alpha=2, beta=1)

    learn = language_model_learner(data_lm, AWD_LSTM, config=config, pretrained=False, **trn_args)

    # load pretrained models
    learn.load_pretrained(**THWIKI_LSTM)
    print("training frozen")
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))
    # train unfrozen
    print("training unfrozen")
    learn.unfreeze()
    learn.fit_one_cycle(5, 1e-3, moms=(0.8, 0.7))
    learn.save_encoder("lm_enc")
    print("Saved finetuned LM to lm_enc")

    # CLS databunch
    print("Creating CLS databunch")
    data_lm = load_data(args.output_dir, f"{args.dataset_name_or_path}_lm.pkl")
    data_lm.sanity_check()

    # classification data
    tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)
    processor = [
        TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
        NumericalizeProcessor(vocab=data_lm.vocab, max_vocab=60000, min_freq=3),
    ]

    data_cls = (
        ItemLists(
            args.output_dir,
            train=TextList.from_df(train_df, args.output_dir, cols=["texts"], processor=processor),
            valid=TextList.from_df(valid_df, args.output_dir, cols=["texts"], processor=processor),
        )
        .label_from_df("labels")
        .databunch(bs=args.batch_size)
    )

    data_cls.sanity_check()
    print(f"Created CLS data bunch with vocab size {len(data_cls.vocab.itos)}")

    # finetune CLS
    print("Finetuning CLS...")
    # model
    config = dict(
        emb_sz=400,
        n_hid=1550,
        n_layers=4,
        pad_token=1,
        qrnn=False,
        output_p=0.4,
        hidden_p=0.2,
        input_p=0.6,
        embed_p=0.1,
        weight_p=0.5,
    )
    trn_args = dict(bptt=70, drop_mult=0.7, alpha=2, beta=1, max_len=500)

    learn = text_classifier_learner(data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
    # load pretrained finetuned model
    learn.load_encoder("lm_enc")
    # train
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))
    learn.unfreeze()
    learn.fit_one_cycle(
        5,
        slice(1e-3 / (2.6 ** 4), 1e-3),
        moms=(0.8, 0.7),
        callbacks=[SaveModelCallback(learn, every="improvement", monitor="accuracy", name="bestmodel")],
    )

    # test
    #databunch
    data_lm = load_data(args.output_dir, f"{args.dataset_name_or_path}_lm.pkl")
    data_lm.sanity_check()

    #classification data
    tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)
    processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
                NumericalizeProcessor(vocab=data_lm.vocab, max_vocab=60000, min_freq=3)]

    data_cls = (ItemLists(args.output_dir, 
                train=TextList.from_df(train_df, args.output_dir, cols=["texts"], processor=processor),
                valid=TextList.from_df(test_df, args.output_dir, cols=["texts"], processor=processor),)
        .label_from_df("labels")
        .databunch(bs=args.batch_size)
        )

    data_cls.sanity_check()
    print(len(data_cls.vocab.itos))

    #model
    config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
                 output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
    trn_args = dict(bptt=70, drop_mult=0.7, alpha=2, beta=1, max_len=500)

    learn = text_classifier_learner(data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
    learn.load("bestmodel")

    # get predictions
    probs, y_true, loss = learn.get_preds(ds_type=DatasetType.Valid, ordered=True, with_loss=True)
    classes = learn.data.train_ds.classes
    y_true = np.array([classes[i] for i in y_true.numpy()])
    preds = np.array([classes[i] for i in probs.argmax(1).numpy()])
    prob = probs.numpy()
    loss = loss.numpy()

    class Preds:
        label_ids = y_true
        predictions = prob

    print(classification_metrics(Preds))


if __name__ == "__main__":
    main()
