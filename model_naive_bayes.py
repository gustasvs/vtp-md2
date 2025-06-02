#!/usr/bin/env python3
import os, ast, pickle
import sentencepiece as spm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from itertools import product

from data_preprocess import lemmatise, clean_up, SP_MODEL_NAME, PREPROCESSED_DATA_DIR

MODEL_PATH = "nb_ekman.pkl"
FOLDS = 3

ANALYSE = False  # set to True to run analysis on different ways to improve the model


sp = spm.SentencePieceProcessor(model_file=f"{SP_MODEL_NAME}.model")


def text_to_tokens(text):
    text = lemmatise(clean_up(text))
    ids = sp.encode(text, out_type=int) 
    return " ".join(map(str, ids))


def load_tsv(name):
    path = os.path.join(PREPROCESSED_DATA_DIR, f"{name}.tsv")
    df = pd.read_csv(path, sep="\t")

    df["token_str"] = df["tokenised_ids"].apply(
        lambda s: " ".join(map(str, ast.literal_eval(s)))
    )
    # return df[["token_str", "ekman_label"]]
    return df[["text", "clean_text", "token_str", "ekman_label"]]


def vectorise(text_series, vec=None):
    if vec is None:
        vec = CountVectorizer(token_pattern=r"\S+")
        X = vec.fit_transform(text_series)
        return X, vec
    else:
        return vec.transform(text_series)


def cross_validate(train_df):
    X_full, vec = vectorise(train_df["token_str"])
    y_full = train_df["ekman_label"].to_numpy()

    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_full), 1):
        clf = MultinomialNB(fit_prior=True)
        clf.fit(X_full[train_idx], y_full[train_idx])

        acc = clf.score(X_full[test_idx], y_full[test_idx])
        scores.append(acc)
        print(f"  fold {fold}: {acc:.3f}")

    print(f"Mean CV accuracy: {sum(scores)/len(scores):.3f}")
    return vec


def train_final(train_df, vec):
    X_train = vec.transform(train_df["token_str"])
    y_train = train_df["ekman_label"]
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, vec, test_df, name="test"):
    X_test = vec.transform(test_df["token_str"])
    y_test = test_df["ekman_label"]
    y_pred = clf.predict(X_test)

    print(f"\n=== {name.upper()} RESULTS ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


def save_model(vec, clf):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((vec, clf), f)
    print(f"[I] Model saved → {MODEL_PATH}")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def train():
    train_df = load_tsv("train")
    dev_df = load_tsv("dev")
    test_df = load_tsv("test")
    # make even MORE data for more precision hopefully
    train_df = pd.concat([train_df, dev_df]).reset_index(drop=True)

    # simple way to make joy not jump out just because it has more samples.
    train_df_original_len = len(train_df)

    max_n = train_df["ekman_label"].value_counts().max()
    balanced_parts = []

    for lbl, group in train_df.groupby("ekman_label"):
        if len(group) < max_n:
            upsampled = resample(
                group, replace=True, n_samples=max_n - len(group), random_state=42
            )
            balanced_parts.append(pd.concat([group, upsampled]))
            balanced_parts.append(group)

    train_df = pd.concat(balanced_parts).reset_index(drop=True)

    print(f"Oversampled from {train_df_original_len} to {len(train_df)} rows.")

    vec = cross_validate(train_df)

    clf = train_final(train_df, vec)

    evaluate(clf, vec, test_df, "test")
    save_model(vec, clf)


def inference():
    assert os.path.exists(MODEL_PATH), "train the model before using inference"

    vec, clf = load_model()

    print("Running inference for model", MODEL_PATH)
    while True:
        raw = input("> ").strip()
        if not raw:
            break

        seq = text_to_tokens(raw)

        X = vec.transform([seq])
        probs = clf.predict_proba(X)[0]

        for lbl, p in zip(clf.classes_, probs):
            print(f"{lbl:8}: {p:.3f}")
        print("PREDICTION:", clf.classes_[probs.argmax()], "\n")


ALPHAS = [0.5, 1.0, 2.0]
REPRESENTATIONS = ["token_str", "text", "clean_text"]
USE_BIGRAMS = [False, True]
USE_TFIDF = [False, True]


def run_analysis(df):
    best = None
    for alpha, rep, bigram, tfidf in product(
        ALPHAS, REPRESENTATIONS, USE_BIGRAMS, USE_TFIDF
    ):
        VecClass = TfidfVectorizer if tfidf else CountVectorizer
        vec = VecClass(token_pattern=r"\S+", ngram_range=(1, 2) if bigram else (1, 1))
        series = df[rep].fillna("")
        X = vec.fit_transform(series)
        y = df["ekman_label"].to_numpy()

        scores = []
        for tr, te in KFold(n_splits=FOLDS, shuffle=True, random_state=42).split(X):
            clf = MultinomialNB(alpha=alpha)
            clf.fit(X[tr], y[tr])
            scores.append(clf.score(X[te], y[te]))

        mean_acc = sum(scores) / len(scores)
        print(
            f"alpha={alpha:<3} rep={rep:<9} bigram={bigram} tfidf={tfidf}  acc={mean_acc:.3f}"
        )

        if best is None or mean_acc > best[0]:
            best = (mean_acc, vec, alpha, rep)

    print(f"\nBest mean CV accuracy: {best[0]:.3f}")
    return best[1], best[2], best[3]


if __name__ == "__main__":
    train_df = pd.concat([load_tsv("train"), load_tsv("dev")]).reset_index(drop=True)
    test_df = load_tsv("test")

    if ANALYSE:
        vec, a, text_col = run_analysis(train_df)

        # final fit on full training data with the best settings
        X_full = vec.fit_transform(train_df[text_col])
        clf = MultinomialNB(alpha=a).fit(X_full, train_df["ekman_label"])

        if text_col != "token_str":
            test_df = test_df.rename(columns={text_col: "token_str"})

        evaluate(clf, vec, test_df, "test")
        save_model(vec, clf)
    else:
        train()

    inference()
