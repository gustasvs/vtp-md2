import os, json, re
import sentencepiece as spm
import pandas as pd
import spacy
import en_core_web_sm

from collections import Counter

VOCAB_SIZE = 8_000


def load_txt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tsv(filename):
    return pd.read_csv(
        filename, sep="\t", header=None, names=["text", "label", "id"], encoding="utf-8"
    )


def filter_dataset(df, name="ds"):
    # we dont need samples that have multiple emotions
    original_len = len(df)
    print(
        f"Removing multiple emotions from dataset {(name + ':').ljust(6)} from {original_len} to",
        end=" ",
    )
    df = df[df["label"].apply(lambda x: len(x.split(",")) == 1)]
    print(f"{len(df)} samples, {len(df) / original_len * 100:.2f}% remaining")
    return df.reset_index(drop=True)


def map_emotions_to_ekman(df, ekman_mapping, emotions_list):
    
    
    code2emotion = {str(i): e for i, e in enumerate(emotions_list)}

    reverse_map = {
        fine: ekman for ekman, fines in ekman_mapping.items() for fine in fines
    }
    reverse_map["neutral"] = "neutral"

    df["emotion"] = df["label"].map(code2emotion)
    df["ekman_label"] = df["emotion"].map(reverse_map)

    return df


def analyze_dataset(df, name="ds"):
    print(f"\n{'=' * 30}")
    print(f" {name} dataset analysis ".center(30, "="), end="\n\n")

    emo_counts = Counter(df["ekman_label"])
    total = sum(emo_counts.values())
    print("Emotion distribution:")
    for emo, cnt in emo_counts.most_common():
        print(f"  {emo}: {cnt} ({cnt/total*100:.2f}%)")
    
    def tokenize(text):
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if t.isalpha()]
    
    all_tokens = []
    lengths = []
    tokens_by_emo = {emo: [] for emo in emo_counts}
    
    for _, row in df.iterrows():
        toks = tokenize(row["text"])
        all_tokens.extend(toks)
        tokens_by_emo[row["ekman_label"]].extend(toks)
        lengths.append(len(toks))
    
    global_freq = Counter(all_tokens)
    print("\nTop 20 tokens overall:")
    for tok, freq in global_freq.most_common(20):
        print(f"  {tok}: {freq}")
    
    avg_len = sum(lengths) / len(lengths)
    ttr = len(global_freq) / len(all_tokens)
    print(f"\nAverage tokens/sample: {avg_len:.2f}")
    print(f"Type to token ratio: {ttr:.4f}")
    
    print("\nTop 10 tokens per Ekman category:")
    for emo, toks in tokens_by_emo.items():
        top10 = Counter(toks).most_common(10)
        words = ", ".join(w for w,_ in top10)
        print(f"  {emo}: {words}")


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

def clean_up(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)              # URLs
    text = re.sub(r"@\w+", " ", text)                               # @mentions
    text = re.sub(r"\b[\w\.-]+?@[\w\.-]+\.\w{2,4}\b", " ", text)    # emails
    text = re.sub(r"[^\w\s]", " ", text)                            # punctuation / emojis
    text = re.sub(r"\d+", "<NUM>", text)                            # numbers
    text = re.sub(r"\s+", " ", text).strip()                        # extra spaces 
    return text

def lemmatise(text: str) -> str:
    doc = nlp(text)
    return " ".join(t.lemma_ for t in doc)

def preprocess_text_for_models(dataframes):
    # accepts all dataframes returning the same order but preprocessed

    # write all text to a temporary file for SentencePiece training
    for df in dataframes:
        df["clean_text"] = df["text"].apply(lambda text: lemmatise(clean_up(text)))
    print("Cleaned text for training:")
    for df in dataframes:
        print(f"  {df['clean_text'].head(5).to_list()}")
    
    with open("tmp_corpus.txt", "w", encoding="utf-8") as f:
        for df in dataframes:
            for text in df["clean_text"]:
                f.write(text + "\n")
    
    spm.SentencePieceTrainer.Train(
        input="tmp_corpus.txt",
        model_prefix="bpe_model",
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0
    )
    os.remove("tmp_corpus.txt")
    sp = spm.SentencePieceProcessor(model_file="bpe_model.model")

    df["input_ids"] = df["clean_text"].apply(lambda t: sp.encode(t, out_type=int))
    
    return df


def preprocess_data():
    # load data from files
    emotions_list = load_txt("data/emotions.txt")
    ekman_mapping = load_json("data/ekman_mapping.json")
    ekman_mapping["neutral"] = ["neutral"]
    dev_data = load_tsv("data/dev.tsv")
    test_data = load_tsv("data/test.tsv")
    train_data = load_tsv("data/train.tsv")

    # remove samples with multiple emotions
    dev_data = filter_dataset(dev_data, name="dev")
    test_data = filter_dataset(test_data, name="test")
    train_data = filter_dataset(train_data, name="train")

    # map emotions to Ekman categories
    dev_data = map_emotions_to_ekman(dev_data, ekman_mapping, emotions_list)
    test_data = map_emotions_to_ekman(test_data, ekman_mapping, emotions_list)
    train_data = map_emotions_to_ekman(train_data, ekman_mapping, emotions_list)

    print("first 5 samples in dev set:")
    for i, row in dev_data.head().iterrows():
        print(f"  {i}: {row['text']} -> {row['ekman_label']}")
        print(row)
    
    analyze_dataset(dev_data, name="dev")
    analyze_dataset(test_data, name="test")
    analyze_dataset(train_data, name="train")

    # preprocess text for models
    [dev_data, test_data, train_data] = preprocess_text_for_models(
        [dev_data, test_data, train_data]
    )

    print("first 5 samples in dev set after preprocessing:")
    for i, row in dev_data.head().iterrows():
        # print(f"  {i}: {row['text']} -> {row['ekman_label']}")
        print(row)
    


if __name__ == "__main__":
    preprocess_data()
