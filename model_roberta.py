import os, torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_preprocess import clean_up, PREPROCESSED_DATA_DIR
import evaluate
from sklearn.metrics import confusion_matrix, classification_report

accuracy_metric = evaluate.load("accuracy")

from transformers import logging
logging.set_verbosity_error()  # suppress warnings from transformers


# MODEL_NAME = "roberta-base"
MODEL_NAME = "SamLowe/roberta-base-go_emotions"
OUTPUT_DIR = "roberta"
EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
NUM_LABELS = len(EKMAN_LABELS)
BATCH_SIZE = 64
N_EPOCHS = 5
LR = 1.3e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAYERS_TO_FREEZE = 10  # number of layers to freeze in the RoBERTa model

LABEL2ID = {lbl: i for i, lbl in enumerate(EKMAN_LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}


def load_tsv(name: str) -> pd.DataFrame:
    path = os.path.join(PREPROCESSED_DATA_DIR, f"{name}.tsv")
    df = pd.read_csv(path, sep="\t")
    df["clean_text"] = df["clean_text"].fillna("").apply(clean_up)
    df["labels"] = df["ekman_label"].map(LABEL2ID).astype(int)
    return df[["clean_text", "labels"]]


def to_hf_dataset(df: pd.DataFrame, tokenizer):
    def _tokenise(batch):
        return tokenizer(
            batch["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    ds = Dataset.from_pandas(df)
    ds = ds.map(_tokenise, batched=True, remove_columns=["clean_text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def train_roberta():
    torch.manual_seed(SEED)

    # train_df = pd.concat([load_tsv("train"), load_tsv("dev")]).reset_index(drop=True)
    train_df = load_tsv("train")
    test_df = load_tsv("test")

    # tokenizer / datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds = to_hf_dataset(train_df, tokenizer)
    test_ds = to_hf_dataset(test_df, tokenizer)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        # OUTPUT_DIR, # uncomment to train the previous model more
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False


    if LAYERS_TO_FREEZE > 0:
        for layer in model.roberta.encoder.layer[:LAYERS_TO_FREEZE]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"disabling {LAYERS_TO_FREEZE} out of {len(model.roberta.encoder.layer)} layers")


    model.to(DEVICE)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator,
    )
    eval_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(N_EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        running_loss = 0.0
        for batch in pbar:
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", running_loss=f"{(running_loss / (pbar.n + 1)):.4f}")


        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in eval_loader:
                labels = batch.pop("labels").to(DEVICE)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = logits.argmax(dim=-1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        print(f"Eval accuracy: {correct/total:.4%}")

        model.save_pretrained(OUTPUT_DIR)
        print(f"Model and tokenizer saved to {OUTPUT_DIR}")

    
    # Save the model and tokenizer

def inference(model_path: str = OUTPUT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    test_ds = load_tsv("test")
    test_ds = to_hf_dataset(test_ds, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    eval_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=data_collator)
    
    model.to(DEVICE)
    model.eval()
    
    total, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print(f"Eval accuracy: {correct/total:.4%}")
    all_labels = [ID2LABEL[label] for label in all_labels]
    all_preds = [ID2LABEL[pred] for pred in all_preds]
    print("Classification report:\n", classification_report(all_labels, all_preds, digits=3))

    print("Starting inference loop. Type 'enter' to stop.")
    while True:
        user_input = input(">> ").strip()
        if user_input == "":
            print("Exiting inference loop.")
            break

        encodings = tokenizer(
            [user_input],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

        with torch.no_grad():
            logits = model(**encodings).logits
            pred_label = ID2LABEL[logits.argmax(dim=-1).item()]

        print(f"Prediction: {pred_label}")

if __name__ == "__main__":
    # inference()
    train_roberta()
