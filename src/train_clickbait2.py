import os
import pickle
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset

from clickbait_lstm import ClickbaitLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return True

        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ─────────────────────────────────────────────
# CLEAN TEXT
# ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# TRAIN FUNCTION
# ─────────────────────────────────────────────
def train():

    # Load datasets
    df1 = pd.read_csv("Data/Clickbait/train1.csv")
    df2 = pd.read_csv("Data/Clickbait/train2.csv")

    df1 = df1.rename(columns={"headline": "title", "clickbait": "label"})

    df2["label"] = df2["label"].astype(str).str.strip().str.lower()
    df2["label"] = df2["label"].replace({"clickbait": 1, "news": 0})

    df = pd.concat([df1, df2], ignore_index=True)

    df = df.dropna(subset=["title", "label"])
    df["label"] = df["label"].astype(int)

    df["title"] = df["title"].astype(str).apply(clean_text)
    df = df[df["title"].str.strip() != ""]

    print("Label distribution:")
    print(df["label"].value_counts())

    # ── SPLIT ─────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["title"], df["label"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    # ── VOCAB ─────────────────────────────
    counter = Counter(" ".join(X_train).split())

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= 2:
            vocab[word] = len(vocab)

    def encode(text, max_len=20):
        tokens = text.split()[:max_len]
        ids = [vocab.get(t, 1) for t in tokens]
        return ids + [0] * (max_len - len(ids))

    # Encode
    X_train_enc = torch.tensor([encode(t) for t in X_train])
    X_val_enc = torch.tensor([encode(t) for t in X_val])
    X_test_enc = torch.tensor([encode(t) for t in X_test])

    y_train = torch.tensor(y_train.values, dtype=torch.float)
    y_val = torch.tensor(y_val.values, dtype=torch.float)
    y_test = torch.tensor(y_test.values, dtype=torch.float)

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train_enc, y_train),
        batch_size=32,
        shuffle=True
    )

    # ── MODEL ─────────────────────────────
    model = ClickbaitLSTM(len(vocab), 64, 128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pos_weight = torch.tensor([
        len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    ]).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    early_stopping = EarlyStopping(patience=3)

    os.makedirs("models/clickbait", exist_ok=True)
    best_model_path = "models/clickbait/best_lstm1.pt"

    # ── TRAINING ───────────────────────────
    for epoch in range(5):

        model.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_enc.to(DEVICE))
            val_loss = criterion(val_outputs, y_val.to(DEVICE)).item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if early_stopping(val_loss):
            torch.save(model.state_dict(), best_model_path)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # ── LOAD BEST MODEL ────────────────────
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # ── FINAL TEST EVALUATION ──────────────
    print("\nEvaluating on TEST set...\n")

    with torch.no_grad():
        outputs = model(X_test_enc.to(DEVICE))

        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    y_true = y_test.numpy()

    clickbait_prob = probs
    non_clickbait_prob = 1 - probs

    print("Accuracy:", accuracy_score(y_true, preds))
    print("\nClassification Report:")
    print(classification_report(y_true, preds))

    print("\nSample Predictions:\n")

    for i in range(5):
        print(f"Sample {i+1}")
        print("Clickbait Prob:", clickbait_prob[i])
        print("Not Clickbait Prob:", non_clickbait_prob[i])
        print("Pred:", preds[i])
        print("True:", y_true[i])
        print("-" * 30)

    # ── SAVE FILES (MATCH UI) ──────────────
    torch.save(model.state_dict(), "models/clickbait/best1_lstm.pt")

    with open("models/clickbait/vocab1.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("models/clickbait/lstm_config1.pkl", "wb") as f:
        pickle.dump({"max_len": 20}, f)

    print("\n Training complete and model saved!")


if __name__ == "__main__":
    train()