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
            return

        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ─────────────────────────────────────────────
# TEXT CLEANING
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
# TRAINING
# ─────────────────────────────────────────────
def train():

    # Load datasets
    df1 = pd.read_csv("Data/Clickbait/train1.csv")
    df2 = pd.read_csv("Data/Clickbait/train2.csv")

    df1 = df1.rename(columns={"headline": "title", "clickbait": "label"})

    df2["label"] = df2["label"].astype(str).str.strip().str.lower()
    df2["label"] = df2["label"].replace({"clickbait": 1, "news": 0})

    # Combine properly
    df = pd.concat([df1, df2], ignore_index=True)

    df = df.dropna(subset=["title", "label"])
    df["label"] = df["label"].astype(int)

    df["title"] = df["title"].astype(str).apply(clean_text)
    df = df[df["title"].str.strip() != ""]

    print("Label distribution:")
    print(df["label"].value_counts())

    # ── TRAIN / TEST SPLIT ─────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["title"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # ── VOCAB (FIXED - frequency based) ───────────
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
    X_test_enc = torch.tensor([encode(t) for t in X_test])

    y_train = torch.tensor(y_train.values, dtype=torch.float)
    y_test = torch.tensor(y_test.values, dtype=torch.float)

    # ── DATA LOADERS (IMPORTANT FIX) ─────────────
    train_dataset = TensorDataset(X_train_enc, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # ── MODEL ────────────────────────────────────
    model = ClickbaitLSTM(len(vocab), 64, 128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── CLASS WEIGHTING (CRITICAL FIX) ──────────
    pos_weight = torch.tensor([
        len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    ]).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    early_stopping = EarlyStopping(patience=3)

    # ── TRAINING LOOP ───────────────────────────
    for epoch in range(25):

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

        # ── VALIDATION ───────────────────────────
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_enc.to(DEVICE))
            val_loss = criterion(val_outputs, y_test.to(DEVICE)).item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        early_stopping(val_loss)

        if early_stopping.early_stop:
            print(f"\n⛔ Early stopping at epoch {epoch+1}")
            break

    # ── FINAL EVALUATION ───────────────────────
    print("\nEvaluating model...\n")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_enc.to(DEVICE))
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    y_true = y_test.numpy()

    print("Accuracy:", accuracy_score(y_true, preds))
    print("\nClassification Report:")
    print(classification_report(y_true, preds))

    # ── SAVE MODEL ──────────────────────────────
    os.makedirs("models/clickbait", exist_ok=True)

    torch.save(model.state_dict(), "models/clickbait/lstm_model.pt")

    with open("models/clickbait/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("models/clickbait/lstm_config.pkl", "wb") as f:
        pickle.dump({"max_len": 20}, f)

    print("\n✅ Training complete and model saved!")


if __name__ == "__main__":
    train()