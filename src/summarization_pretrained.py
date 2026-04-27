import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "t5-small"
MAX_SAMPLES = 2000
BATCH_SIZE = 8
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# CLEAN TEXT
# ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z. ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_local_csv(path):
    df = pd.read_csv(path)

    df['text'] = df['article'].astype(str).apply(clean_text)
    df['summary'] = df['highlights'].astype(str).apply(clean_text)

    df = df.dropna(subset=["text", "summary"])
    return df[["text", "summary"]]

test_df = load_local_csv("Data/Summ/test.csv")
test_df = test_df.head(MAX_SAMPLES)

print(f"Evaluating on {len(test_df)} samples")

# ─────────────────────────────────────────────
# LOAD MODEL (NO TRAINING)
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# ─────────────────────────────────────────────
# GENERATE SUMMARIES (BATCHED)
# ─────────────────────────────────────────────
predictions = []
references = []

for i in range(0, len(test_df), BATCH_SIZE):
    batch = test_df.iloc[i:i+BATCH_SIZE]

    inputs = ["summarize: " + text for text in batch["text"].tolist()]

    encodings = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **encodings,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=4
        )

    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions.extend(preds)
    references.extend(batch["summary"].tolist())

    print(f"Processed {i + len(batch)} / {len(test_df)}")

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
rouge = evaluate.load("rouge")

results = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True
)

print("\n📊 ROUGE RESULTS:")
print(f"ROUGE-1: {results['rouge1']:.4f}")
print(f"ROUGE-2: {results['rouge2']:.4f}")
print(f"ROUGE-L: {results['rougeL']:.4f}")

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

with open("results/rouge_scores.txt", "w") as f:
    f.write(str(results))

print("\n✅ Evaluation complete. Results saved.")