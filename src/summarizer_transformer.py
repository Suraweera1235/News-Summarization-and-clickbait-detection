import os
import re
import string
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "facebook/bart-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z. ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─────────────────────────────────────────────
# LOAD DATA (train/val/test)
# Expect columns: "text", "summary"
# ─────────────────────────────────────────────
train_df = pd.read_csv("data/Summ/train.csv")
val_df   = pd.read_csv("data/Summ/validation.csv")
test_df  = pd.read_csv("data/Summ/test.csv")

for df in [train_df, val_df, test_df]:
        df['text'] = df['article'].astype(str).apply(clean_text)
        df['summary'] = df['highlights'].astype(str).apply(clean_text)

train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)
test_ds  = Dataset.from_pandas(test_df)

# ─────────────────────────────────────────────
# TOKENIZER + MODEL
# ─────────────────────────────────────────────
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

max_input_length = 512
max_target_length = 128

# ─────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────
def preprocess(batch):
    inputs = tokenizer(
        batch["text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        batch["summary"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = targets["input_ids"]
    return inputs

train_ds = train_ds.map(preprocess, batched=True)
val_ds   = val_ds.map(preprocess, batched=True)
test_ds  = test_ds.map(preprocess, batched=True)

# ─────────────────────────────────────────────
# DATA COLLATOR
# ─────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ─────────────────────────────────────────────
# ROUGE METRIC
# ─────────────────────────────────────────────
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = torch.where(
        torch.tensor(labels) != -100,
        torch.tensor(labels),
        tokenizer.pad_token_id
    )

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }

# ─────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./bart_summarizer",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=50
)

# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
trainer.train()

# ─────────────────────────────────────────────
# FINAL TEST EVALUATION
# ─────────────────────────────────────────────
print("\nEvaluating on TEST set...\n")

test_results = trainer.evaluate(test_ds)
print(test_results)

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
os.makedirs("models/summarizer", exist_ok=True)
trainer.save_model("models/summarizer")
tokenizer.save_pretrained("models/summarizer")

print("\n✅ Training complete. Model saved.")