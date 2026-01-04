import torch
import evaluate
from transformers import BartTokenizer, BartForConditionalGeneration
from preprocessing import load_summarization_data
from tqdm import tqdm

# -----------------------------
# Load ROUGE metric
# -----------------------------
rouge = evaluate.load("rouge")

# -----------------------------
# Load dataset
# -----------------------------
_, _, test_df = load_summarization_data(
    "Data/Summ/train.csv",
    "Data/Summ/validation.csv",
    "Data/Summ/test.csv"
)

# Clean data
test_df = test_df.dropna(subset=["clean_article", "clean_summary"])
test_df["clean_article"] = test_df["clean_article"].astype(str)
test_df["clean_summary"] = test_df["clean_summary"].astype(str)

# 🚀 LIMIT samples for fast evaluation
MAX_SAMPLES = 200   # change to full set only for final run
test_df = test_df.sample(n=min(MAX_SAMPLES, len(test_df)), random_state=42)

print("Test samples:", len(test_df))

# -----------------------------
# Load PRETRAINED BART (NO PATH ERRORS)
# -----------------------------
MODEL_NAME = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -----------------------------
# Batch inference (FAST)
# -----------------------------
BATCH_SIZE = 8   # increase if GPU allows

articles = test_df["clean_article"].tolist()
references = test_df["clean_summary"].tolist()
predictions = []

for i in tqdm(range(0, len(articles), BATCH_SIZE)):
    batch_articles = articles[i:i + BATCH_SIZE]

    inputs = tokenizer(
        batch_articles,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=2,        # faster than 4
            early_stopping=True
        )

    batch_summaries = tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True
    )

    predictions.extend(batch_summaries)

# -----------------------------
# ROUGE Evaluation
# -----------------------------
results = rouge.compute(
    predictions=predictions,
    references=references[:len(predictions)]
)

print("\nROUGE Evaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
