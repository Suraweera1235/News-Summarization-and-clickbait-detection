import streamlit as st
import torch
import pickle
import re
import os
import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from src.clickbait_lstm import ClickbaitLSTM

# ─────────────────────────────────────────────
# PATH FIX
# ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

# Clickbait model (PyTorch)
click_model = ClickbaitLSTM(0, 64, 128)  # placeholder, will be overwritten

click_model.load_state_dict(
    torch.load("models/clickbait/lstm_model.pt", map_location=DEVICE)
)
click_model.to(DEVICE)
click_model.eval()

with open("models/clickbait/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# ─────────────────────────────────────────────
# SUMMARIZATION MODEL (PRETRAINED BART) ⭐
# ─────────────────────────────────────────────
SUM_MODEL = "facebook/bart-large-cnn"

sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL).to(DEVICE)


# ─────────────────────────────────────────────
# CLICKBAIT PREDICTION
# ─────────────────────────────────────────────
def encode(text, max_len=20):
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    return ids + [0] * (max_len - len(ids))


def predict_clickbait(text):
    vec = torch.tensor([encode(text)]).to(DEVICE)

    with torch.no_grad():
        output = click_model(vec)
        prob = torch.sigmoid(output).item()

    label = "Clickbait" if prob >= 0.5 else "Not Clickbait"
    return label, prob


# ─────────────────────────────────────────────
# SUMMARIZATION FUNCTION (BART)
# ─────────────────────────────────────────────
def summarize_text(text):

    text = text[:1024]  # safety limit

    inputs = sum_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = sum_model.generate(
            **inputs,
            max_length=120,
            min_length=30,
            num_beams=4,
            length_penalty=1.2
        )

    return sum_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ─────────────────────────────────────────────
# UI CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="News AI System", layout="wide")

st.title("📰 News Summarization & Clickbait Detection")

# ─────────────────────────────────────────────
# DARK MODE (OPTIONAL UI ONLY)
# ─────────────────────────────────────────────
dark_mode = st.toggle("🌙 Dark Mode")


# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
user_input = st.text_area("Enter News Article or Headline", height=180)

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter text")
    else:

        clean_input = clean_text(user_input)

        # ── CLICKBAIT ──
        label, prob = predict_clickbait(clean_input)

        # ── SUMMARY ──
        summary = summarize_text(user_input)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Clickbait Detection")

            st.metric("Result", label)
            st.progress(float(prob))
            st.write(f"Confidence: {prob:.2f}")

        with col2:
            st.subheader("Summary")

            st.success(summary)