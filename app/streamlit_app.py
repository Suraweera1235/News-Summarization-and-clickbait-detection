import streamlit as st
import torch
import pickle
import re
import string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.clickbait_lstm import ClickbaitLSTM


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLICKBAIT_MODEL_PATH = "models/clickbait/best1_lstm.pt"
VOCAB_PATH = "models/clickbait/vocab1.pkl"
CONFIG_PATH = "models/clickbait/lstm_config1.pkl"

SUM_MODEL_NAME = "t5-small"


# ─────────────────────────────
# CLEAN TEXT
# ─────────────────────────────
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────
# CLICKBAIT MODEL (CACHE)
# ─────────────────────────────
@st.cache_resource
def load_clickbait_model():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    model = ClickbaitLSTM(len(vocab), 64, 128)
    model.load_state_dict(torch.load(CLICKBAIT_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, vocab, config


# ─────────────────────────────
# SUMMARIZER (CACHE)
# ─────────────────────────────
@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ─────────────────────────────
# ENCODE TEXT
# ─────────────────────────────
def encode(text, vocab, max_len=20):
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    return ids + [0] * (max_len - len(ids))


# ─────────────────────────────
# CLICKBAIT PREDICTION (FIXED)
# ─────────────────────────────
def predict_clickbait(text, model, vocab):
    cleaned = clean_text(text)
    encoded = torch.tensor([encode(cleaned, vocab)]).to(DEVICE)

    with torch.no_grad():
        output = model(encoded)
        prob = torch.sigmoid(output).item()

    clickbait_prob = prob
    non_clickbait_prob = 1 - prob

    label = "Clickbait" if clickbait_prob > 0.5 else "Not Clickbait"

    return clickbait_prob, non_clickbait_prob, label


# ─────────────────────────────
# SUMMARIZATION
# ─────────────────────────────
def summarize(article, tokenizer, model):
    inputs = tokenizer(
        "summarize: " + article,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=64,
            num_beams=2,
            early_stopping=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ─────────────────────────────
# UI
# ─────────────────────────────
st.set_page_config(page_title="News AI System", layout="wide")

st.title("📰 News Summarization + Clickbait Detection System")

tab1, tab2 = st.tabs(["📌 Clickbait Detection", "📝 Text Summarization"])


# ─────────────────────────────
# LOAD MODELS ONCE
# ─────────────────────────────
click_model, vocab, config = load_clickbait_model()
tokenizer, sum_model = load_summarizer()


# ─────────────────────────────
# CLICKBAIT TAB
# ─────────────────────────────
with tab1:
    st.subheader("Detect whether a headline is clickbait")

    text = st.text_area("Enter News Headline")

    if st.button("Predict Clickbait"):

        if text.strip():

            click_prob, non_click_prob, label = predict_clickbait(
                text, click_model, vocab
            )

            st.success(label)

            # ── IMPROVED DISPLAY ──
            st.write("### Confidence Breakdown")
            st.write(f"🟥 Clickbait Probability: {click_prob:.2f}")
            st.write(f"🟩 Not Clickbait Probability: {non_click_prob:.2f}")

        else:
            st.warning("Please enter a headline")


# ─────────────────────────────
# SUMMARIZATION TAB
# ─────────────────────────────
with tab2:
    st.subheader("Generate summary of news article")

    article = st.text_area("Enter News Article", height=250)

    if st.button("Generate Summary"):

        if article.strip():
            summary = summarize(article, tokenizer, sum_model)
            st.info(summary)
        else:
            st.warning("Please enter article text")


# ─────────────────────────────
# FOOTER
# ─────────────────────────────
st.markdown("---")
st.caption("Built using PyTorch + Transformers + Streamlit")