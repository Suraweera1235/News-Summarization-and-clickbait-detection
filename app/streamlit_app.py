import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="News Analyzer", layout="centered")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load clickbait model (cached)
# -----------------------------
@st.cache_resource
def load_clickbait_model():
    click_model = joblib.load("models/clickbait/clickbait_model.pkl")
    vectorizer = joblib.load("models/clickbait/vectorizer.pkl")
    return click_model, vectorizer

# -----------------------------
# Load summarization model (cached)
# -----------------------------
@st.cache_resource
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


click_model, vectorizer = load_clickbait_model()
tokenizer, summ_model = load_summarization_model()

# -----------------------------
# UI
# -----------------------------
st.title("📰 News Summarization & Clickbait Detection")

text = st.text_area(
    "Enter News Headline or Article",
    height=250,
    placeholder="Paste a news article or headline here..."
)

# -----------------------------
# Analyze
# -----------------------------
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # -------- Clickbait Detection --------
        tfidf_text = vectorizer.transform([text])
        click_pred = click_model.predict(tfidf_text)[0]

        st.subheader("Clickbait Detection")
        st.write("Prediction:", "🚨 Clickbait" if click_pred == 1 else "✅ Not Clickbait")

        # -------- Summarization --------
        st.subheader("Summary")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            summary_ids = summ_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=120,
                min_length=30,
                num_beams=2,
                early_stopping=True
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.success(summary)
