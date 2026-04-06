import streamlit as st
import pickle
from src.preprocessing import clean_text
from src.summarization_model import summarize_text
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---------------------------
# Load Models
# ---------------------------
click_model = pickle.load(open("models/clickbait/clickbait_model.pkl", "rb"))
click_vectorizer = pickle.load(open("models/clickbait/vectorizer.pkl", "rb"))

sum_vectorizer = pickle.load(open("models/summarization/vectorizer.pkl", "rb"))


# ---------------------------
# Clickbait Prediction
# ---------------------------
def predict_clickbait(text):
    vec = click_vectorizer.transform([text])
    pred = click_model.predict(vec)[0]
    prob = click_model.predict_proba(vec)[0]

    confidence = max(prob)

    if pred == 1:
        return "🚨 Clickbait", confidence
    else:
        return "📰 Not Clickbait", confidence


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="News AI System", layout="centered")

st.title("🧠 News Summarization & Clickbait Detection")

user_input = st.text_area("Enter News Article or Headline")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("Processing..."):

            # Clean input
            clean_input = clean_text(user_input)

            # ---------------------------
            # Clickbait Detection
            # ---------------------------
            label, confidence = predict_clickbait(clean_input)

            st.subheader("📌 Clickbait Detection")
            st.write(label)
            st.progress(float(confidence))

            # ---------------------------
            # Summarization
            # ---------------------------
            st.subheader("🧠 Summary")

            summary = summarize_text(user_input, sum_vectorizer)

            st.write(summary)