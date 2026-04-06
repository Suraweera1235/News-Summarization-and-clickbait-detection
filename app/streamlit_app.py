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
    return ("Clickbait", confidence) if pred == 1 else ("Not Clickbait", confidence)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="News AI System", layout="wide")

# ---------------------------
# Dark Mode Toggle (UI only)
# ---------------------------
dark_mode = st.toggle("Dark Mode")

# Simple color adaptation (no breaking layout)
if dark_mode:
    bg = "#0f172a"
    card_bg = "#1e293b"
    text = "#e2e8f0"
else:
    bg = "#f5f7fa"
    card_bg = "#ffffff"
    text = "#1f2937"

# ---------------------------
# CSS (SAFE + RESPONSIVE)
# ---------------------------
st.markdown(f"""
<style>
.block-container {{
    padding-top: 2rem;
}}

body {{
    background-color: {bg};
    color: {text};
}}

.card {{
    background-color: {card_bg};
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
}}

.section-title {{
    font-weight: 600;
    margin-bottom: 8px;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
}}

.summary-box {{
    padding: 10px;
    border-radius: 6px;
    border-left: 3px solid #6366f1;
    line-height: 1.5;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.title("News Summarization & Clickbait Detection")

# ---------------------------
# Input Section
# ---------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    user_input = st.text_area("Enter News Article or Headline", height=150)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Analyze Button
# ---------------------------
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Processing..."):

            clean_input = clean_text(user_input)
            label, confidence = predict_clickbait(clean_input)

            # Dynamic color
            if label == "Clickbait":
                color = "red"
            else:
                color = "green"

            # ---------------------------
            # Responsive Columns
            # ---------------------------
            col1, col2 = st.columns([1, 1])

            # ---------------------------
            # LEFT: Clickbait Detection
            # ---------------------------
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Clickbait Detection</div>', unsafe_allow_html=True)

                st.markdown(f"### :{color}[{label}]")

                # Clean progress bar (colored)
                st.progress(confidence)

                st.write(f"Confidence: {confidence*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            # ---------------------------
            # RIGHT: Summary
            # ---------------------------
            with col2:
                summary = summarize_text(user_input, sum_vectorizer)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)