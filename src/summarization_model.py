import pandas as pd
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# Text Cleaning
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'[^a-zA-Z. ]', '', text)      # keep letters + dots
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------
# Train Vectorizer
# ---------------------------
def train_summarizer():
    df = pd.read_csv("Data/Summ/train.csv")
    df = df.dropna()

    # Clean articles
    df['article'] = df['article'].apply(clean_text)

    # Train TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(df['article'])

    # Save
    pickle.dump(vectorizer, open("models/summarization/vectorizer.pkl", "wb"))

    print("✅ Summarization vectorizer trained and saved!")


# ---------------------------
# Summarization Function
# ---------------------------
def summarize_text(text, vectorizer, num_sentences=2):

    # Split into sentences
    sentences = text.split('.')

    # Clean sentences
    cleaned_sentences = [clean_text(s) for s in sentences if s.strip() != ""]

    if len(cleaned_sentences) == 0:
        return "No content to summarize."

    # Convert to vectors
    X = vectorizer.transform(cleaned_sentences)

    # Score sentences
    scores = X.sum(axis=1).A1   # convert matrix to array

    # Rank sentences
    ranked = np.argsort(scores)[::-1]

    # Select top sentences
    selected_sentences = [sentences[i] for i in ranked[:num_sentences]]

    return '. '.join(selected_sentences).strip()


# ---------------------------
# Run training manually
# ---------------------------
if __name__ == "__main__":
    train_summarizer()
