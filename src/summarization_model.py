import pandas as pd
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z. ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_summarization_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for df in [train_df, val_df, test_df]:
        df['clean_article'] = df['article'].astype(str).apply(clean_text)
        df['clean_summary'] = df['highlights'].astype(str).apply(clean_text)

    return train_df, val_df, test_df

def train_summarizer():
    df = pd.read_csv("Data/Summ/train.csv")

    
    df = df[['article', 'highlights']].dropna()

   
    df['article'] = df['article'].astype(str).apply(clean_text)

    
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(df['article'])

    with open("models/summarization/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Summarization vectorizer trained and saved!")


def summarize_text(text, vectorizer, num_sentences=2):

    sentences = text.split('.')

    cleaned_sentences = [
        clean_text(s) for s in sentences if s.strip() != ""
    ]

    if len(cleaned_sentences) == 0:
        return "No content to summarize."

    X = vectorizer.transform(cleaned_sentences)

    scores = X.sum(axis=1).A1

    ranked = np.argsort(scores)[::-1]

    selected = [sentences[i].strip() for i in ranked[:num_sentences]]

    return '. '.join(selected)

if __name__ == "__main__":
    train_summarizer()