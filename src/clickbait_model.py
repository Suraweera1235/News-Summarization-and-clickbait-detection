from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from preprocessing import load_clickbait_data
import pandas as pd
import numpy as np
import os


clickbait_paths = ["Data/Clickbait/train1.csv", "Data/Clickbait/train2.csv"]


clickbait_df = load_clickbait_data(clickbait_paths)


print("Columns in DataFrame:", clickbait_df.columns)
print("Any NaNs in headlines?", clickbait_df['clean_headline'].isnull().sum())
print("Any NaNs in labels?", clickbait_df['label'].isnull().sum())


# Fill missing headlines with empty strings
clickbait_df['clean_headline'] = clickbait_df['clean_headline'].fillna('')

# Drop rows with missing labels
clickbait_df = clickbait_df.dropna(subset=['label'])

# Map string labels to integers
label_mapping = {'news': 0, 'clickbait': 1}
clickbait_df['label'] = clickbait_df['label'].map(label_mapping)

# Drop rows that didn’t match the mapping
clickbait_df = clickbait_df.dropna(subset=['label'])
clickbait_df['label'] = clickbait_df['label'].astype(int)


X = clickbait_df['clean_headline']
y = clickbait_df['label']


vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)


model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)


os.makedirs('models/clickbait', exist_ok=True)
joblib.dump(model, 'models/clickbait/clickbait_model.pkl')
joblib.dump(vectorizer, 'models/clickbait/vectorizer.pkl')


y_pred = model.predict(X_tfidf)
print("Clickbait Detection Report (Train Data):")
print(classification_report(y, y_pred))
