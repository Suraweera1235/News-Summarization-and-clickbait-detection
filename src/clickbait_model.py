import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def add_features(df):

    df = df.copy()
    raw = df['raw_title']
 
    df['has_number']      = raw.str.contains(r'\d+',        regex=True).astype(int)
    df['has_question']    = raw.str.contains(r'\?',         regex=True).astype(int)
    df['has_exclamation'] = raw.str.contains(r'!',          regex=True).astype(int)
    df['has_caps']        = raw.str.contains(r'[A-Z]{2,}',  regex=True).astype(int)
    df['word_count']      = raw.str.split().str.len()
    df['char_count']      = raw.str.len()
 
    return df


df1 = pd.read_csv("Data/Clickbait/train1.csv")  # headline, clickbait
df2 = pd.read_csv("Data/Clickbait/train2.csv")  # label, title


df1 = df1.rename(columns={
    "headline": "title",
    "clickbait": "label"
})


# Clean second dataset labels
df2['label'] = df2['label'].astype(str).str.strip().str.lower()
df2['label'] = df2['label'].replace({
    'clickbait': 1,
    'news': 0
})


df = pd.concat([df1, df2], ignore_index=True)


df = df.dropna(subset=['title', 'label'])

df['label'] = df['label'].astype(int)
df['raw_title'] = df['title'].astype(str)
df['title'] = df['title'].astype(str).apply(clean_text)

# Remove empty titles
df = df[df['title'].str.strip() != ""]

print("Label Distribution:")
print(df['label'].value_counts())

print("Dataset size:", len(df))


df = add_features(df)


X_train, X_test, y_train, y_test = train_test_split(
    df, df['label'], test_size=0.2, random_state=42
)

print("\nFitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True  
)

# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)


# model = LogisticRegression(
#     class_weight='balanced',
#     max_iter=1000
# )

# model.fit(X_train_vec, y_train)

# y_pred = model.predict(X_test_vec)

# print("\n Model Performance:")
# print(classification_report(y_test, y_pred))


# with open("models/clickbait/clickbait_model.pkl", "wb") as f:
#     pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open("models/clickbait/vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# print("\n Clickbait model trained and saved!")

X_train_tfidf = vectorizer.fit_transform(X_train['title'])
X_test_tfidf  = vectorizer.transform(X_test['title'])
 
# Stack TF-IDF with handcrafted features
extra_features = ['has_number', 'has_question', 'has_exclamation',
                    'has_caps', 'word_count', 'char_count']
 
from scipy.sparse import hstack, csr_matrix
X_train_vec = hstack([X_train_tfidf, csr_matrix(X_train[extra_features].values)])
X_test_vec  = hstack([X_test_tfidf,  csr_matrix(X_test[extra_features].values)])
 
# Train both models and pick the better one
print("Training Logistic Regression...")
lr_model = LogisticRegression(class_weight='balanced', max_iter=3000)
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)
 
print("Training LinearSVC...")
svm_model = LinearSVC(class_weight='balanced', max_iter=1000)
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)
 
# Print both reports
print("\n── Logistic Regression Performance ──")
print(classification_report(y_test, lr_pred))
 
print("\n── LinearSVC Performance ──")
print(classification_report(y_test, svm_pred))
 
# Save the better model (LinearSVC typically wins on text tasks)
print("\nSaving Logistic Regression model and vectorizer...")
with open("models/clickbait/clickbait_model.pkl", "wb") as f:
    pickle.dump(lr_model, f, protocol=pickle.HIGHEST_PROTOCOL)
 
with open("models/clickbait/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
 
print("✅ Clickbait model trained and saved!")