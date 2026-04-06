import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import clean_text

# ---------------------------
# Load datasets
# ---------------------------
df1 = pd.read_csv("Data/Clickbait/train1.csv")  # headline, clickbait
df2 = pd.read_csv("Data/Clickbait/train2.csv")  # label, title

# ---------------------------
# Fix columns
# ---------------------------
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

# ---------------------------
# Combine datasets
# ---------------------------
df = pd.concat([df1, df2], ignore_index=True)

# ---------------------------
# Clean data
# ---------------------------
df = df.dropna(subset=['title', 'label'])

df['label'] = df['label'].astype(int)
df['title'] = df['title'].astype(str).apply(clean_text)

# Remove empty titles
df = df[df['title'].str.strip() != ""]

print("📊 Label Distribution:")
print(df['label'].value_counts())

print("📊 Dataset size:", len(df))

# ---------------------------
# Features
# ---------------------------
X = df['title']
y = df['label']

# ---------------------------
# Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Vectorizer
# ---------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# Model
# ---------------------------
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)

model.fit(X_train_vec, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = model.predict(X_test_vec)

print("\n📈 Model Performance:")
print(classification_report(y_test, y_pred))

# ---------------------------
# Save
# ---------------------------
with open("models/clickbait/clickbait_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("models/clickbait/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n🎉 Clickbait model trained and saved!")