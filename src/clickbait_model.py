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
df1 = pd.read_csv("Data/Clickbait/train1.csv")
df2 = pd.read_csv("Data/Clickbait/train2.csv")

# Combine
df = pd.concat([df1, df2], ignore_index=True)

# ---------------------------
# CLEAN DATA (IMPORTANT)
# ---------------------------
df = df.dropna(subset=['label', 'title'])

# Clean labels properly
df['label'] = df['label'].astype(str)
df['label'] = df['label'].str.strip().str.lower()

print("🔍 Cleaned labels:")
print(df['label'].value_counts())

# Convert labels safely
df['label'] = df['label'].replace({
    'clickbait': 1,
    'news': 0
})

# Remove anything unexpected
df = df[df['label'].isin([0, 1])]

print("\n✅ After mapping:")
print(df['label'].value_counts())

print("\n📊 Final dataset size:", len(df))

# ---------------------------
# Clean text
# ---------------------------
df['title'] = df['title'].apply(clean_text)

# ---------------------------
# Features
# ---------------------------
X = df['title']
y = df['label']

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Vectorization
# ---------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# Model (balanced!)
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
# Save model
# ---------------------------
pickle.dump(model, open("models/clickbait/clickbait_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/clickbait/vectorizer.pkl", "wb"))

print("\n🎉 Clickbait model trained and saved successfully!")