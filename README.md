# News-Summarization-and-clickbait-detection
### cd D:\DL_MiniProject\News-Summarization-and-clickbait-detection
### python -m streamlit run app/streamlit_app.py

# 📰 News Summarization & Clickbait Detection

A deep learning and NLP project that combines **extractive news summarization** and **clickbait headline detection** into a single interactive Streamlit web application.

---

## 🧠 Overview

This project tackles two NLP tasks:

1. **News Summarization** — Given a news article, generate a concise summary using TF-IDF extractive summarization, evaluated against a pre-trained `facebook/bart-large-cnn` baseline using ROUGE scores.
2. **Clickbait Detection** — Classify whether a news headline is clickbait or not using a Logistic Regression model trained on TF-IDF features.

Both tasks are accessible through an interactive Streamlit web app.

---

## 🗂️ Project Structure

```
News-Summarization-and-clickbait-detection/
│
├── app/
│   └── streamlit_app.py              # Streamlit web application
│
├── Data/
│   ├── Clickbait/
│   │   ├── train1.csv                # headline, clickbait columns
│   │   └── train2.csv                # title, label columns
│   └── Summ/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── models/
│   ├── clickbait/
│   │   ├── clickbait_model.pkl       # Trained logistic regression model
│   │   └── vectorizer.pkl            # Fitted TF-IDF vectorizer
│   └── summarization/
│       └── vectorizer.pkl            # Fitted TF-IDF vectorizer
│
├── notebooks/                        # Experimentation notebooks
│
├── src/
│   ├── clickbait_model.py            # Clickbait model training
│   ├── summarization.py              # Summarization logic
│   ├── evaluation.py                 # ROUGE evaluation using BART baseline
│   └── preprocessing.py             # Text cleaning utilities
│
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

| Task | Dataset | Source |
|------|---------|--------|
| Summarization | CNN/DailyMail Newspaper Text Summarization | [Kaggle](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail) |
| Clickbait Detection | News Clickbait Dataset | [Kaggle](https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset) |

---

## 🏗️ Model Architecture

### Summarization
- **Approach:** Extractive summarization using TF-IDF sentence scoring
- Sentences are ranked by their cumulative TF-IDF weight; the top-N sentences are returned as the summary
- **Evaluation:** Compared against `facebook/bart-large-cnn` outputs using ROUGE-1, ROUGE-2, ROUGE-L

### Clickbait Detection
- **Approach:** Binary classification (clickbait = 1, non-clickbait = 0)
- **Features:** TF-IDF vectors (top 5,000 features, English stop words removed)
- **Model:** Logistic Regression with balanced class weights
- **Evaluation:** Precision, Recall, F1-score, Accuracy

---

## 📈 Results

### Summarization (ROUGE Scores)

> Evaluated on 200 test samples from CNN/DailyMail against `facebook/bart-large-cnn`.

| Metric | Score |
|--------|-------|
| ROUGE-1 | — |
| ROUGE-2 | — |
| ROUGE-L | — |

*Run `src/evaluation.py` to generate scores.*

### Clickbait Detection

| Metric | Score |
|--------|-------|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |

*Scores are printed after running `src/clickbait_model.py`.*

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Suraweera1235/News-Summarization-and-clickbait-detection.git
cd News-Summarization-and-clickbait-detection

# 2. Install dependencies
pip install -r requirements.txt
```

### Train the Clickbait Model

```bash
python src/clickbait_model.py
```

This trains the logistic regression model and saves `clickbait_model.pkl` and `vectorizer.pkl` to `models/clickbait/`.

### Train the Summarization Vectorizer

```bash
python src/summarization.py
```

This fits the TF-IDF vectorizer on the training data and saves it to `models/summarization/`.

### Run ROUGE Evaluation

```bash
python src/evaluation.py
```

### Launch the App

```bash
python -m streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| scikit-learn | TF-IDF vectorization, Logistic Regression |
| Hugging Face Transformers | BART model for evaluation baseline |
| Hugging Face `evaluate` | ROUGE metric computation |
| PyTorch | BART inference |
| Streamlit | Web application |
| Pandas / NumPy | Data processing |

---

## ⚠️ Notes

- The summarization model is **extractive** (selects existing sentences). For abstractive summarization (generates new text), consider fine-tuning BART or T5.
- The `evaluation.py` script uses `facebook/bart-large-cnn` as a reference — it does **not** fine-tune BART, it uses it out-of-the-box to benchmark ROUGE scores.
- First run of `evaluation.py` will download the BART model (~1.6 GB). A GPU is recommended but not required.

---

## 📚 References

- [CNN/DailyMail Dataset on Kaggle](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
- [News Clickbait Dataset on Kaggle](https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset)
- [facebook/bart-large-cnn on Hugging Face](https://huggingface.co/facebook/bart-large-cnn)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)

---

## 👤 Author

**Suraweera1235** — [GitHub](https://github.com/Suraweera1235)