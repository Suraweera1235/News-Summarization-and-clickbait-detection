import re
import pandas as pd

def clean_text(text):
    """
    Cleans text by lowercasing, removing URLs, special chars, and extra spaces.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_clickbait_data(paths):
    """
    Loads and combines clickbait CSV files, cleans headlines.
    """
    df_list = [pd.read_csv(p) for p in paths]
    df = pd.concat(df_list, ignore_index=True)
    df['clean_headline'] = df['headline'].apply(clean_text)
    return df

def load_summarization_data(train_path, val_path, test_path):
    """
    Loads summarization CSV files and cleans articles and summaries.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    for df in [train_df, val_df, test_df]:
        df['clean_article'] = df['article'].apply(clean_text)
        df['clean_summary'] = df['highlights'].apply(clean_text)
        
    return train_df, val_df, test_df
