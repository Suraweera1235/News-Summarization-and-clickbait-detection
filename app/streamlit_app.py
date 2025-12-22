# import streamlit as st
# import joblib
# from transformers import BartTokenizer, BartForConditionalGeneration

# # Load clickbait model
# click_model = joblib.load("models/clickbait/clickbait_model.pkl")
# vectorizer = joblib.load("models/clickbait/vectorizer.pkl")

# # Load summarization model
# tokenizer = BartTokenizer.from_pretrained("models/summarization/bart_tokenizer")
# summ_model = BartForConditionalGeneration.from_pretrained("models/summarization/bart_model")

# st.title("News Summarization & Clickbait Detection")

# text = st.text_area("Enter News Headline or Article")

# if st.button("Analyze"):
#     # Clickbait
#     tfidf_text = vectorizer.transform([text])
#     click_pred = click_model.predict(tfidf_text)[0]
#     st.write("Clickbait Prediction:", "Yes" if click_pred==1 else "No")
    
#     # Summarization
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
#     summary_ids = summ_model.generate(inputs['input_ids'], max_length=120, min_length=30, num_beams=4)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     st.write("Summary:", summary)
