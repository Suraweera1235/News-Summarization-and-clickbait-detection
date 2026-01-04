from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from preprocessing import load_summarization_data


_, _, test_df = load_summarization_data(
    "Data/Summ/train.csv",
    "Data/Summ/validation.csv",
    "Data/Summ/test.csv"
)


test_df = test_df.dropna(subset=["clean_article"])
test_df["clean_article"] = test_df["clean_article"].astype(str)

print("Test samples:", len(test_df))


model_name = "facebook/bart-large-cnn"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def summarize_article(text, max_input=256, max_output=64):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_input
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


for i in range(3):  # summarize first 3 articles
    print(f"\nArticle {i+1}")
    print(test_df.iloc[i]["clean_article"][:500], "...")

    summary = summarize_article(test_df.iloc[i]["clean_article"])
    print("\nSummary:")
    print(summary)
