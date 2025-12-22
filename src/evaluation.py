# import torch
# import evaluate
# from transformers import BartTokenizer, BartForConditionalGeneration
# from preprocessing import load_summarization_data


# rouge = evaluate.load("rouge")

# _, _, test_df = load_summarization_data(
#     "Data/Summ/train.csv",
#     "Data/Summ/validation.csv",
#     "Data/Summ/test.csv"
# )

# # Remove NaNs
# test_df = test_df.dropna(subset=["clean_article", "clean_summary"])
# test_df["clean_article"] = test_df["clean_article"].astype(str)
# test_df["clean_summary"] = test_df["clean_summary"].astype(str)

# print("Test samples:", len(test_df))


# tokenizer = BartTokenizer.from_pretrained("models/summarization/bart_tokenizer")
# model = BartForConditionalGeneration.from_pretrained("models/summarization/bart_model")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# predictions = []
# references = []

# for _, row in test_df.iterrows():
#     inputs = tokenizer(
#         row["clean_article"],
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=256
#     ).to(device)

#     with torch.no_grad():
#         summary_ids = model.generate(
#             inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_length=64,
#             num_beams=4,
#             early_stopping=True
#         )

#     pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     predictions.append(pred_summary)
#     references.append(row["clean_summary"])


# results = rouge.compute(predictions=predictions, references=references)

# print("\nROUGE Evaluation Results:")
# for k, v in results.items():
#     print(f"{k}: {v:.4f}")

