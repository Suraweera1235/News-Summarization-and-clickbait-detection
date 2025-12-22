# from transformers import BartForConditionalGeneration, BartTokenizer
# from torch.utils.data import Dataset, DataLoader
# import torch
# import os
# from preprocessing import load_summarization_data

# # -----------------------------
# # Load dataset
# # -----------------------------
# train_df, val_df, test_df = load_summarization_data(
#     "Data/Summ/train.csv",
#     "Data/Summ/validation.csv",
#     "Data/Summ/test.csv"
# )

# # -----------------------------
# # Fix NaN values (VERY IMPORTANT)
# # -----------------------------
# train_df = train_df.dropna(subset=['clean_article', 'clean_summary'])
# val_df = val_df.dropna(subset=['clean_article', 'clean_summary'])

# train_df['clean_article'] = train_df['clean_article'].astype(str)
# train_df['clean_summary'] = train_df['clean_summary'].astype(str)
# val_df['clean_article'] = val_df['clean_article'].astype(str)
# val_df['clean_summary'] = val_df['clean_summary'].astype(str)

# print("Train samples:", len(train_df))
# print("Validation samples:", len(val_df))

# # -----------------------------
# # Tokenizer (lighter settings)
# # -----------------------------
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# # -----------------------------
# # Dataset class
# # -----------------------------
# class SummarizationDataset(Dataset):
#     def __init__(self, articles, summaries, tokenizer, max_input=256, max_output=64):
#         self.articles = articles
#         self.summaries = summaries
#         self.tokenizer = tokenizer
#         self.max_input = max_input
#         self.max_output = max_output

#     def __len__(self):
#         return len(self.articles)

#     def __getitem__(self, idx):
#         input_text = self.articles[idx]
#         target_text = self.summaries[idx]

#         input_enc = self.tokenizer(
#             input_text,
#             max_length=self.max_input,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )

#         target_enc = self.tokenizer(
#             target_text,
#             max_length=self.max_output,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )

#         labels = target_enc["input_ids"].squeeze()
#         labels[labels == tokenizer.pad_token_id] = -100

#         return {
#             "input_ids": input_enc["input_ids"].squeeze(),
#             "attention_mask": input_enc["attention_mask"].squeeze(),
#             "labels": labels
#         }

# # -----------------------------
# # Datasets & Loaders
# # -----------------------------
# train_dataset = SummarizationDataset(
#     train_df["clean_article"].tolist(),
#     train_df["clean_summary"].tolist(),
#     tokenizer
# )

# val_dataset = SummarizationDataset(
#     val_df["clean_article"].tolist(),
#     val_df["clean_summary"].tolist(),
#     tokenizer
# )

# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1)

# # -----------------------------
# # Model
# # -----------------------------
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # -----------------------------
# # Optimizer
# # -----------------------------
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# # -----------------------------
# # Training loop (1 epoch)
# # -----------------------------
# model.train()
# for step, batch in enumerate(train_loader):
#     optimizer.zero_grad()

#     input_ids = batch["input_ids"].to(device)
#     attention_mask = batch["attention_mask"].to(device)
#     labels = batch["labels"].to(device)

#     outputs = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         labels=labels
#     )

#     loss = outputs.loss
#     loss.backward()
#     optimizer.step()

#     if step % 50 == 0:
#         print(f"Step {step} | Loss: {loss.item():.4f}")

# # -----------------------------
# # Save model
# # -----------------------------
# os.makedirs("models/summarization", exist_ok=True)
# model.save_pretrained("models/summarization/bart_model")
# tokenizer.save_pretrained("models/summarization/bart_tokenizer")

# print("Summarization model saved successfully")

