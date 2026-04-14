from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import torch
import wandb

def setup_wandb():
    wandb.login()
    return wandb.init(
        entity="ERNIE-AML-2026",
        project="aml-miniproject-ernie",
        # ...
    )

wandb_run = setup_wandb()
# Data Retrieval
data = load_dataset("stanfordnlp/imdb")
train_data, test_data = data["train"], data["test"]

# Create a validation set from the training data
split_idx = int(0.9 * len(train_data))
val_data = train_data.select(range(split_idx, len(train_data)))
train_data = train_data.select(range(split_idx))

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)
# Tokenize the datasets
train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)