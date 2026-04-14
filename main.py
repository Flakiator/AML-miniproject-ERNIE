import os

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 256
# Data Retrieval
data = load_dataset("stanfordnlp/imdb")
train_data, test_data = data["train"], data["test"]
# Create a validation set from the training data
split_idx = int(0.9 * len(train_data))
val_data = train_data.select(range(split_idx, len(train_data)))
train_data = train_data.select(range(split_idx))

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("boltuix/bert-lite")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)

# Tokenize the datasets (this gives us input_ids and attention_mask)
# input_ids: token IDs for the input text
# attention_mask: indicates which tokens are padding (0) and which are not (1)
train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)
val_data = val_data.map(tokenize, batched=True)

# Set the format for PyTorch
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Model Initialization
model = BertForSequenceClassification.from_pretrained("boltuix/bert-lite", num_labels=2)

# Training Loop
# Check for GPU and use it if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

epochs = 2
def train():
    for epoch in range(epochs):
        model.train()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True
        )
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch} done")

# Evaluation on the validation set
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],)
            
            preds = torch.argmax(outputs.logits, dim=1)
            
            correct += (preds == batch["label"]).sum().item()
            total += len(batch["label"])

    print("Validation accuracy:", correct / total)

# Run training and evaluation if no checkpoint exists
if os.path.exists("bert_imdb_checkpoint.pth"):
    model.load_state_dict(torch.load("bert_imdb_checkpoint.pth", map_location=device))
else:
    train()
    # make checkpoint
    torch.save(model.state_dict(), "bert_imdb_checkpoint.pth")
# Evaluate the resulting model
evaluate()

# Test the model on input text

def predict_review(text):
    model.eval()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1).squeeze(0)
        pred_id = torch.argmax(probs).item()

    return {
        "label": pred_id,
        "confidence": float(probs[pred_id].item()),
        "prob_negative": float(probs[0].item()),
        "prob_positive": float(probs[1].item()),
    }

while True:
    txt = input("Enter a movie review (or 'quit'): ").strip()
    if txt.lower() == "quit":
        break
    result = predict_review(txt)
    print(result)