import os

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import wandb
from numbers import Number
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 256
epochs = 3

# Configurations for experiment tracking
record_stats = True  # Set to True to enable Weights & Biases logging
model_name = "distilbert/distilbert-base-uncased" 
data_set_name = "stanfordnlp/imdb"
wandb_run_name = f"bert-lite-imdb-bs{BATCH_SIZE}-lr{LEARNING_RATE}-ep{epochs}"
wandb_project_name = "aml-miniproject-ernie"

# Data Retrieval
data = load_dataset("stanfordnlp/imdb")
train_data, test_data = data["train"], data["test"]



# Create a validation set from the training data
split_idx = int(0.9 * len(train_data))
val_data = train_data.select(range(split_idx, len(train_data)))
train_data = train_data.select(range(split_idx))

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

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

# Model Initialization
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training Loop
# Select one runtime device and keep model/inputs aligned.
if torch.cuda.is_available():
    device = torch.device("cuda") # NVIDIA GPU support
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Apple Silicon GPU support
else:
    device = torch.device("cpu")
model.to(device)

# Initialize Weights & Biases for experiment tracking
if record_stats:
    wandb.init(
        project=wandb_project_name,
        entity="ERNIE-AML-2026",
        name=wandb_run_name,
        config={
            "model_name": model_name,
            "dataset": data_set_name,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "epochs": epochs,
        },
        reinit=True,
    )

# Define a compute_metrics function for the Trainer to use during evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
    }

# Set up training arguments for the Hugging Face Trainer
training_args = TrainingArguments(
    num_train_epochs=epochs,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",                          # Evaluate the model at the end of each epoch
    save_strategy="epoch",                          # Save the model at the end of each epoch
    weight_decay=0.01,                              # Regularization to prevent overfitting by adding a penalty to the loss function based on the magnitude of the model's weights.
    learning_rate=LEARNING_RATE,
    logging_steps=50,
    logging_strategy="steps",                       # Log training metrics every 50 steps
    report_to="wandb" if record_stats else "none",  # Log to Weights & Biases if enabled
    run_name=wandb_run_name if record_stats else None,
    fp16=device.type == "cuda",                     # Use mixed precision only on CUDA
    use_cpu=device.type == "cpu",
    load_best_model_at_end=True,                    # Load the best model at the end of training based on evaluation metrics
    metric_for_best_model="f1",                     # Use F1 score to determine the best model
)

# Sets optimizer to AdamW by default
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

# Run training and evaluation if no checkpoint exists
if os.path.exists("bert_imdb_checkpoint.pth"):
    state_dict = torch.load("bert_imdb_checkpoint.pth", map_location="cpu")
    model.load_state_dict(state_dict)
else:
    trainer.train()
    # make checkpoint
    torch.save(model.state_dict(), "bert_imdb_checkpoint.pth")
    
model.to(device) # Move model to the device

# Evaluate the resulting model
evaluation = trainer.evaluate()

print("Evaluation results:", evaluation)

# Test the model on input text
def predict_review(text):
    model.eval()
    model_device = next(model.parameters()).device

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    enc = {k: v.to(model_device) for k, v in enc.items()}

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