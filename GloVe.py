import os

import re
from datasets import load_dataset, Dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import torchtext.vocab as vocab


# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
epochs = 10

# Configurations for experiment tracking
record_stats = False  # Set to True to enable Weights & Biases logging
test_model = True    # Set to True to skip training and only run evaluation and prediction (make sure to have a checkpoint in ./results)
model_name = "google-bert/bert-base-uncased" 
data_set_name = "stanfordnlp/imdb"
wandb_run_name = f"bert-base-uncased-imdb-bs{BATCH_SIZE}-lr{LEARNING_RATE}-ep{epochs}"
wandb_project_name = "aml-miniproject-ernie"

# Data Retrieval
data = load_dataset("stanfordnlp/imdb")
# Filter dataset html tags using regex
data = data.map(lambda x: {"text": re.sub(r"<.*?>", "", x["text"])})  # Remove HTML tags from the text

train_data, test_data = data["train"], data["test"]

# Shuffle the training data to ensure validation set isn't biased of order
train_data = train_data.shuffle(seed=42)

# Create a validation set from the training data
split_idx = int(0.9 * len(train_data))
val_data = train_data.select(range(split_idx, len(train_data)))
train_data = train_data.select(range(split_idx))


glove = vocab.GloVe(name="42B", dim=300)

def get_glove_embedding(text):
    tokens = text.lower().split()  # simple whitespace tokenization
    vectors = [glove[token] for token in tokens if token in glove.stoi]
    if not vectors:
        return torch.zeros(300)
    return torch.stack(vectors).mean(dim=0)  # mean pooling → single 300-dim vector


def embed_text(example):
    embedding = get_glove_embedding(example["text"])
    return {"embeddings": embedding.numpy()}  # .map() expects numpy, not tensors

# Precompute embeddings for all splits
train_data = train_data.map(embed_text)
val_data   = val_data.map(embed_text)
test_data  = test_data.map(embed_text)

# Set format for PyTorch
train_data.set_format(type="torch", columns=["embeddings", "label"])
val_data.set_format(type="torch", columns=["embeddings", "label"])
test_data.set_format(type="torch", columns=["embeddings", "label"])

# Create model classifier
class GloVeClassifier(torch.nn.Module):
    def __init__(self, embedding_dim=300, num_labels=2):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_labels)
        )
    
    def forward(self, embeddings, label=None):
        logits = self.classifier(embeddings)
        loss = None
        if label is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, label)
        return SequenceClassifierOutput(loss=loss, logits=logits)

model = GloVeClassifier()

# Select one runtime device and keep model/inputs aligned.
if torch.cuda.is_available():
    device = torch.device("cuda") # NVIDIA GPU support
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Apple Silicon GPU support
else:
    device = torch.device("cpu")
model.to(device)

# TRAIN THE MODEL
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
    eval_strategy="epoch",                               # Evaluate the model at the end of each epoch
    save_strategy="epoch",                               # Save the model at the end of each epoch
    save_total_limit=2,                                  # Only keep the 2 most recent checkpoints to save disk space
    weight_decay=0.01,                                   # Regularization to prevent overfitting by adding a penalty to the loss function based on the magnitude of the model's weights.
    learning_rate=LEARNING_RATE,
    logging_steps=50,
    logging_strategy="steps",                            # Log training metrics every 50 steps
    report_to=["wandb"] if record_stats else "none",     # Log to Weights & Biases if enabled
    output_dir="./results_glove",                              # Directory to save model checkpoints and logs
    run_name=wandb_run_name if record_stats else None,
    fp16=device.type == "cuda" or device.type == "mps",                          # Use mixed precision on CUDA and MPS (if gpu is available)
    load_best_model_at_end=True,                         # Load the best model at the end of training based on evaluation metrics
    metric_for_best_model="f1",                          # Use F1 score to determine the best model
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
trainer.train()

evaluation = trainer.evaluate(eval_dataset=test_data)
print("Test results:", evaluation)