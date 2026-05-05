import re
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import wandb

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
epochs = 10

# Configurations for experiment tracking
record_stats = True  # Set to True to enable Weights & Biases logging
test_model = True  # Set to True to skip training and only run evaluation and prediction (make sure to have a checkpoint in ./results)
data_set_name = "stanfordnlp/imdb"
wandb_run_name = f"glove-imdb-bs{BATCH_SIZE}-lr{LEARNING_RATE}-ep{epochs}"
wandb_project_name = "aml-miniproject-ernie"

wandb.init(project=wandb_project_name)
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

# --- GloVe: load from local .txt (no torchtext, no checks) ---
# IMPORTANT: This path must point to an extracted embedding .txt file (not the .zip).
GLOVE_TXT_PATH = Path("glove.6B.300d.txt")
GLOVE_DIM = 300


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def build_vocab_from_splits(*splits) -> set[str]:
    vocab_set: set[str] = set()
    for split in splits:
        for ex in split:
            vocab_set.update(tokenize(ex["text"]))
    return vocab_set


def load_glove_selected(glove_txt_path: Path, vocab_set: set[str], dim: int) -> dict[str, np.ndarray]:
    # If glove_txt_path doesn't exist, this will crash (as requested).
    vectors: dict[str, np.ndarray] = {}
    with open(glove_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if not parts:
                continue
            word = parts[0]
            if word not in vocab_set:
                continue
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.shape[0] != dim:
                continue
            vectors[word] = vec
    return vectors


print("Building dataset vocabulary...")
vocab_set = build_vocab_from_splits(train_data, val_data, test_data)
print(f"Vocab size: {len(vocab_set):,}")

print("Loading vectors for dataset vocab...")
glove = load_glove_selected(GLOVE_TXT_PATH, vocab_set=vocab_set, dim=GLOVE_DIM)
print(f"Loaded vectors: {len(glove):,}")

misses = 0


def get_glove_embedding(text: str):
    global misses
    tokens = tokenize(text)
    vectors = []
    for token in tokens:
        vec = glove.get(token)
        if vec is None:
            misses += 1
        else:
            vectors.append(vec)
    if not vectors:
        return np.zeros(GLOVE_DIM, dtype=np.float32)
    return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)


def embed_text(example):
    embedding = get_glove_embedding(example["text"])
    return {"embeddings": embedding}


# Precompute embeddings for all splits
train_data = train_data.map(embed_text)
val_data = val_data.map(embed_text)
test_data = test_data.map(embed_text)
print(f"Total embedding misses: {misses}")

# Set format for PyTorch
train_data = train_data.rename_column("label", "labels")
val_data   = val_data.rename_column("label", "labels")
test_data  = test_data.rename_column("label", "labels")

train_data.set_format(type="torch", columns=["embeddings", "labels"])
val_data.set_format(type="torch", columns=["embeddings", "labels"])
test_data.set_format(type="torch", columns=["embeddings", "labels"])


# Create model classifier
class GloVeClassifier(torch.nn.Module):
    def __init__(self, embedding_dim=300, num_labels=2):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_labels),
        )

    # HF Trainer usually passes `labels`, but some configs pass `label`
    def forward(self, embeddings, labels=None):
        logits = self.classifier(embeddings)
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


model = GloVeClassifier()

# Select one runtime device and keep model/inputs aligned.
if torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU support
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU support
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
    eval_strategy="epoch",  # Evaluate the model at the end of each epoch
    save_strategy="epoch",  # Save the model at the end of each epoch
    save_total_limit=2,  # Only keep the 2 most recent checkpoints to save disk space
    weight_decay=0.01,
    # Regularization to prevent overfitting by adding a penalty to the loss function based on the magnitude of the model's weights.
    learning_rate=LEARNING_RATE,
    logging_steps=50,
    logging_strategy="steps",  # Log training metrics every 50 steps
    report_to=["wandb"] if record_stats else "none",  # Log to Weights & Biases if enabled
    output_dir="./results_glove",  # Directory to save model checkpoints and logs
    run_name=wandb_run_name if record_stats else None,
    fp16=device.type == "cuda" or device.type == "mps",  # Use mixed precision on CUDA/MPS
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

# Avoid notebook-only callback state errors during standalone evaluate() calls
from transformers.utils.notebook import NotebookProgressCallback

trainer.remove_callback(NotebookProgressCallback)

# Run training and evaluation
trainer.train()

evaluation = trainer.evaluate(eval_dataset=test_data)
print("Test results:", evaluation)

if record_stats:
    wandb.log({
        "test/accuracy":  evaluation["eval_accuracy"],
        "test/f1":        evaluation["eval_f1"],
        "test/precision": evaluation["eval_precision"],
        "test/recall":    evaluation["eval_recall"],
        "test/mcc":       evaluation["eval_mcc"],
        "test/loss":      evaluation["eval_loss"],
    })
wandb.finish()
