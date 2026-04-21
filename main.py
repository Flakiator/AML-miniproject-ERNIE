import os
import re
from importlib import metadata

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from numbers import Number

import evaluate
import numpy as np


def _normalize_version(version):
    return version.split("+", 1)[0]


def _get_installed_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _get_required_torch_version_for_torchvision():
    try:
        requirements = metadata.distribution("torchvision").requires or []
    except metadata.PackageNotFoundError:
        return None

    for requirement in requirements:
        if not requirement.lower().startswith("torch "):
            continue

        match = re.search(r"==\s*([0-9]+(?:\.[0-9]+){1,2}(?:\+[A-Za-z0-9_.-]+)?)", requirement)
        if match:
            return match.group(1)

    return None


def check_torchvision_compatibility():
    torch_version = _get_installed_version("torch")
    torchvision_version = _get_installed_version("torchvision")

    if not torch_version or not torchvision_version:
        return

    required_torch_version = _get_required_torch_version_for_torchvision()
    if not required_torch_version:
        return

    if _normalize_version(torch_version) != _normalize_version(required_torch_version):
        raise RuntimeError(
            "Incompatible PyTorch install detected: "
            f"torch=={torch_version}, torchvision=={torchvision_version}. "
            f"This torchvision build expects torch=={required_torch_version}. "
            "This project does not use torchvision. Uninstall torchvision from this venv "
            "or reinstall matching torch/torchvision versions, then run the script again."
        )

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }


check_torchvision_compatibility()

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128
epochs = 3

# Configurations for experiment tracking
record_stats = True  # Set to True to enable Weights & Biases logging
model_name = "distilbert-base-cased"
data_set_name = "stanfordnlp/imdb"
wandb_run_name = f"distilbert-base-cased_imdb-bs{BATCH_SIZE}-lr{LEARNING_RATE}-ep{epochs}"
wandb_project_name = "aml-miniproject-ernie"
checkpoint_path = "distilbert_imdb_checkpoint_dynamic_padding_with_metrics.pth"

# Data Retrieval
data = load_dataset(data_set_name)
train_data, test_data = data["train"], data["test"]
# Create a validation set from the training data
split_idx = int(0.9 * len(train_data))
val_data = train_data.select(range(split_idx, len(train_data)))
train_data = train_data.select(range(split_idx))

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

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
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training Loop
# Select one runtime device and keep model/inputs aligned.
if torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU support
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU support
else:
    device = torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Initialize Weights & Biases for experiment tracking
if record_stats:
    import wandb

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

training_args = TrainingArguments(
    num_train_epochs=epochs,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01, # Regularization
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    report_to=["wandb"] if record_stats else "none",
    run_name=wandb_run_name if record_stats else None,
    fp16=device.type == "cuda",  # Use mixed precision only on CUDA
    use_cpu=device.type == "cpu",
)

# each review is still truncated to MAX_SEQ_LENGTH
# but padding is added only up to the longest example in the current batch
# that usually makes training faster, especially when many texts are shorter than 128
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Run training and evaluation if no checkpoint exists
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
else:
    trainer.train()
    # make checkpoint
    torch.save(model.state_dict(), checkpoint_path)

model.to(device)  # Move model to the device

# Evaluate the resulting model
evaluation = trainer.evaluate()

print("Evaluation results:", evaluation)

if record_stats:
    wandb.log({f"final/{k}": v for k, v in evaluation.items() if isinstance(v, Number)})
    wandb.finish()


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
