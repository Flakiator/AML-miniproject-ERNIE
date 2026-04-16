from datasets import load_dataset

from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          DataCollatorWithPadding,)
from torch.optim import AdamW # Standard way to import it now
from torch.utils.data import DataLoader
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import load

def setup_wandb():
    wandb.login()
    return wandb.init(
        entity="ERNIE-AML-2026",
        project="aml-miniproject-ernie",
        config={
            "learning_rate": HYPERPARAMETERS["learning_rate"],
            "dataset": "stanfordnlp/imdb",
            "epochs": HYPERPARAMETERS["num_train_epochs"],
        },
    )

HYPERPARAMETERS = {
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
}
model_name = "distilbert-base-cased"
wandb_run = setup_wandb()
# Data Retrieval
data = load_dataset("stanfordnlp/imdb")
train_data, test_data = data["train"], data["test"]
# Create a validation set from the training data
# split_idx = int(0.9 * len(train_data))
# val_data = train_data.select(range(split_idx, len(train_data)))
# train_data = train_data.select(range(split_idx))
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Tokenize a sample text
sample_text = "I absolutely loved this movie! Highly recommend it."
tokens = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Apply the tokenizer to the dataset
tokenized_datasets = data.map(tokenize_function, batched=True)

# Initialize the model

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# IMDB already provides numeric labels (0=negative, 1=positive).
if "label" not in data["train"].column_names:
    raise ValueError("Expected a 'label' column in the dataset.")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",  # explicit integration
    run_name=f"{model_name}-imdb-lr{HYPERPARAMETERS['learning_rate']}",
    learning_rate=HYPERPARAMETERS["learning_rate"],
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=HYPERPARAMETERS["num_train_epochs"],
    weight_decay=HYPERPARAMETERS["weight_decay"],
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
)

# Defining a Custom Metric
# Load a metric (F1-score in this case)
metric = load("f1")

# Define a custom compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,                        # Pre-trained BERT model
    args=training_args,                 # Training arguments
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,        # Efficient batching
    compute_metrics=compute_metrics     # Custom metric
)

# Start training
trainer.train()

results = trainer.evaluate()
print(results)
wandb.finish()
