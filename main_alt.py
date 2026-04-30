import os

from safetensors.torch import load_file
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 256
epochs = 3

# Configurations for experiment tracking
record_stats = False  # Set to True to enable Weights & Biases logging
test_model = True    # Set to True to skip training and only run evaluation and prediction (make sure to have a checkpoint in ./results)
model_name = "google-bert/bert-base-uncased" 
data_set_name = "stanfordnlp/imdb"
wandb_run_name = f"bert-base-uncased-imdb-bs{BATCH_SIZE}-lr{LEARNING_RATE}-ep{epochs}"
wandb_project_name = "aml-miniproject-ernie"

# Data Retrieval
data = load_dataset("stanfordnlp/imdb")
train_data, test_data = data["train"], data["test"]

# Shuffle the training data to ensure validation set isn't biased of order
train_data = train_data.shuffle(seed=42)

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

# Model Definition
class BertForSentiment(torch.nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        # ← OUTPUT HEAD: change num_labels to adapt to a different task
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # ← INPUT HEAD: pass token_type_ids here for two-sequence tasks
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # ← Take CLS token for sequence classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_output))

        # Compute loss if labels are provided (needed for Trainer compatibility)
        loss = None
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

        # Return in the format the Trainer expects
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)

# Try to load old model
last_checkpoint = get_last_checkpoint("./results")  # from transformers.trainer_utils

# Initialize the model
if test_model:
    print(f"Loading model from checkpoint: {last_checkpoint}")
    state_dict = load_file(os.path.join(last_checkpoint, "model.safetensors"))
    model = BertForSentiment(model_name)
    model.load_state_dict(state_dict)
else:
    print("No checkpoint found, initializing new model.")
    model = BertForSentiment(model_name)


# Select one runtime device and keep model/inputs aligned.
if torch.cuda.is_available():
    device = torch.device("cuda") # NVIDIA GPU support
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Apple Silicon GPU support
else:
    device = torch.device("cpu")
model.to(device)

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
    output_dir="./results",                              # Directory to save model checkpoints and logs
    run_name=wandb_run_name if record_stats else None,
    fp16=device.type == "cuda",                          # Use mixed precision only on CUDA (if gpu is available)
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
if not test_model:
    trainer.train()
    
# Evaluate the resulting model
evaluation = trainer.evaluate(eval_dataset=test_data)

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