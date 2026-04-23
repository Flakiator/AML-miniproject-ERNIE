import os
from dataclasses import dataclass
from numbers import Number

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from sentiment_app.environment import check_torchvision_compatibility, get_device
from sentiment_app.model import create_model


@dataclass(frozen=True)
class Settings:
    batch_size: int = 256 #16
    learning_rate: float = 1e-5
    max_seq_length: int = 512 #64
    epochs: int = 3
    record_stats: bool = True
    model_name: str = "bert-base-cased"
    dataset_name: str = "stanfordnlp/imdb"
    wandb_project_name: str = "aml-miniproject-ernie"
    wandb_entity: str = "ERNIE-AML-2026"
    checkpoint_path: str = "bert-base-cased_imdb_checkpoint_dynamic_padding_with_metrics.pth"

    @property
    def wandb_run_name(self):
        return (
            f"{self.model_name}_imdb-bs{self.batch_size}"
            f"-lr{self.learning_rate}-ep{self.epochs}"
        )


SETTINGS = Settings()
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    logits = predictions
    predicted_labels = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predicted_labels, references=labels)
    f1 = f1_metric.compute(predictions=predicted_labels, references=labels, average="binary")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }


def load_and_prepare_datasets(settings):
    dataset = load_dataset(settings.dataset_name)
    train_data, test_data = dataset["train"], dataset["test"]

    train_data = train_data.shuffle(seed=67)

    split_idx = int(0.9 * len(train_data))
    val_data = train_data.select(range(split_idx, len(train_data)))
    train_data = train_data.select(range(split_idx))

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=settings.max_seq_length)

    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    train_data.set_format(type="torch", columns=columns)
    val_data.set_format(type="torch", columns=columns)
    test_data.set_format(type="torch", columns=columns)

    return tokenizer, train_data, val_data, test_data


def initialize_wandb(settings):
    if not settings.record_stats:
        return None

    import wandb

    wandb.init(
        project=settings.wandb_project_name,
        entity=settings.wandb_entity,
        name=settings.wandb_run_name,
        config={
            "model_name": settings.model_name,
            "dataset": settings.dataset_name,
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "max_seq_length": settings.max_seq_length,
            "epochs": settings.epochs,
        },
        reinit=True,
    )
    return wandb


def create_trainer(model, tokenizer, train_data, val_data, device, settings):
    training_args = TrainingArguments(
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01, # L2 Regularization
        learning_rate=settings.learning_rate,
        logging_steps=50,
        report_to=["wandb"] if settings.record_stats else "none",
        run_name=settings.wandb_run_name if settings.record_stats else None,
        fp16=device.type == "cuda",
        use_cpu=device.type == "cpu",
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


def load_checkpoint_or_train(model, trainer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return

    trainer.train()
    torch.save(model.state_dict(), checkpoint_path)


def predict_review(model, tokenizer, text, max_seq_length):
    model.eval()
    model_device = next(model.parameters()).device

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
        probabilities = torch.softmax(output.logits, dim=1).squeeze(0)
        predicted_label = torch.argmax(probabilities).item()

    return {
        "label": predicted_label,
        "confidence": float(probabilities[predicted_label].item()),
        "prob_negative": float(probabilities[0].item()),
        "prob_positive": float(probabilities[1].item()),
    }


def interactive_loop(model, tokenizer, settings):
    while True:
        text = input("Enter a movie review (or 'quit'): ").strip()
        if text.lower() == "quit":
            break

        result = predict_review(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_seq_length=settings.max_seq_length,
        )
        print(result)


def run():
    settings = SETTINGS
    check_torchvision_compatibility()

    tokenizer, train_data, val_data, _test_data = load_and_prepare_datasets(settings)
    model = create_model(model_name=settings.model_name, num_classes=2)
    device = get_device()
    model.to(device)

    wandb_run = initialize_wandb(settings)
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        device=device,
        settings=settings,
    )

    load_checkpoint_or_train(model=model, trainer=trainer, checkpoint_path=settings.checkpoint_path)

    model.to(device)
    evaluation = trainer.evaluate()
    print("Evaluation results:", evaluation)

    if wandb_run is not None:
        wandb_run.log({f"final/{key}": value for key, value in evaluation.items() if isinstance(value, Number)})
        wandb_run.finish()

    interactive_loop(model, tokenizer, settings)
