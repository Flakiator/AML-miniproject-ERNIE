import os
import re
from collections import Counter
from dataclasses import dataclass
from html import unescape
from numbers import Number
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from sentiment_app.environment import check_torchvision_compatibility, get_device
from sentiment_app.model import create_glove_baseline_model, create_glove_simple_model, create_model


@dataclass(frozen=True)
class Settings:
    batch_size: int = 128 #16
    learning_rate: float = 0.05 #1e-5
    max_seq_length: int = 264 #64
    epochs: int = 15
    record_stats: bool = True
    model_type: str = "glove_simple"  # "bert" | "glove" | "glove_simple"
    model_name: str = "bert-base-cased"
    dataset_name: str = "stanfordnlp/imdb"
    wandb_project_name: str = "aml-miniproject-ernie"
    wandb_entity: str = "ERNIE-AML-2026"
    glove_vectors_path: str = (
        "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
    )
    glove_freeze_embeddings: bool = True
    glove_min_frequency: int = 2
    glove_vocab_size: int = 50000
    glove_hidden_dim: int = 128
    checkpoint_path: str = (
        f"{model_type}_{model_name}_imdb_checkpoint_dynamic_padding_with_metrics-"
        f"{max_seq_length}-{batch_size}.pth"
    )

    @property
    def wandb_run_name(self):
        return (
            f"{self.model_type}-{self.model_name}_imdb-bs{self.batch_size}"
            f"-lr{self.learning_rate}-ep{self.epochs}"
        )


SETTINGS = Settings()
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
TOKEN_PATTERN = re.compile(r"\b\w+\b|[^\w\s]")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


class SimpleWordTokenizer:
    def __init__(self, vocab, max_seq_length, lowercase=True):
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.lowercase = lowercase
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.pad_token_id = vocab[PAD_TOKEN]
        self.unk_token_id = vocab[UNK_TOKEN]

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        return TOKEN_PATTERN.findall(text)

    def encode(self, text, truncation=True, padding="max_length", max_length=None):
        max_length = max_length or self.max_seq_length
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]

        if truncation:
            token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        if padding == "max_length":
            pad_length = max(max_length - len(token_ids), 0)
            token_ids = token_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }

    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=True,
        padding="max_length",
        max_length=None,
    ):
        if isinstance(text, str):
            encoded = self.encode(
                text,
                truncation=truncation,
                padding=padding,
                max_length=max_length,
            )
            if return_tensors == "pt":
                return {key: torch.tensor([value], dtype=torch.long) for key, value in encoded.items()}
            return encoded

        batch = [
            self.encode(item, truncation=truncation, padding=padding, max_length=max_length)
            for item in text
        ]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
                "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
            }
        return {
            "input_ids": [item["input_ids"] for item in batch],
            "attention_mask": [item["attention_mask"] for item in batch],
        }


def glove_data_collator(features):
    return {
        "input_ids": torch.stack([feature["input_ids"] for feature in features]),
        "attention_mask": torch.stack([feature["attention_mask"] for feature in features]),
        "labels": torch.stack([feature["label"] for feature in features]),
    }


def build_glove_vocabulary(train_texts, settings):
    counter = Counter()
    for text in train_texts:
        counter.update(TOKEN_PATTERN.findall(text.lower()))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    for token, frequency in counter.most_common():
        if frequency < settings.glove_min_frequency:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= settings.glove_vocab_size:
            break
    return vocab


def load_glove_embedding_matrix(vocab, vectors_path):
    vectors_path = Path(vectors_path)
    if not vectors_path.is_absolute():
        vectors_path = Path.cwd() / vectors_path

    if not vectors_path.exists():
        raise FileNotFoundError(f"GloVe vectors file not found: {vectors_path}")

    embedding_dim = None
    found_tokens = 0
    embedding_matrix = None

    with vectors_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue

            if embedding_dim is None:
                embedding_dim = len(parts) - 1
                embedding_matrix = np.random.normal(
                    loc=0.0,
                    scale=0.6,
                    size=(len(vocab), embedding_dim),
                ).astype(np.float32)
                embedding_matrix[vocab[PAD_TOKEN]] = np.zeros(embedding_dim, dtype=np.float32)

            if len(parts) <= embedding_dim:
                continue

            token = " ".join(parts[:-embedding_dim])
            values = parts[-embedding_dim:]
            token_id = vocab.get(token)
            if token_id is None:
                continue

            embedding_matrix[token_id] = np.asarray(values, dtype=np.float32)
            found_tokens += 1

    if embedding_matrix is None:
        raise ValueError(f"No embeddings could be read from: {vectors_path}")

    print(
        f"Loaded {found_tokens} pretrained vectors out of {len(vocab)} vocabulary entries "
        f"from {vectors_path.name}"
    )
    return embedding_matrix


def strip_html_tags(text):
    text = unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    return " ".join(text.split())


def load_and_prepare_bert_datasets(settings):
    dataset = load_dataset(settings.dataset_name)

    train_data, test_data = dataset["train"], dataset["test"]

    train_data = train_data.shuffle()

    split_idx = int(0.9 * len(train_data))
    val_data = train_data.select(range(split_idx, len(train_data))).shuffle()
    train_data = train_data.select(range(split_idx)).shuffle()
    test_data = test_data.shuffle()

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

    def tokenize(batch):
        cleaned_texts = [strip_html_tags(text) for text in batch["text"]]
        return tokenizer(cleaned_texts, truncation=True, max_length=settings.max_seq_length)

    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    train_data.set_format(type="torch", columns=columns)
    val_data.set_format(type="torch", columns=columns)
    test_data.set_format(type="torch", columns=columns)

    return tokenizer, train_data, val_data, test_data, None, None


def load_and_prepare_glove_datasets(settings):
    dataset = load_dataset(settings.dataset_name)
    train_data, test_data = dataset["train"], dataset["test"]

    train_data = train_data.shuffle(seed=67)

    split_idx = int(0.9 * len(train_data))
    val_data = train_data.select(range(split_idx, len(train_data)))
    train_data = train_data.select(range(split_idx))

    vocab = build_glove_vocabulary(train_data["text"], settings)
    tokenizer = SimpleWordTokenizer(vocab=vocab, max_seq_length=settings.max_seq_length)
    embedding_matrix = load_glove_embedding_matrix(vocab, settings.glove_vectors_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=settings.max_seq_length)

    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    train_data.set_format(type="torch", columns=columns)
    val_data.set_format(type="torch", columns=columns)
    test_data.set_format(type="torch", columns=columns)

    model_kwargs = {
        "embedding_matrix": embedding_matrix,
        "num_classes": 2,
        "padding_idx": tokenizer.pad_token_id,
        "freeze_embeddings": settings.glove_freeze_embeddings,
        "hidden_dim": settings.glove_hidden_dim,
    }
    return tokenizer, train_data, val_data, test_data, model_kwargs, glove_data_collator


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
    if settings.model_type == "bert":
        return load_and_prepare_bert_datasets(settings)
    if settings.model_type in ("glove", "glove_simple"):
        tokenizer, train_data, val_data, test_data, model_kwargs, collator = (
            load_and_prepare_glove_datasets(settings)
        )
        # Simple mean-pooling model doesn't use hidden_dim
        if settings.model_type == "glove_simple":
            model_kwargs.pop("hidden_dim", None)
        return tokenizer, train_data, val_data, test_data, model_kwargs, collator
    raise ValueError(f"Unsupported model_type: {settings.model_type}")


def initialize_wandb(settings):
    if not settings.record_stats:
        return None

    import wandb

    wandb.init(
        project=settings.wandb_project_name,
        entity=settings.wandb_entity,
        name=settings.wandb_run_name,
        config={
            "model_type": settings.model_type,
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


def create_trainer(model, tokenizer, train_data, val_data, device, settings, data_collator=None):
    training_args = TrainingArguments(
        warmup_ratio=0.5,
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        # eval_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01, # L2 Regularization
        learning_rate=settings.learning_rate,
        logging_steps=50,
        report_to=["wandb"] if settings.record_stats else "none",
        run_name=settings.wandb_run_name if settings.record_stats else None,
        fp16=device.type == "cuda",
        use_cpu=device.type == "cpu",
        output_dir="output_dir"
    )
    if data_collator is None:
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

    tokenizer, train_data, val_data, _test_data, model_kwargs, data_collator = load_and_prepare_datasets(settings)
    if settings.model_type == "bert":
        model = create_model(model_name=settings.model_name, num_classes=2)
    elif settings.model_type == "glove":
        model = create_glove_baseline_model(**model_kwargs)
    elif settings.model_type == "glove_simple":
        model = create_glove_simple_model(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {settings.model_type}")
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
        data_collator=data_collator,
    )

    load_checkpoint_or_train(model=model, trainer=trainer, checkpoint_path=settings.checkpoint_path)

    model.to(device)
    evaluation = trainer.evaluate()
    print("Evaluation results:", evaluation)

    if wandb_run is not None:
        wandb_run.log({f"final/{key}": value for key, value in evaluation.items() if isinstance(value, Number)})
        wandb_run.finish()

    interactive_loop(model, tokenizer, settings)