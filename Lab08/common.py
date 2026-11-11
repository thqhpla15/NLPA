import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bert-base-uncased"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_imdb_splits(seed=42, validation_size=0.1, train_size=None, test_size=None):
    dataset = load_dataset("stanfordnlp/imdb")
    train_texts = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])

    if train_size is not None:
        train_texts, _, train_labels, _ = train_test_split(
            train_texts, train_labels,
            train_size=min(train_size, len(train_texts)),
            random_state=seed, stratify=train_labels
        )

    train_x, val_x, train_y, val_y = train_test_split(
        train_texts, train_labels, test_size=validation_size,
        random_state=seed, stratify=train_labels
    )

    if test_size is not None:
        test_x, _, test_y, _ = train_test_split(
            test_texts, test_labels,
            train_size=min(test_size, len(test_texts)),
            random_state=seed, stratify=test_labels
        )
    else:
        test_x, test_y = test_texts, test_labels

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def make_loader(texts, labels, tokenizer, max_length, batch_size, shuffle=False):
    return DataLoader(
        TextDataset(texts, labels, tokenizer, max_length),
        batch_size=batch_size, shuffle=shuffle
    )


def load_tokenizer(model_name=MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def build_model(model_name=MODEL_NAME):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )


def evaluate(model, loader, device="cpu"):
    model.eval()
    all_labels, all_predictions, all_probabilities = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision, "recall": recall, "f1": f1,
        "labels": all_labels, "predictions": all_predictions,
        "probabilities": all_probabilities
    }


def train_model(model, train_loader, val_loader, epochs=1, learning_rate=2e-5,
                device="cpu", freeze_encoder=False):
    model.to(device)
    if freeze_encoder:
        for name, parameter in model.named_parameters():
            if "classifier" not in name and "score" not in name:
                parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    start = time.time()
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()

        val = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / max(len(train_loader), 1),
            "val_accuracy": val["accuracy"], "val_f1": val["f1"]
        })

    return history, time.time() - start


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_run(name, rows, config):
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    with open(RESULTS_DIR / f"{name}_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def save_json(name, data):
    with open(RESULTS_DIR / name, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def run_training(train_data, val_data, test_data, max_length=128, batch_size=8,
                 epochs=1, learning_rate=2e-5, freeze_encoder=False, seed=42):
    set_seed(seed)
    tokenizer = load_tokenizer()
    train_loader = make_loader(*train_data, tokenizer, max_length, batch_size, True)
    val_loader = make_loader(*val_data, tokenizer, max_length, batch_size)
    test_loader = make_loader(*test_data, tokenizer, max_length, batch_size)
    model = build_model()
    history, elapsed = train_model(
        model, train_loader, val_loader, epochs, learning_rate,
        "cpu", freeze_encoder
    )
    metrics = evaluate(model, test_loader)
    total, trainable = count_parameters(model)
    return model, tokenizer, metrics, history, elapsed, total, trainable
