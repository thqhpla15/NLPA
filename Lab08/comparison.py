import argparse
import time
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import (
    load_imdb_splits, make_loader, train_model, evaluate,
    count_parameters, save_run, set_seed
)


class ScratchTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_size=128, layers=2, heads=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads,
            dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=attention_mask == 0)
        logits = self.classifier(hidden[:, 0])
        loss = None if labels is None else nn.functional.cross_entropy(logits, labels)
        return type("Output", (), {"loss": loss, "logits": logits})


def train_scratch(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    start = time.time()
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
        evaluate(model, val_loader)
    return time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    train, val, test = load_imdb_splits(
        seed=args.seed, train_size=args.train_size, test_size=args.test_size
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_loader = make_loader(*train, tokenizer, args.max_length, args.batch_size, True)
    val_loader = make_loader(*val, tokenizer, args.max_length, args.batch_size)
    test_loader = make_loader(*test, tokenizer, args.max_length, args.batch_size)

    scratch = ScratchTransformerClassifier(tokenizer.vocab_size, args.max_length)
    scratch_time = train_scratch(scratch, train_loader, val_loader, args.epochs, 2e-4)
    scratch_metrics = evaluate(scratch, test_loader)
    scratch_total, scratch_trainable = count_parameters(scratch)

    pretrained = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    _, pretrained_time = train_model(
        pretrained, train_loader, val_loader, args.epochs, 2e-5
    )
    pretrained_metrics = evaluate(pretrained, test_loader)
    pretrained_total, pretrained_trainable = count_parameters(pretrained)

    rows = []
    for name, metrics, elapsed, total, trainable in [
        ("Transformer from scratch (small)", scratch_metrics, scratch_time,
         scratch_total, scratch_trainable),
        ("BERT-base-uncased fine-tuned", pretrained_metrics, pretrained_time,
         pretrained_total, pretrained_trainable)
    ]:
        rows.append({
            "model": name,
            **{key: metrics[key] for key in ["accuracy", "precision", "recall", "f1"]},
            "training_time_seconds": elapsed,
            "total_parameters": total, "trainable_parameters": trainable
        })
    save_run("comparison", rows, vars(args))
    print(rows)


if __name__ == "__main__":
    main()
