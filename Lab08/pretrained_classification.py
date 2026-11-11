import argparse
from common import (
    load_imdb_splits, load_tokenizer, make_loader, build_model,
    train_model, evaluate, count_parameters, save_run, save_json, RESULTS_DIR
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, val, test = load_imdb_splits(
        seed=args.seed, train_size=args.train_size, test_size=args.test_size
    )
    tokenizer = load_tokenizer()
    train_loader = make_loader(*train, tokenizer, args.max_length, args.batch_size, True)
    val_loader = make_loader(*val, tokenizer, args.max_length, args.batch_size)
    test_loader = make_loader(*test, tokenizer, args.max_length, args.batch_size)
    model = build_model()
    history, elapsed = train_model(
        model, train_loader, val_loader, args.epochs, 2e-5
    )
    metrics = evaluate(model, test_loader)
    total, trainable = count_parameters(model)

    model.save_pretrained(RESULTS_DIR / "bert_finetuned")
    tokenizer.save_pretrained(RESULTS_DIR / "bert_finetuned")
    row = {
        "model": "bert-base-uncased",
        **{key: metrics[key] for key in ["accuracy", "precision", "recall", "f1"]},
        "training_time_seconds": elapsed,
        "total_parameters": total, "trainable_parameters": trainable
    }
    save_run("pretrained_classification", [row], vars(args))
    save_json("pretrained_history.json", history)
    print(row)


if __name__ == "__main__":
    main()
