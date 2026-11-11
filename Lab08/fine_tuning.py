import argparse
from common import load_imdb_splits, run_training, save_run, RESULTS_DIR


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
    rows = []

    for mode, frozen in [("Frozen", True), ("Fine-tuning", False)]:
        model, tokenizer, metrics, history, elapsed, total, trainable = run_training(
            train, val, test, args.max_length, args.batch_size,
            args.epochs, freeze_encoder=frozen, seed=args.seed
        )
        rows.append({
            "mode": mode, "encoder_trainable": not frozen,
            **{key: metrics[key] for key in ["accuracy", "precision", "recall", "f1"]},
            "training_time_seconds": elapsed,
            "total_parameters": total, "trainable_parameters": trainable
        })
        if not frozen:
            model.save_pretrained(RESULTS_DIR / "bert_finetuned")
            tokenizer.save_pretrained(RESULTS_DIR / "bert_finetuned")
    save_run("fine_tuning", rows, vars(args))
    print(rows)


if __name__ == "__main__":
    main()
