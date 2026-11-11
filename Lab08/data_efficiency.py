import argparse
import matplotlib.pyplot as plt
from common import load_imdb_splits, run_training, save_run, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    full_train, val, test = load_imdb_splits(
        seed=args.seed, train_size=args.train_size, test_size=args.test_size
    )
    texts, labels = full_train
    rows = []

    for fraction in [0.10, 0.25, 0.50, 1.00]:
        size = max(2, int(len(texts) * fraction))
        subset = (texts[:size], labels[:size])
        _, _, metrics, _, elapsed, _, _ = run_training(
            subset, val, test, args.max_length, args.batch_size,
            args.epochs, seed=args.seed
        )
        rows.append({
            "training_fraction": fraction, "training_examples": size,
            "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "training_time_seconds": elapsed
        })
        print(rows[-1])

    save_run("data_efficiency", rows, vars(args))
    plt.figure()
    plt.plot([r["training_fraction"] * 100 for r in rows],
             [r["accuracy"] for r in rows], marker="o", label="Accuracy")
    plt.plot([r["training_fraction"] * 100 for r in rows],
             [r["f1"] for r in rows], marker="s", label="F1")
    plt.xlabel("Training data (%)")
    plt.ylabel("Score")
    plt.title("BERT data efficiency on IMDb")
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "data_efficiency.png", dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
