import argparse
import matplotlib.pyplot as plt
from common import load_imdb_splits, run_training, save_run, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train, val, test = load_imdb_splits(
        seed=args.seed, train_size=args.train_size, test_size=args.test_size
    )
    rows = []

    for length in args.lengths:
        _, _, metrics, _, elapsed, _, _ = run_training(
            train, val, test, length, args.batch_size, args.epochs, seed=args.seed
        )
        rows.append({
            "max_length": length,
            **{key: metrics[key] for key in ["accuracy", "precision", "recall", "f1"]},
            "training_time_seconds": elapsed
        })
        print(rows[-1])

    save_run("sequence_length", rows, vars(args))
    plt.figure()
    plt.plot([r["max_length"] for r in rows], [r["accuracy"] for r in rows],
             marker="o", label="Accuracy")
    plt.plot([r["max_length"] for r in rows], [r["f1"] for r in rows],
             marker="s", label="F1")
    plt.xlabel("Maximum sequence length")
    plt.ylabel("Score")
    plt.title("Sequence length comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "sequence_length.png", dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
