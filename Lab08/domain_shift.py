import argparse
import matplotlib.pyplot as plt
from datasets import load_dataset
from common import (
    load_imdb_splits, load_tokenizer, make_loader, build_model,
    train_model, evaluate, save_run, RESULTS_DIR
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--imdb-test-size", type=int, default=1000)
    parser.add_argument("--other-test-size", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train, val, imdb_test = load_imdb_splits(
        seed=args.seed, train_size=args.train_size, test_size=args.imdb_test_size
    )
    other = load_dataset("rotten_tomatoes", split="test")
    other_test = (
        list(other["text"])[:args.other_test_size],
        list(other["label"])[:args.other_test_size]
    )

    tokenizer = load_tokenizer()
    train_loader = make_loader(*train, tokenizer, args.max_length, args.batch_size, True)
    val_loader = make_loader(*val, tokenizer, args.max_length, args.batch_size)
    imdb_loader = make_loader(*imdb_test, tokenizer, args.max_length, args.batch_size)
    other_loader = make_loader(*other_test, tokenizer, args.max_length, args.batch_size)
    model = build_model()
    _, elapsed = train_model(model, train_loader, val_loader, args.epochs, 2e-5)
    in_metrics = evaluate(model, imdb_loader)
    out_metrics = evaluate(model, other_loader)

    rows = []
    for domain, metrics in [
        ("IMDb test (in-domain)", in_metrics),
        ("Rotten Tomatoes test (cross-dataset)", out_metrics)
    ]:
        rows.append({
            "evaluation_domain": domain,
            **{key: metrics[key] for key in ["accuracy", "precision", "recall", "f1"]},
            "training_time_seconds": elapsed
        })
    save_run("domain_shift", rows, vars(args))
    plt.figure()
    plt.bar([r["evaluation_domain"] for r in rows], [r["accuracy"] for r in rows])
    plt.ylabel("Accuracy")
    plt.title("IMDb vs Rotten Tomatoes")
    plt.xticks(rotation=10, ha="right")
    plt.grid(axis="y")
    plt.savefig(RESULTS_DIR / "domain_shift.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(rows)


if __name__ == "__main__":
    main()
