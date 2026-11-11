import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import load_imdb_splits, make_loader, evaluate, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/bert_finetuned")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _, _, test = load_imdb_splits(
        seed=args.seed, train_size=2000, test_size=args.test_size
    )
    texts, labels = test
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    loader = make_loader(texts, labels, tokenizer, args.max_length, args.batch_size)
    metrics = evaluate(model, loader)

    rows = []
    for text, true_label, pred, probs in zip(
        texts, metrics["labels"], metrics["predictions"], metrics["probabilities"]
    ):
        rows.append({
            "text": text, "true_label": true_label, "prediction": pred,
            "confidence": max(probs),
            "error_type": "false_positive" if pred == 1 and true_label == 0
                          else "false_negative" if pred == 0 and true_label == 1
                          else "correct"
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "error_predictions.csv", index=False)

    matrix = confusion_matrix(metrics["labels"], metrics["predictions"])
    display = ConfusionMatrixDisplay(matrix, display_labels=["negative", "positive"])
    display.plot()
    plt.title("IMDb confusion matrix")
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()

    errors = frame[frame["error_type"] != "correct"].sort_values(
        "confidence", ascending=False
    )
    errors.head(20).to_csv(RESULTS_DIR / "most_confident_errors.csv", index=False)

    frame["confidence_group"] = pd.cut(
        frame["confidence"], [0, 0.60, 0.80, 1.000001],
        labels=["Low", "Medium", "High"], include_lowest=True
    )
    summary = frame.groupby("confidence_group", observed=False).apply(
        lambda group: pd.Series({
            "count": len(group),
            "accuracy": (group["true_label"] == group["prediction"]).mean()
        })
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "confidence_analysis.csv", index=False)

    plt.figure()
    plt.bar(summary["confidence_group"].astype(str), summary["accuracy"])
    plt.xlabel("Confidence group")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by confidence group")
    plt.ylim(0, 1)
    plt.savefig(RESULTS_DIR / "confidence_analysis.png", dpi=160, bbox_inches="tight")
    plt.close()

    print({key: metrics[key] for key in ["accuracy", "precision", "recall", "f1"]})
    print("False positives:", int((frame["error_type"] == "false_positive").sum()))
    print("False negatives:", int((frame["error_type"] == "false_negative").sum()))


if __name__ == "__main__":
    main()
