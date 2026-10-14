import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from config import Config

class ModelEvaluator:
    def evaluate(self, true_labels, predicted_labels, model_name="TF-IDF"):
        # Calculate standard binary classification metrics
        results = {
            "Model": model_name,
            "Accuracy": accuracy_score(true_labels, predicted_labels),
            "Precision": precision_score(
                true_labels, predicted_labels, zero_division=0
            ),
            "Recall": recall_score(
                true_labels, predicted_labels, zero_division=0
            ),
            "F1-score": f1_score(
                true_labels, predicted_labels, zero_division=0
            )
        }

        print(f"\n========== {model_name} EVALUATION ==========")

        for metric, value in results.items():
            if metric != "Model":
                print(f"{metric}: {value:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                true_labels,
                predicted_labels,
                target_names=["Negative", "Positive"],
                zero_division=0
            )
        )

        # Generate and save the confusion matrix
        display = ConfusionMatrixDisplay.from_predictions(
            true_labels,
            predicted_labels,
            display_labels=["Negative", "Positive"],
            cmap="Blues"
        )

        display.ax_.set_title(f"{model_name} Confusion Matrix")
        display.figure_.tight_layout()

        # Create a unique filename for each model
        safe_model_name = (model_name.lower().replace(" + ", "_").replace(" ", "_"))

        output_path = Config.OUTPUT_DIR / f"{safe_model_name}_confusion_matrix.png"
        display.figure_.savefig(output_path, dpi=300)
        plt.close(display.figure_)

        print(f"Confusion matrix saved to: {output_path}")

        return results