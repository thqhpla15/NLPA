import pandas as pd
from config import Config

class ErrorAnalyzer:
    def analyze(self, texts, true_labels, predicted_labels, model_name="TF-IDF"):
        # Build a DataFrame containing predictions and their actual labels
        results = pd.DataFrame({
            "text": list(texts),
            "true_label": list(true_labels),
            "predicted_label": list(predicted_labels)
        })

        results["true_sentiment"] = results["true_label"].map(
            Config.LABEL_NAMES
        )
        results["predicted_sentiment"] = results["predicted_label"].map(
            Config.LABEL_NAMES
        )

        # Separate correct and incorrect predictions
        correct = results[
            results["true_label"] == results["predicted_label"]
            ]
        incorrect = results[
            results["true_label"] != results["predicted_label"]
            ]

        print(f"\n========== {model_name} ERROR ANALYSIS ==========")
        print(f"Correct predictions: {len(correct)}")
        print(f"Incorrect predictions: {len(incorrect)}")

        # Display a small sample from each group
        self._display_examples(correct, "CORRECT EXAMPLES", 3)
        self._display_examples(incorrect, "MISCLASSIFIED EXAMPLES", 3)

        # Save all misclassified reviews for further inspection
        output_path = Config.OUTPUT_DIR / "error_analysis.csv"
        incorrect.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"\nMisclassified reviews saved to: {output_path}")

        return correct, incorrect


    @staticmethod
    def _display_examples(dataframe, title, number_of_examples):
        print(f"\n--- {title} ---")

        if dataframe.empty:
            print("No examples available.")
            return

        # Use a fixed random seed so the displayed examples are reproducible
        examples = dataframe.sample(
            n=min(number_of_examples, len(dataframe)),
            random_state=Config.RANDOM_SEED
        )

        for index, (_, row) in enumerate(examples.iterrows(), start=1):
            print(f"\nExample {index}")
            print(f"Actual: {row['true_sentiment']}")
            print(f"Predicted: {row['predicted_sentiment']}")
            print(f"Review: {row['text'][:1000]}")