import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from config import Config
from dataset_loader import DatasetLoader
from tfidf_classifier import TfidfClassifier

def main():
    Config.create_directories()

    # Load the IMDb dataset
    loader = DatasetLoader()
    loader.load()

    # Prepare the original training and test sets
    train_data = loader.get_dataframe(split="train")
    test_data = loader.get_dataframe(split="test")

    # Recreate the same training split used in previous experiments
    from data_splitter import DataSplitter

    splitter = DataSplitter()
    train_data, _ = splitter.split(train_data)

    # Train the selected model using the training data only
    model = TfidfClassifier()
    model.fit(train_data)

    # Evaluate the model on the untouched test set
    predictions = model.predict(test_data[Config.TEXT_COLUMN])
    true_labels = test_data[Config.LABEL_COLUMN]

    # Calculate the final evaluation metrics
    results = {
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy": accuracy_score(true_labels, predictions),
        "Precision": precision_score(true_labels, predictions, zero_division=0),
        "Recall": recall_score(true_labels, predictions, zero_division=0),
        "F1-score": f1_score(true_labels, predictions, zero_division=0)
    }

    print("\n========== FINAL TEST EVALUATION ==========")
    print(f"Test samples: {len(test_data)}")
    print(f"Accuracy:  {results['Accuracy']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall:    {results['Recall']:.4f}")
    print(f"F1-score:  {results['F1-score']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        true_labels,
        predictions,
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

    # Save final test results
    output_path = Config.OUTPUT_DIR / "final_test_results.csv"
    pd.DataFrame([results]).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Final results saved to: {output_path}")


if __name__ == "__main__":
    main()