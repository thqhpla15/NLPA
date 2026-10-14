import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config
from dataset_loader import DatasetLoader
from data_splitter import DataSplitter


def run_experiment(train_data, validation_data, ngram_range, experiment_name):
    # Create a TF-IDF vectorizer for the selected n-gram configuration
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        stop_words="english"
    )

    # Fit preprocessing on training data only
    start_time = time.time()
    train_vectors = vectorizer.fit_transform(train_data["text"])
    validation_vectors = vectorizer.transform(validation_data["text"])
    preprocessing_time = time.time() - start_time

    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=Config.RANDOM_SEED
    )

    start_time = time.time()
    model.fit(train_vectors, train_data["label"])
    training_time = time.time() - start_time

    # Evaluate on the same validation set
    predictions = model.predict(validation_vectors)
    true_labels = validation_data["label"]

    results = {
        "Features": experiment_name,
        "Vocabulary Size": len(vectorizer.vocabulary_),
        "Preprocessing Time (s)": round(preprocessing_time, 2),
        "Training Time (s)": round(training_time, 2),
        "Accuracy": accuracy_score(true_labels, predictions),
        "Precision": precision_score(true_labels, predictions, zero_division=0),
        "Recall": recall_score(true_labels, predictions, zero_division=0),
        "F1-score": f1_score(true_labels, predictions, zero_division=0)
    }

    print(f"\n========== {experiment_name} ==========")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return results


def main():
    Config.create_directories()

    loader = DatasetLoader()
    loader.load()

    dataframe = loader.get_dataframe(split="train")

    splitter = DataSplitter()
    train_data, validation_data = splitter.split(dataframe)

    results = []

    results.append(
        run_experiment(
            train_data,
            validation_data,
            ngram_range=(1, 1),
            experiment_name="Unigram"
        )
    )

    results.append(
        run_experiment(
            train_data,
            validation_data,
            ngram_range=(1, 2),
            experiment_name="Unigram + Bigram"
        )
    )

    comparison = pd.DataFrame(results)
    output_path = Config.OUTPUT_DIR / "ngram_comparison.csv"
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== N-GRAM COMPARISON ==========")
    print(comparison.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()