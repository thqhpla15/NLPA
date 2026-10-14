import pandas as pd

from config import Config
from dataset_loader import DatasetLoader
from dataset_inspector import DatasetInspector
from data_splitter import DataSplitter
from tfidf_classifier import TfidfClassifier
from word2vec_classifier import Word2VecClassifier
from model_evaluator import ModelEvaluator
from error_analyzer import ErrorAnalyzer

def main():
    Config.create_directories()

    loader = DatasetLoader()
    loader.load()

    train_dataframe = loader.get_dataframe(split="train")

    inspector = DatasetInspector()
    inspector.inspect(train_dataframe)

    splitter = DataSplitter()
    train_data, validation_data = splitter.split(train_dataframe)

    evaluator = ModelEvaluator()
    results = []

    # Experiment 1: Train and evaluate TF-IDF
    tfidf_model = TfidfClassifier()
    tfidf_model.fit(train_data)

    tfidf_predictions = tfidf_model.predict(validation_data["text"])

    tfidf_results = evaluator.evaluate(
        validation_data["label"],
        tfidf_predictions,
        model_name="TF-IDF + Logistic Regression"
    )

    results.append(tfidf_results)
    # Inspect correct and incorrect TF-IDF predictions
    error_analyzer = ErrorAnalyzer()
    error_analyzer.analyze(
    texts=validation_data["text"],
    true_labels=validation_data["label"],
    predicted_labels=tfidf_predictions,
    model_name="TF-IDF + Logistic Regression"
)

    # Experiment 2: Train and evaluate Word2Vec
    word2vec_model = Word2VecClassifier()
    word2vec_model.fit(train_data)

    word2vec_predictions = word2vec_model.predict(
        validation_data["text"]
    )

    word2vec_results = evaluator.evaluate(
        validation_data["label"],
        word2vec_predictions,
        model_name="Word2Vec + Logistic Regression"
    )

    results.append(word2vec_results)

    # Compare both models using the same validation set
    comparison = pd.DataFrame(results)
    comparison_path = Config.OUTPUT_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))
    print(f"\nComparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()