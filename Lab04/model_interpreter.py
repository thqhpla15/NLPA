import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import Config
from dataset_loader import DatasetLoader
from data_splitter import DataSplitter


def main():
    Config.create_directories()

    loader = DatasetLoader()
    loader.load()

    dataframe = loader.get_dataframe(split="train")

    splitter = DataSplitter()
    train_data, _ = splitter.split(dataframe)

    # Fit TF-IDF using training reviews only
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english"
    )

    train_vectors = vectorizer.fit_transform(train_data["text"])

    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=Config.RANDOM_SEED
    )

    model.fit(train_vectors, train_data["label"])

    # Retrieve feature names and their corresponding weights
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    feature_weights = pd.DataFrame({
        "Feature": feature_names,
        "Weight": weights
    })

    # Sort features by their contribution to each class
    top_positive = feature_weights.nlargest(20, "Weight")
    top_negative = feature_weights.nsmallest(20, "Weight")

    print("\n========== TOP POSITIVE FEATURES ==========")
    print(top_positive.to_string(index=False))

    print("\n========== TOP NEGATIVE FEATURES ==========")
    print(top_negative.to_string(index=False))

    # Save the feature weights for further inspection
    top_positive.to_csv(
        Config.OUTPUT_DIR / "top_positive_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    top_negative.to_csv(
        Config.OUTPUT_DIR / "top_negative_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFeature weights saved to output directory.")


if __name__ == "__main__":
    main()