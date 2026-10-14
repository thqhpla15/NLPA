from dataset_loader import DatasetLoader
from data_splitter import DataSplitter
from tfidf_classifier import TfidfClassifier

def main():
    # Load the IMDb dataset
    loader = DatasetLoader()
    loader.load()

    dataframe = loader.get_dataframe(split="train")

    # Use the same training procedure as the previous experiments
    splitter = DataSplitter()
    train_data, _ = splitter.split(dataframe)

    # Train the TF-IDF model
    model = TfidfClassifier()
    model.fit(train_data)

    print("\n========== SENTIMENT PREDICTOR ==========")
    print("Enter a movie review to predict its sentiment.")
    print("Type 'exit' to quit.")

    while True:
        review = input("\nYour review: ").strip()

        if review.lower() == "exit":
            print("Program ended.")
            break

        if not review:
            print("Please enter a non-empty review.")
            continue

        sentiment = model.predict_sentiment(review)
        print(f"Predicted sentiment: {sentiment}")


if __name__ == "__main__":
    main()