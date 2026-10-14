import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from config import Config

class TfidfClassifier:
    def __init__(self):
        # Initialize the TF-IDF vectorizer and classifier
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english"
        )

        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=Config.RANDOM_SEED
        )

        self.training_time = 0
        self.vocabulary_size = 0


    def fit(self, train_data):
        # Learn the vocabulary and IDF weights from training texts only
        X_train = self.vectorizer.fit_transform(train_data["text"])
        y_train = train_data["label"]

        # Train the classification model
        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        self.vocabulary_size = len(self.vectorizer.vocabulary_)

        print("\n========== TF-IDF MODEL ==========")
        print(f"Vocabulary size: {self.vocabulary_size}")
        print(f"Training time: {self.training_time:.2f} seconds")

        return self


    def predict(self, texts):
        # Transform new texts without refitting the vectorizer
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)


    def transform(self, texts):
        # Expose TF-IDF vectors for inspection if needed
        return self.vectorizer.transform(texts)


    def predict_sentiment(self, text):
        # Transform the input review using the trained TF-IDF vectorizer
        text_vector = self.vectorizer.transform([text])

        # Predict the sentiment label
        prediction = self.classifier.predict(text_vector)[0]

        return Config.LABEL_NAMES[prediction]