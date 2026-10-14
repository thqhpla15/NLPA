import re
import time

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

from config import Config


class Word2VecClassifier:
    def __init__(self, vector_size=100, window=5, min_count=2):
        # Configure the Word2Vec model
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

        self.word2vec_model = None
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=Config.RANDOM_SEED
        )

        self.training_time = 0


    def tokenize(self, text):
        # Convert text to lowercase and extract word tokens
        return re.findall(r"\b\w+\b", text.lower())


    def fit(self, train_data):
        start_time = time.time()

        # Tokenize training reviews only
        tokenized_reviews = [
            self.tokenize(text)
            for text in train_data["text"]
        ]

        # Learn word embeddings from training reviews
        self.word2vec_model = Word2Vec(
            sentences=tokenized_reviews,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            seed=Config.RANDOM_SEED,
            epochs=10
        )

        # Convert each training review into an averaged word vector
        X_train = self.transform(train_data["text"])
        y_train = train_data["label"]

        # Train Logistic Regression on review vectors
        self.classifier.fit(X_train, y_train)

        self.training_time = time.time() - start_time

        print("\n========== WORD2VEC MODEL ==========")
        print(f"Vector size: {self.vector_size}")
        print(f"Vocabulary size: {len(self.word2vec_model.wv)}")
        print(f"Training time: {self.training_time:.2f} seconds")

        return self


    def transform(self, texts):
        # Represent each review by averaging its known word vectors
        vectors = []

        for text in texts:
            tokens = self.tokenize(text)

            word_vectors = [
                self.word2vec_model.wv[token]
                for token in tokens
                if token in self.word2vec_model.wv
            ]

            if word_vectors:
                review_vector = np.mean(word_vectors, axis=0)
            else:
                review_vector = np.zeros(self.vector_size)

            vectors.append(review_vector)

        return np.array(vectors, dtype=np.float32)


    def predict(self, texts):
        # Predict sentiment labels for new reviews
        X = self.transform(texts)
        return self.classifier.predict(X)