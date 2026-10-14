import re
import numpy as np
from gensim.models import Word2Vec
from config import Config
from dataset_loader import DatasetLoader
from data_splitter import DataSplitter


def tokenize(text):
    return re.findall(r"\b[a-z]+\b", text.lower())


def average_word_vectors(tokens, model):
    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def main():
    loader = DatasetLoader()
    loader.load()

    dataframe = loader.get_dataframe("train")
    train_data, _ = DataSplitter().split(dataframe)

    sentences = [
        tokenize(text)
        for text in train_data[Config.TEXT_COLUMN]
    ]

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=1,
        seed=Config.RANDOM_SEED,
        epochs=10
    )

    sentence_1 = "this movie is good"
    sentence_2 = "this movie is not good"

    tokens_1 = tokenize(sentence_1)
    tokens_2 = tokenize(sentence_2)

    vector_1 = average_word_vectors(tokens_1, model)
    vector_2 = average_word_vectors(tokens_2, model)

    similarity = np.dot(vector_1, vector_2) / (
            np.linalg.norm(vector_1) * np.linalg.norm(vector_2) + 1e-10
    )

    print("\n========== WORD ORDER EXPERIMENT ==========")
    print(f"Sentence 1: {sentence_1}")
    print(f"Sentence 2: {sentence_2}")
    print(f"Vector dimension: {model.vector_size}")
    print(f"Cosine similarity: {similarity:.4f}")
    print(f"Vector 1 shape: {vector_1.shape}")
    print(f"Vector 2 shape: {vector_2.shape}")


if __name__ == "__main__":
    main()