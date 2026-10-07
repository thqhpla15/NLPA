import numpy as np

class WordEmbedding:
    def build_vocabulary(self, corpus):
        vocabulary = set()

        for sentence in corpus:
            words = sentence.split()
            vocabulary.update(words)

        vocabulary = sorted(vocabulary)

        word_to_index = {
            word: i for i, word in enumerate(vocabulary)
        }

        return vocabulary, word_to_index


    def build_cooccurrence_matrix(self, corpus, vocabulary, word_to_index, window=2):
        matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=int)

        for sentence in corpus:
            words = sentence.split()

            for i, word in enumerate(words):
                word_index = word_to_index[word]

                start = max(0, i - window)
                end = min(len(words), i + window + 1)

                for j in range(start, end):
                    if i == j:
                        continue

                    context_word = words[j]
                    context_index = word_to_index[context_word]

                    matrix[word_index][context_index] += 1

        return matrix


    def cosine_similarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    
    def most_similar(self, word, matrix, vocabulary, word_to_index, top_k=5):
        word_index = word_to_index[word]
        word_vector = matrix[word_index]

        similarities = []

        for i, vector in enumerate(matrix):
            if i == word_index:
                continue

            similarity = self.cosine_similarity(word_vector, vector)
            similarities.append((vocabulary[i], similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]