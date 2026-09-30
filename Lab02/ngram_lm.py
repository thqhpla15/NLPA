import math
from collections import Counter

class NGramLanguageModel:

    def __init__(self, n):
        if n < 1:
            raise ValueError("n must be >= 1")

        self.n = n
        self.vocabulary = set()

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        self.total_tokens = 0

    def build_vocabulary(self, corpus):
        self.vocabulary = set()

        for sentence in corpus:
            for word in sentence:
                self.vocabulary.add(word)

        return self.vocabulary


    def count_ngrams(self, corpus):
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()

        self.total_tokens = 0

        for sentence in corpus:
            self.total_tokens += len(sentence)

            for word in sentence:
                self.unigram_counts[word] += 1

            for i in range(len(sentence) - 1):
                bigram = (sentence[i], sentence[i + 1])
                self.bigram_counts[bigram] += 1

            for i in range(len(sentence) - 2):
                trigram = (sentence[i], sentence[i + 1], sentence[i + 2])
                self.trigram_counts[trigram] += 1


    def fit(self, corpus):
        self.build_vocabulary(corpus)
        self.count_ngrams(corpus)


    def train_unigram(self):
        probabilities = {}

        for word, count in self.unigram_counts.items():
            probabilities[word] = count / self.total_tokens

        return probabilities


    def train_bigram(self):
        probabilities = {}

        for (word1, word2), count in self.bigram_counts.items():
            probabilities[(word1, word2)] = count / self.unigram_counts[word1]

        return probabilities


    def train_trigram(self):
        probabilities = {}

        for (word1, word2, word3), count in self.trigram_counts.items():
            probabilities[(word1, word2, word3)] = (
                    count / self.bigram_counts[(word1, word2)]
            )

        return probabilities


    def probability(self, context, word, smoothing=False):
        vocabulary_size = len(self.vocabulary)

        if self.n == 1 or len(context) == 0:
            count = self.unigram_counts[word]

            if smoothing:
                return (count + 1) / (self.total_tokens + vocabulary_size)

            if self.total_tokens == 0:
                return 0.0

            return count / self.total_tokens

        if self.n == 2:
            previous_word = context[-1]
            bigram = (previous_word, word)

            numerator = self.bigram_counts[bigram]
            denominator = self.unigram_counts[previous_word]

            if smoothing:
                return (numerator + 1) / (denominator + vocabulary_size)

            if denominator == 0:
                return 0.0

            return numerator / denominator

        if self.n == 3:
            previous_word1 = context[-2]
            previous_word2 = context[-1]

            trigram = (previous_word1, previous_word2, word)
            bigram = (previous_word1, previous_word2)

            numerator = self.trigram_counts[trigram]
            denominator = self.bigram_counts[bigram]

            if smoothing:
                return (numerator + 1) / (denominator + vocabulary_size)

            if denominator == 0:
                return 0.0

            return numerator / denominator


    def sentence_probability(self, sentence, smoothing=False):
        probability = 1.0

        if self.n == 1:
            for word in sentence:
                probability *= self.probability(
                    [], word, smoothing
                )

        elif self.n == 2:
            probability *= self.probability(
                [], sentence[0], smoothing
            )

            for i in range(1, len(sentence)):
                context = [sentence[i - 1]]
                probability *= self.probability(
                    context, sentence[i], smoothing
                )

        elif self.n == 3:
            probability *= self.probability(
                [], sentence[0], smoothing
            )

            if len(sentence) >= 2:
                probability *= self.probability(
                    [sentence[0]],
                    sentence[1],
                    smoothing
                )

            for i in range(2, len(sentence)):
                context = [sentence[i - 2], sentence[i - 1]]
                probability *= self.probability(
                    context, sentence[i], smoothing
                )

        return probability


    def sentence_log_probability(self, sentence, smoothing=False):
        log_probability = 0.0

        if self.n == 1:
            for word in sentence:
                probability = self.probability([], word, smoothing)

                if probability == 0:
                    return float("-inf")

                log_probability += math.log(probability)

        elif self.n == 2:
            probability = self.probability([], sentence[0], smoothing)

            if probability == 0:
                return float("-inf")

            log_probability += math.log(probability)

            for i in range(1, len(sentence)):
                context = [sentence[i - 1]]
                probability = self.probability(
                    context, sentence[i], smoothing
                )

                if probability == 0:
                    return float("-inf")

                log_probability += math.log(probability)

        elif self.n == 3:
            probability = self.probability([], sentence[0], smoothing)

            if probability == 0:
                return float("-inf")

            log_probability += math.log(probability)

            if len(sentence) >= 2:
                probability = self.probability(
                    [sentence[0]],
                    sentence[1],
                    smoothing
                )

                if probability == 0:
                    return float("-inf")

                log_probability += math.log(probability)

            for i in range(2, len(sentence)):
                context = [sentence[i - 2], sentence[i - 1]]
                probability = self.probability(
                    context, sentence[i], smoothing
                )

                if probability == 0:
                    return float("-inf")

                log_probability += math.log(probability)

        return log_probability


    def perplexity(self, sentence, smoothing=False):
        log_probability = self.sentence_log_probability(
            sentence, smoothing
        )

        if log_probability == float("-inf"):
            return float("inf")

        return math.exp(-log_probability / len(sentence))


    def next_word_distribution(self, context, smoothing=False):
        probabilities = {}

        for word in self.vocabulary:
            probabilities[word] = self.probability(
                context, word, smoothing
            )

        return sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )


    def next_word(self, context):
        probabilities = {}

        if self.n == 1:
            for word in self.vocabulary:
                probabilities[word] = self.probability([], word)

        elif self.n == 2:
            if len(context) < 1:
                return []

            previous_word = context[-1]

            for word in self.vocabulary:
                probabilities[word] = self.probability(
                    [previous_word],
                    word
                )

        elif self.n == 3:
            if len(context) < 2:
                return []

            previous_word1 = context[-2]
            previous_word2 = context[-1]

            for word in self.vocabulary:
                probabilities[word] = self.probability(
                    [previous_word1, previous_word2],
                    word
                )

        return sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )