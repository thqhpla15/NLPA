from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from Lab01.implementation import (
    tokenize,
    build_vocabulary,
    compute_counts,
    compute_tf,
    compute_idf,
    compute_tfidf,
    cosine_similarity
)


# ============================================================
# PART E - SO SÁNH VỚI SCIKIT-LEARN
D1 = "cat eats fish"
D2 = "dog eats fish"
D3 = "cat likes fish"

CORPUS = [D1, D2, D3]
TOLERANCE = 1e-9

def approx_equal(a, b):
    return abs(a - b) < TOLERANCE


# ============================================================
# 8.5. SO SÁNH VOCABULARY VÀ COUNT

def compare_counts():
    tokenized_docs = [
        tokenize(document)
        for document in CORPUS
    ]

    vocabulary, term_to_index = build_vocabulary(
        tokenized_docs
    )

    count_matrix = [
        compute_counts(tokens, term_to_index)
        for tokens in tokenized_docs
    ]

    vectorizer = CountVectorizer()

    sklearn_counts = vectorizer.fit_transform(
        CORPUS
    ).toarray()

    sklearn_vocabulary = (
        vectorizer.get_feature_names_out().tolist()
    )

    assert vocabulary == sklearn_vocabulary

    assert count_matrix == sklearn_counts.tolist()

    print("Vocabulary + Count: MATCH")


# ============================================================
# SO SÁNH TF

def compare_tf():
    tokenized_docs = [
        tokenize(document)
        for document in CORPUS
    ]

    vocabulary, term_to_index = build_vocabulary(
        tokenized_docs
    )

    count_matrix = [
        compute_counts(tokens, term_to_index)
        for tokens in tokenized_docs
    ]

    student_tf = [
        compute_tf(counts)
        for counts in count_matrix
    ]

    for counts, student_row in zip(
            count_matrix,
            student_tf
    ):
        total = sum(counts)

        if total == 0:
            reference_row = [
                                0.0
                            ] * len(counts)
        else:
            reference_row = [
                count / total
                for count in counts
            ]

        for student_value, reference_value in zip(
                student_row,
                reference_row
        ):
            assert approx_equal(
                student_value,
                reference_value
            )

    print("TF: MATCH")


# ============================================================
# SO SÁNH IDF

def compare_idf():
    tokenized_docs = [
        tokenize(document)
        for document in CORPUS
    ]

    vocabulary, term_to_index = build_vocabulary(
        tokenized_docs
    )

    count_matrix = [
        compute_counts(tokens, term_to_index)
        for tokens in tokenized_docs
    ]

    student_idf = compute_idf(
        count_matrix
    )

    vectorizer = TfidfVectorizer(
        smooth_idf=False,
        norm=None
    )

    vectorizer.fit(CORPUS)

    sklearn_idf = vectorizer.idf_

    for student_value, sklearn_value in zip(
            student_idf,
            sklearn_idf
    ):
        assert approx_equal(
            student_value + 1,
            sklearn_value
        )

    print(
        "IDF: MATCH AFTER SKLEARN +1 CONVENTION"
    )


# ============================================================
# SO SÁNH TF-IDF

def compare_tfidf():
    tokenized_docs = [
        tokenize(document)
        for document in CORPUS
    ]

    vocabulary, term_to_index = build_vocabulary(
        tokenized_docs
    )

    count_matrix = [
        compute_counts(tokens, term_to_index)
        for tokens in tokenized_docs
    ]

    # TF-IDF của implementation

    student_idf = compute_idf(
        count_matrix
    )

    student_tfidf = []

    for counts in count_matrix:
        tf = compute_tf(counts)

        tfidf = compute_tfidf(
            tf,
            student_idf
        )

        student_tfidf.append(tfidf)

    # Lấy IDF từ sklearn

    vectorizer = TfidfVectorizer(
        smooth_idf=False,
        norm=None
    )

    vectorizer.fit(CORPUS)

    sklearn_idf = vectorizer.idf_

    # Kiểm tra công thức TF-IDF của implementation
    # với IDF convention của sklearn

    for counts, student_row in zip(
            count_matrix,
            student_tfidf
    ):
        total = sum(counts)

        for count, student_value, sklearn_value in zip(
                counts,
                student_row,
                sklearn_idf
        ):
            if total == 0:
                expected_student = 0.0
            else:
                tf = count / total
                student_idf_value = (
                        sklearn_value - 1
                )

                expected_student = (
                        tf * student_idf_value
                )

            assert approx_equal(
                student_value,
                expected_student
            )

    print("TF-IDF: MATCH WITH SKLEARN IDF CONVENTION")


# ============================================================
# SO SÁNH COSINE SIMILARITY

def compare_cosine():
    vector_a = [
        1.0,
        1.0,
        1.0
    ]

    vector_b = [
        1.0,
        1.0,
        0.0
    ]

    student_result = cosine_similarity(
        vector_a,
        vector_b
    )

    sklearn_result = sklearn_cosine(
        [vector_a],
        [vector_b]
    )[0][0]

    assert approx_equal(
        student_result,
        sklearn_result
    )

    print("Cosine Similarity: MATCH")


# ============================================================
# CHẠY SO SÁNH

def run_comparison():
    print("=" * 60)
    print("SO SÁNH IMPLEMENTATION VỚI SCIKIT-LEARN")
    print("=" * 60)

    compare_counts()
    compare_tf()
    compare_idf()
    compare_tfidf()
    compare_cosine()

    print()
    print("Tất cả các phép so sánh đều PASS.")

if __name__ == "__main__":
    run_comparison()