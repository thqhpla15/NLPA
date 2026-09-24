import math

# ============================================================
# PART E - CORE IMPLEMENTATION
# 8.1. Mục tiêu
# Tự xây một phiên bản TF-IDF tối giản.
# 8.2. Các hàm cần xây dựng
# - build_vocabulary()
# - compute_counts()
# - compute_tf()
# - compute_idf()
# - compute_tfidf()
# - cosine_similarity()


# ============================================================
# 8.2. TOKENIZE
# Chuyển text thành các token:
# - chuyển về chữ thường
# - chỉ giữ các ký tự chữ
# - gặp dấu cách hoặc dấu câu thì kết thúc một từ


def tokenize(text):
    text = text.lower()

    tokens = []
    current = []

    for ch in text:
        if ch.isalpha():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []

    if current:
        tokens.append("".join(current))

    return tokens


# ============================================================
# 8.2. BUILD VOCABULARY
# Gom tất cả các từ xuất hiện trong corpus thành vocabulary.
# Sắp xếp alphabet để thứ tự các cột trong ma trận cố định.


def build_vocabulary(tokenized_docs):
    unique_terms = set()

    for tokens in tokenized_docs:
        unique_terms.update(tokens)

    vocabulary = sorted(unique_terms)

    term_to_index = {
        term: index
        for index, term in enumerate(vocabulary)
    }

    return vocabulary, term_to_index


# ============================================================
# 8.2. COMPUTE COUNTS
# Đếm số lần mỗi từ xuất hiện trong một document.


def compute_counts(tokens, term_to_index):
    counts = [0] * len(term_to_index)

    for token in tokens:
        index = term_to_index.get(token)

        if index is not None:
            counts[index] += 1

    return counts


# ============================================================
# 8.2. COMPUTE TF
# TF = số lần từ xuất hiện / tổng số từ trong document.


def compute_tf(counts):
    total_tokens = sum(counts)

    if total_tokens == 0:
        return [0.0] * len(counts)

    return [
        count / total_tokens
        for count in counts
    ]


# ============================================================
# 8.2. COMPUTE IDF
# IDF = log(N / df)
# N là số document.
# df là số document chứa term.


def compute_idf(count_matrix):
    number_of_documents = len(count_matrix)

    if number_of_documents == 0:
        return []

    vocabulary_size = len(count_matrix[0])
    document_frequency = [0] * vocabulary_size

    for counts in count_matrix:
        for j, count in enumerate(counts):
            if count > 0:
                document_frequency[j] += 1

    idf = []

    for df in document_frequency:
        if df == 0:
            idf.append(0.0)
        else:
            idf.append(
                math.log(number_of_documents / df)
            )

    return idf


# ============================================================
# 8.2. COMPUTE TF-IDF
# TF-IDF = TF * IDF.


def compute_tfidf(tf, idf):
    return [
        tf_value * idf_value
        for tf_value, idf_value in zip(tf, idf)
    ]


# ============================================================
# 8.2. COSINE SIMILARITY
# Dùng cosine similarity để đo độ giống nhau giữa hai vector.
# cosine = (A . B) / (|A| * |B|)


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ============================================================
# 8.3. CORPUS KIỂM THỬ

D1 = "cat eats fish"
D2 = "dog eats fish"
D3 = "cat likes fish"

TEST_CORPUS = [D1, D2, D3]


# ============================================================
# 8.4. UNIT TESTS
TOLERANCE = 1e-9

def approx_equal(a, b):
    return abs(a - b) < TOLERANCE


# Test tokenize

def test_tokenize():
    assert tokenize(D1) == [
        "cat",
        "eats",
        "fish"
    ]

    assert tokenize(D2) == [
        "dog",
        "eats",
        "fish"
    ]

    assert tokenize(D3) == [
        "cat",
        "likes",
        "fish"
    ]

    assert tokenize(
        "Cat, EATS fish!!"
    ) == [
               "cat",
               "eats",
               "fish"
           ]

    print("test_tokenize: OK")


# Test build_vocabulary

def test_build_vocabulary():
    tokenized_docs = [
        tokenize(document)
        for document in TEST_CORPUS
    ]

    vocabulary, term_to_index = build_vocabulary(
        tokenized_docs
    )

    expected_vocabulary = [
        "cat",
        "dog",
        "eats",
        "fish",
        "likes"
    ]

    assert vocabulary == expected_vocabulary

    assert len(term_to_index) == 5

    assert term_to_index["cat"] == 0
    assert term_to_index["fish"] == 3

    print("test_build_vocabulary: OK")


# Test compute_counts

def test_compute_counts():
    tokenized_docs = [
        tokenize(document)
        for document in TEST_CORPUS
    ]

    vocabulary, term_to_index = build_vocabulary(
        tokenized_docs
    )

    counts_d1 = compute_counts(
        tokenized_docs[0],
        term_to_index
    )

    assert counts_d1 == [
        1, 0, 1, 1, 0
    ]

    counts_d2 = compute_counts(
        tokenized_docs[1],
        term_to_index
    )

    assert counts_d2 == [
        0, 1, 1, 1, 0
    ]

    counts_d3 = compute_counts(
        tokenized_docs[2],
        term_to_index
    )

    assert counts_d3 == [
        1, 0, 0, 1, 1
    ]

    print("test_compute_counts: OK")


# Test compute_tf

def test_compute_tf():
    tokenized_docs = [
        tokenize(document)
        for document in TEST_CORPUS
    ]

    vocabulary, term_to_index = build_vocabulary(
        tokenized_docs
    )

    counts_d1 = compute_counts(
        tokenized_docs[0],
        term_to_index
    )

    tf_d1 = compute_tf(counts_d1)

    assert approx_equal(
        tf_d1[term_to_index["cat"]],
        1 / 3
    )

    assert approx_equal(
        tf_d1[term_to_index["dog"]],
        0.0
    )

    assert approx_equal(
        sum(tf_d1),
        1.0
    )

    print("test_compute_tf: OK")


# Test compute_idf

def test_compute_idf():
    tokenized_docs = [
        tokenize(document)
        for document in TEST_CORPUS
    ]

    vocabulary, term_to_index = build_vocabulary(
        tokenized_docs
    )

    count_matrix = [
        compute_counts(tokens, term_to_index)
        for tokens in tokenized_docs
    ]

    idf = compute_idf(count_matrix)

    assert approx_equal(
        idf[term_to_index["fish"]],
        0.0
    )

    expected_idf_dog = math.log(3)

    assert approx_equal(
        idf[term_to_index["dog"]],
        expected_idf_dog
    )

    assert idf[term_to_index["fish"]] == min(idf)

    print("test_compute_idf: OK")


# Test compute_tfidf

def test_compute_tfidf():
    tokenized_docs = [
        tokenize(document)
        for document in TEST_CORPUS
    ]

    vocabulary, term_to_index = build_vocabulary(
        tokenized_docs
    )

    count_matrix = [
        compute_counts(tokens, term_to_index)
        for tokens in tokenized_docs
    ]

    idf = compute_idf(count_matrix)

    tf_d1 = compute_tf(count_matrix[0])

    tfidf_d1 = compute_tfidf(
        tf_d1,
        idf
    )

    expected_tfidf_cat = (
                                 1 / 3
                         ) * math.log(3 / 2)

    assert approx_equal(
        tfidf_d1[term_to_index["cat"]],
        expected_tfidf_cat
    )

    assert approx_equal(
        tfidf_d1[term_to_index["dog"]],
        0.0
    )

    assert approx_equal(
        tfidf_d1[term_to_index["fish"]],
        0.0
    )

    print("test_compute_tfidf: OK")


# Test cosine_similarity

def test_cosine_similarity():
    vector = [
        1.0,
        2.0,
        3.0
    ]

    assert approx_equal(
        cosine_similarity(vector, vector),
        1.0
    )

    vector_a = [
        1.0,
        0.0
    ]

    vector_b = [
        0.0,
        1.0
    ]

    assert approx_equal(
        cosine_similarity(vector_a, vector_b),
        0.0
    )

    zero_vector = [
        0.0,
        0.0,
        0.0
    ]

    assert cosine_similarity(
        zero_vector,
        vector
    ) == 0.0

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

    expected_similarity = 2 / math.sqrt(6)

    assert approx_equal(
        cosine_similarity(vector_a, vector_b),
        expected_similarity
    )

    print("test_cosine_similarity: OK")


# ============================================================
# CHẠY UNIT TESTS

def run_all_tests():
    print("=" * 60)
    print("CHẠY UNIT TESTS - PART E")

    test_tokenize()
    test_build_vocabulary()
    test_compute_counts()
    test_compute_tf()
    test_compute_idf()
    test_compute_tfidf()
    test_cosine_similarity()

    print()
    print("Tất cả các test đều PASS.")

if __name__ == "__main__":
    run_all_tests()