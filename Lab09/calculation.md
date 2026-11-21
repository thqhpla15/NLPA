# Calculation – LAB 09: Named Entity Recognition

## 1. BIO Tagging

Các từ và nhãn BIO tương ứng:

| Token | BIO Label | Giải thích |
|---|---|---|
| Elon | B-PER | Bắt đầu một thực thể chỉ người |
| Musk | I-PER | Tiếp tục thực thể chỉ người |
| founded | O | Không thuộc thực thể |
| SpaceX | B-ORG | Bắt đầu một thực thể chỉ tổ chức |
| in | O | Không thuộc thực thể |
| California | B-LOC | Bắt đầu một thực thể chỉ địa điểm |
| . | O | Không thuộc thực thể |

**Kết quả:**

```text
Elon        → B-PER
Musk        → I-PER
founded     → O
SpaceX      → B-ORG
in          → O
California  → B-LOC
.           → O
```

## 2. Entity Boundary

Các token và nhãn BIO tương ứng:

| Token | BIO Label |
|---|---|
| University | B-ORG |
| of | I-ORG |
| California | I-ORG |
| Berkeley | I-ORG |

**Giải thích:**

BIO tagging không chỉ xác định loại thực thể mà còn xác định vị trí bắt đầu và kết thúc của thực thể đó.

- `B-ORG`: Đánh dấu token đầu tiên của một thực thể thuộc loại Organization.
- `I-ORG`: Đánh dấu các token tiếp theo nằm trong cùng thực thể.
- `O`: Đánh dấu token không thuộc thực thể nào.

Nếu gán tất cả token là `ORG`, mô hình chỉ biết các token thuộc loại tổ chức nhưng không xác định được ranh giới của từng thực thể. Vì vậy, cần phân biệt `B-ORG` và `I-ORG` để biểu diễn chính xác entity boundary.

## 3. Shape Calculation

Các thông số:

- Batch size (B) = 8
- Sequence length (T) = 128
- Hidden dimension (d) = 768
- Number of labels (C) = 9

### 3.1. Hidden States

Hidden states được tạo ra từ pretrained Transformer.

Công thức:

```text
H ∈ R^(B × T × d)
```

Thay số:

```text
H = (8, 128, 768)
```

Mỗi token được biểu diễn bằng một vector có 768 chiều.

### 3.2. Logits

Classification head sử dụng một Linear layer để chuyển hidden representation thành điểm dự đoán cho từng nhãn.

Công thức:

```text
Z = HW + b
```

Trong đó:
- H: Hidden states.
- W: Trọng số của Linear layer.
- b: Bias.
- Z: Logits.

Shape:

```text
Z ∈ R^(B × T × C)

Z = (8, 128, 9)
```

Mỗi token có 9 giá trị logits tương ứng với 9 nhãn NER.

### 3.3. Labels

Mỗi token có một nhãn tương ứng.

Shape:

```text
Y ∈ R^(B × T)

Y = (8, 128)
```

Mỗi vị trí trong labels chứa một label ID tương ứng với token tại cùng vị trí.

### 3.4. Tổng hợp

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| Hidden States | (8, 128, 768) | Biểu diễn của từng token |
| Logits | (8, 128, 9) | Điểm dự đoán cho 9 nhãn |
| Labels | (8, 128) | Nhãn thực tế của từng token |

## 4. Tokenization Ratio

Tỷ lệ token được tạo ra sau khi tokenizer xử lý văn bản được tính bằng:

```text
R = Number of subword tokens / Number of words
```

Trong đó:
- Number of subword tokens: Tổng số token sau khi tokenizer xử lý.
- Number of words: Tổng số từ ban đầu.

**Ý nghĩa:**

- R gần 1: Phần lớn từ được giữ nguyên thành một token.
- R lớn hơn 1: Một số từ được chia thành nhiều subword.
- R càng lớn: Mức độ phân tách từ thành subword càng cao.

Để so sánh chính xác giữa các nhóm câu, cần tính R riêng cho câu ngắn, câu dài, entity và non-entity.

## 5. Precision, Recall và F1-score

Trong NER, các metric cần được tính ở entity level để đánh giá khả năng nhận diện toàn bộ thực thể.

### 5.1. Precision

```text
Precision = TP / (TP + FP)
```

Đo tỷ lệ thực thể được dự đoán là đúng trên tổng số thực thể mà mô hình dự đoán.

### 5.2. Recall

```text
Recall = TP / (TP + FN)
```

Đo tỷ lệ thực thể thực tế được mô hình nhận diện chính xác.

### 5.3. F1-score

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

F1-score là trung bình điều hòa của Precision và Recall.

Trong đó:
- TP (True Positive): Thực thể được dự đoán chính xác cả loại và ranh giới.
- FP (False Positive): Thực thể được dự đoán nhưng không khớp với thực thể thực tế.
- FN (False Negative): Thực thể thực tế nhưng mô hình không nhận diện chính xác.

### 5.4. Entity-level Evaluation

Ví dụ:

```text
Gold:
New York → B-LOC I-LOC

Prediction:
New York → B-LOC O
```

Mặc dù token `New` được dự đoán đúng, toàn bộ thực thể `New York` không được nhận diện hoàn chỉnh.

Do đó, kết quả này không được tính là một True Positive ở entity level.

**Kết luận:** Token-level metrics và entity-level metrics có cách đánh giá khác nhau. Entity-level evaluation phù hợp hơn để đánh giá khả năng nhận diện thực thể hoàn chỉnh của mô hình NER.
