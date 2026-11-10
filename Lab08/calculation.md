# Calculation

## 1. MLM Loss

Trong ví dụ này, token cần dự đoán là `eats`. Model gán xác suất cho token này là `0.70`.

Dùng công thức:

\[
L = -\log(P(\text{token đúng}))
\]

Thay số:

\[
L = -\log(0.70) \approx 0.3567
\]

Nếu xác suất dự đoán `eats` thay đổi:

| Xác suất của `eats` | Loss |
|---:|---:|
| 0.95 | 0.0513 |
| 0.70 | 0.3567 |
| 0.01 | 4.6052 |

Có thể thấy xác suất dành cho token đúng càng cao thì loss càng thấp. Nếu model gán xác suất rất thấp cho đáp án đúng, loss sẽ tăng mạnh.

Cross-Entropy phù hợp với MLM vì tại mỗi vị trí bị che, model cần chọn token đúng trong toàn bộ vocabulary. Hàm loss đánh giá mức độ model đặt xác suất vào token thật: dự đoán càng tốt thì loss càng nhỏ.

## 2. Tokenization

Câu được đưa vào tokenizer:

```text
The cats are playing.
```

Tokenization phụ thuộc vào tokenizer và vocabulary được sử dụng. Vì vậy, cần lấy token và ID trực tiếp từ tokenizer của thí nghiệm, thay vì tự suy đoán.

Ví dụ minh họa với `bert-base-uncased`, nếu các từ không bị tách thành subword:

```text
[CLS] the cats are playing . [SEP]
```

- ` [CLS] `: token đặc biệt thường được dùng làm biểu diễn đầu sequence trong bài toán classification.
- ` [SEP] `: đánh dấu ranh giới hoặc kết thúc sequence.
- `the`, `cats`, `are`, `playing`, `.`: các token nội dung.

Nếu không thêm padding, attention mask tương ứng là:

```text
[1, 1, 1, 1, 1, 1, 1]
```

Các vị trí đều là token hợp lệ nên có giá trị `1`. Nếu sequence được padding thêm, các vị trí padding sẽ có giá trị `0`.

Với cách tách minh họa ở trên:

| Thành phần | Số lượng |
|---|---:|
| Từ (không tính dấu câu) | 4 |
| Token nội dung (có dấu câu) | 5 |
| Special tokens | 2 |
| Tổng token | 7 |

**Lưu ý:** Bảng đếm trên chỉ minh họa trường hợp mỗi từ được giữ thành một token. Kết quả thực tế, bao gồm token IDs, phải lấy từ tokenizer được cung cấp trong LAB. Một từ có thể được tách thành nhiều subword.
