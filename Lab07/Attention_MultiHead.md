# LAB 07 – From Attention to Transformer

# Bài 6. Bài tập tính toán – Một Attention Head

## 1. Đề bài

Cho ma trận đầu vào:

```math
X \in \mathbb{R}^{3 \times 4}
```

Một attention head sử dụng:

```math
d_k = 2
```

Các ma trận trọng số:

```math
W_Q, W_K, W_V \in \mathbb{R}^{4 \times 2}
```

Yêu cầu tính:

```math
Q = XW_Q
```

```math
K = XW_K
```

```math
V = XW_V
```

Tiếp theo:

```math
S = \frac{QK^T}{\sqrt{2}}
```

```math
A = \text{softmax}(S)
```

Cuối cùng:

```math
Z = AV
```

## 2. Kiểm tra kích thước ma trận

Dựa vào quy tắc nhân ma trận: nếu ma trận A có kích thước $`m \times n`$, ma trận B có kích thước $`n \times p`$ thì tích AB có kích thước $`m \times p`$.

| Phép tính | Kích thước | Kết quả |
|---|---|---|
| $`X`$ | $`3 \times 4`$ | 3 token, mỗi token có vector 4 chiều |
| $`W_Q, W_K, W_V`$ | $`4 \times 2`$ | Chiếu từ 4 chiều xuống 2 chiều |
| $`Q = XW_Q`$ | $`(3 \times 4)(4 \times 2)`$ | $`3 \times 2`$ |
| $`K = XW_K`$ | $`(3 \times 4)(4 \times 2)`$ | $`3 \times 2`$ |
| $`V = XW_V`$ | $`(3 \times 4)(4 \times 2)`$ | $`3 \times 2`$ |
| $`K^T`$ | Chuyển vị của $`3 \times 2`$ | $`2 \times 3`$ |
| $`QK^T`$ | $`(3 \times 2)(2 \times 3)`$ | $`3 \times 3`$ |
| $`S = QK^T/\sqrt{2}`$ | $`3 \times 3`$ | Ma trận điểm attention |
| $`A = \text{softmax}(S)`$ | $`3 \times 3`$ | Ma trận trọng số attention |
| $`Z = AV`$ | $`(3 \times 3)(3 \times 2)`$ | $`3 \times 2`$ |

**Kết luận:**

Đầu vào có 3 token. Mỗi token được chiếu thành Query, Key và Value có 2 chiều. Attention tạo ra ma trận trọng số $`3 \times 3`$, sau đó kết hợp với $`V`$ để tạo đầu ra $`Z`$ có kích thước $`3 \times 2`$.

## 3. Ý nghĩa từng bước

### Bước 1 – Tạo Query, Key và Value

```math
Q = XW_Q
```

```math
K = XW_K
```

```math
V = XW_V
```

Ba ma trận trọng số $`W_Q`$, $`W_K`$, $`W_V`$ là các phép chiếu được học trong quá trình huấn luyện. Chúng biến đổi biểu diễn đầu vào $`X`$ thành ba không gian đặc trưng khác nhau:

- **Query (Q):** Thông tin mà mỗi token sử dụng để tìm kiếm những token liên quan.
- **Key (K):** Thông tin dùng để so khớp với Query.
- **Value (V):** Thông tin được tổng hợp theo trọng số attention.

### Bước 2 – Tính điểm Attention

```math
S = \frac{QK^T}{\sqrt{d_k}}
```

Vì $`d_k = 2`$, ta có:

```math
S = \frac{QK^T}{\sqrt{2}}
```

Phép nhân $`QK^T`$ tính độ tương thích giữa Query của từng token với Key của tất cả token.

Chia cho $`\sqrt{d_k}`$ giúp điều chỉnh độ lớn của điểm số trước khi đưa qua softmax, tránh các giá trị quá lớn làm softmax trở nên quá tập trung.

### Bước 3 – Tính Attention Weights

```math
A = \text{softmax}(S)
```

Softmax được áp dụng theo từng hàng của ma trận $`S`$.

Mỗi hàng của $`A`$ biểu diễn mức độ một token chú ý đến các token trong sequence. Tổng các phần tử trên mỗi hàng bằng 1.

### Bước 4 – Tạo Context Vector

```math
Z = AV
```

Mỗi hàng của $`Z`$ là tổng có trọng số của các vector Value. Trọng số được lấy từ ma trận attention $`A`$.

Kết quả cuối cùng:

```math
Z \in \mathbb{R}^{3 \times 2}
```

## 4. Lưu ý về kết quả tính toán

Đề bài hiện chỉ cung cấp kích thước của các ma trận $`X`$, $`W_Q`$, $`W_K`$, $`W_V`$, chưa cung cấp giá trị cụ thể của từng phần tử bên trong các ma trận.

Do đó, ta có thể xác định chính xác kích thước và quy trình tính toán, nhưng chưa thể tính ra các giá trị số cụ thể của $`Q`$, $`K`$, $`V`$, $`S`$, $`A`$, $`Z`$.

Muốn tính kết quả số, cần có các ma trận đầu vào với những phần tử cụ thể.

## 5. Kết luận bài 6

Một attention head trong Transformer sử dụng cơ chế Scaled Dot-Product Attention đã học ở LAB 06.

Điểm khác biệt là đầu vào được chiếu qua ba ma trận trọng số $`W_Q`$, $`W_K`$, $`W_V`$ để tạo Query, Key và Value trước khi tính attention.

Như vậy, một attention head chính là cơ chế attention của LAB 06 kết hợp với các projection matrices.

---

# Bài 7. Multi-Head Attention

## 1. Đề bài

Giả sử có:

- 2 attention heads.
- Mỗi head tạo ra vector có dimension bằng 2.
- Có $`T`$ token trong sequence.

Khi đó:

```math
\text{head}_1 \in \mathbb{R}^{T \times 2}
```

```math
\text{head}_2 \in \mathbb{R}^{T \times 2}
```

Sau khi concatenation:

```math
\text{Concat}(\text{head}_1, \text{head}_2) \in \mathbb{R}^{T \times 4}
```

**Câu hỏi:** Vì sao cần concatenation trước khi áp dụng $`W^O`$?

## 2. Trả lời

Cần concatenation trước khi áp dụng $`W^O`$ vì mỗi attention head học và tạo ra một biểu diễn đặc trưng riêng.

Concatenation ghép các biểu diễn này lại theo chiều đặc trưng, giúp giữ thông tin từ tất cả các heads trong một vector chung.

Cụ thể:

```math
\text{head}_1 \in \mathbb{R}^{T \times 2}
```

```math
\text{head}_2 \in \mathbb{R}^{T \times 2}
```

Ghép theo chiều cuối (feature dimension):

```math
H = \text{Concat}(\text{head}_1, \text{head}_2)
```

Kết quả:

```math
H \in \mathbb{R}^{T \times 4}
```

Sau đó áp dụng phép chiếu đầu ra:

```math
\text{MultiHead}(Q,K,V) = HW^O
```

Trong trường hợp này, nếu muốn đầu ra có dimension bằng 4 thì:

```math
W^O \in \mathbb{R}^{4 \times 4}
```

Kích thước phép nhân:

```math
(T \times 4)(4 \times 4) = T \times 4
```

Ma trận $`W^O`$ học cách kết hợp và biến đổi thông tin từ các heads thành một biểu diễn chung.

Nhờ đó, mô hình có thể tận dụng đồng thời nhiều kiểu quan hệ mà các attention heads đã học.

## 3. Vì sao không áp dụng $`W^O`$ riêng cho từng head ngay từ đầu?

Nếu xử lý riêng từng head, mô hình chưa có một biểu diễn chung chứa thông tin từ tất cả các heads tại cùng một vị trí.

Concatenation tạo ra biểu diễn chung trước, để $`W^O`$ có thể học cách kết hợp thông tin giữa các heads.

Điều này giúp mô hình khai thác được nhiều loại đặc trưng khác nhau thay vì chỉ sử dụng kết quả từ một attention head riêng lẻ.

## 4. Kết luận bài 7

Concatenation giúp ghép đầu ra của các attention heads theo chiều đặc trưng.

Sau đó, ma trận $`W^O`$ thực hiện phép chiếu tuyến tính để kết hợp thông tin từ các heads thành đầu ra của Multi-Head Attention.

Nhờ Multi-Head Attention, Transformer có thể học nhiều kiểu quan hệ giữa các token song song trong cùng một sequence.

---

# Tổng kết kiến thức cần nhớ

| Thành phần | Vai trò |
|---|---|
| $`W_Q, W_K, W_V`$ | Chiếu đầu vào thành Query, Key, Value |
| $`QK^T`$ | Tính độ tương thích giữa Query và Key |
| $`\sqrt{d_k}`$ | Điều chỉnh độ lớn điểm attention |
| Softmax | Chuyển điểm số thành attention weights |
| $`AV`$ | Tổng hợp Value theo attention weights |
| Multi-Head Attention | Chạy nhiều attention heads song song |
| Concatenation | Ghép đầu ra các heads theo chiều đặc trưng |
| $`W^O`$ | Kết hợp và chiếu biểu diễn đã ghép thành đầu ra chung |

## Ý chính của LAB 07

Transformer mở rộng self-attention bằng cách sử dụng nhiều attention heads và kết hợp chúng trong một kiến trúc hoàn chỉnh.

- Một attention head là Scaled Dot-Product Attention với các ma trận chiếu $`W_Q`$, $`W_K`$, $`W_V`$.
- Multi-Head Attention chạy nhiều attention heads song song.
- Concatenation ghép đầu ra của các heads.
- Ma trận $`W^O`$ kết hợp thông tin từ các heads thành một biểu diễn chung.

Đây là bước quan trọng để chuyển từ cơ chế Attention của LAB 06 sang kiến trúc Transformer trong LAB 07.
