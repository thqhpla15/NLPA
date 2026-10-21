# Bài tính toán — LAB 05: Sequence Models

## 1. Tính toán bước lan truyền tiến của RNN

### Dữ liệu đầu vào

Vector đầu vào:

`x_t = [1, 0]^T`

Hidden state trước đó:

`h_(t-1) = [0.5, 0.2]^T`

Ma trận trọng số đầu vào:

```text
W_xh = [[0.5, 0.2],
        [0.1, 0.4]]
```

Ma trận trọng số hồi quy:

```text
W_hh = [[0.3, 0.1],
        [0.2, 0.5]]
```

Bias:

```text
b_h = [0, 0]^T
```

Công thức của RNN:

```text
a_t = W_xh * x_t + W_hh * h_(t-1) + b_h

h_t = tanh(a_t)
```

### Bước 1: Tính thành phần từ đầu vào

```text
W_xh * x_t
= [[0.5, 0.2],       [[1],
   [0.1, 0.4]]   *    [0]]
```

Thực hiện phép nhân ma trận:

```text
= [0.5 * 1 + 0.2 * 0,
   0.1 * 1 + 0.4 * 0]

= [0.5, 0.1]^T
```

### Bước 2: Tính thành phần hồi quy

```text
W_hh * h_(t-1)
= [[0.3, 0.1],       [[0.5],
   [0.2, 0.5]]   *    [0.2]]
```

Thực hiện phép nhân ma trận:

```text
= [0.3 * 0.5 + 0.1 * 0.2,
   0.2 * 0.5 + 0.5 * 0.2]

= [0.15 + 0.02,
   0.10 + 0.10]

= [0.17, 0.20]^T
```

### Bước 3: Tính vector trước hàm kích hoạt

Thay các kết quả vào công thức:

```text
a_t = [0.5, 0.1]^T
    + [0.17, 0.20]^T
    + [0, 0]^T
```

Cộng từng thành phần:

```text
a_t = [0.67, 0.30]^T
```

### Bước 4: Tính hidden state mới

Áp dụng hàm tanh cho từng phần tử:

```text
h_t = [tanh(0.67), tanh(0.30)]^T
```

Tính gần đúng:

```text
tanh(0.67) ≈ 0.585
tanh(0.30) ≈ 0.291
```

Do đó:

```text
h_t ≈ [0.585, 0.291]^T
```

**Kết quả:** Hidden state sau khi xử lý đầu vào hiện tại xấp xỉ `[0.585, 0.291]^T`.

---

## 2. Bài tính toán các cổng LSTM

### Dữ liệu đầu vào

- Cổng quên: `f_t = 0.9`
- Cổng đầu vào: `i_t = 0.2`
- Trạng thái ô ứng viên: `c_tilde = 0.5`
- Trạng thái ô trước đó: `c_(t-1) = 0.8`

Công thức cập nhật trạng thái ô:

```text
c_t = f_t * c_(t-1) + i_t * c_tilde
```

### Bước 1: Thay các giá trị vào công thức

```text
c_t = 0.9 * 0.8 + 0.2 * 0.5
```

### Bước 2: Tính từng thành phần

Thành phần thông tin cũ được giữ lại:

```text
f_t * c_(t-1) = 0.9 * 0.8 = 0.72
```

Thành phần thông tin mới được bổ sung:

```text
i_t * c_tilde = 0.2 * 0.5 = 0.10
```

### Bước 3: Tính trạng thái ô mới

```text
c_t = 0.72 + 0.10 = 0.82
```

**Kết quả: `c_t = 0.82`.**

### Giải thích kết quả

Khi cổng quên gần bằng 1, mô hình có xu hướng giữ lại nhiều thông tin từ trạng thái ô trước đó. Khi cổng quên gần bằng 0, mô hình có xu hướng loại bỏ phần lớn thông tin cũ.

Cổng đầu vào kiểm soát lượng thông tin từ trạng thái ô ứng viên được bổ sung vào trạng thái ô hiện tại.
