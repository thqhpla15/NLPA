# LAB 06 - Calculation

## 1. Tính Attention Scores

Cho query `q = [1, 2]` và các key:

- `k1 = [1, 0]`
- `k2 = [0, 1]`
- `k3 = [1, 1]`

Công thức tính attention score:

`s(i) = q^T * k(i)`

Tính từng score:

- `s1 = 1*1 + 2*0 = 1`
- `s2 = 1*0 + 2*1 = 2`
- `s3 = 1*1 + 2*1 = 3`

Kết quả: `s = [1, 2, 3]`

## 2. Tính Softmax

Cho các attention scores: `s = [1, 2, 3]`.

Công thức Softmax:

`alpha(i) = exp(s(i)) / sum(exp(s(j)))`

Tính giá trị hàm mũ:

- `exp(1) ≈ 2.7183`
- `exp(2) ≈ 7.3891`
- `exp(3) ≈ 20.0855`

Tổng:

`Z = exp(1) + exp(2) + exp(3) ≈ 30.1929`

Tính các attention weights:

- `alpha1 = exp(1)/Z ≈ 0.0900`
- `alpha2 = exp(2)/Z ≈ 0.2447`
- `alpha3 = exp(3)/Z ≈ 0.6652`

Kiểm tra tổng:

`0.0900 + 0.2447 + 0.6652 ≈ 1`

Kết quả: `alpha = [0.0900, 0.2447, 0.6652]`

Score càng lớn thì attention weight càng lớn vì hàm số mũ tăng theo giá trị đầu vào.

## 3. Tính Context Vector

Cho các attention weights:

`alpha = [0.1, 0.2, 0.7]`

Các value vectors:

- `v1 = [1, 0]`
- `v2 = [0, 1]`
- `v3 = [2, 2]`

Công thức tính context vector:

`c = sum(alpha(i) * v(i))`

Thay số:

`c = 0.1*[1, 0] + 0.2*[0, 1] + 0.7*[2, 2]`

Tính thành phần thứ nhất:

`c1 = 0.1*1 + 0.2*0 + 0.7*2 = 1.5`

Tính thành phần thứ hai:

`c2 = 0.1*0 + 0.2*1 + 0.7*2 = 1.6`

Kết quả: `c = [1.5, 1.6]`

Vector `v3` đóng góp nhiều nhất vì có attention weight lớn nhất, bằng `0.7`.

## 4. Tính toán Attention hoàn chỉnh

Cho query:

`q = [1, 0]`

Các key vectors:

- `k1 = [1, 0]`
- `k2 = [0, 1]`
- `k3 = [1, 1]`

Các value vectors:

- `v1 = [1, 0]`
- `v2 = [0, 2]`
- `v3 = [2, 1]`

### Bước 1: Tính Attention Scores

- `s1 = 1*1 + 0*0 = 1`
- `s2 = 1*0 + 0*1 = 0`
- `s3 = 1*1 + 0*1 = 1`

Kết quả: `s = [1, 0, 1]`

### Bước 2: Tính Attention Weights

Công thức Softmax:

`alpha(i) = exp(s(i)) / sum(exp(s(j)))`

Tính tổng:

`Z = exp(1) + exp(0) + exp(1) ≈ 6.4366`

Các attention weights:

- `alpha1 = exp(1)/Z ≈ 0.4223`
- `alpha2 = exp(0)/Z ≈ 0.1554`
- `alpha3 = exp(1)/Z ≈ 0.4223`

Kết quả: `alpha ≈ [0.4223, 0.1554, 0.4223]`

### Bước 3: Tính Context Vector

Công thức:

`c = alpha1*v1 + alpha2*v2 + alpha3*v3`

Thay các value vectors:

`c = 0.4223*[1, 0] + 0.1554*[0, 2] + 0.4223*[2, 1]`

Thành phần thứ nhất:

`c1 = 0.4223 + 0 + 0.8446 ≈ 1.2669`

Thành phần thứ hai:

`c2 = 0 + 0.3108 + 0.4223 ≈ 0.7331`

Kết quả cuối cùng:

`c ≈ [1.2669, 0.7331]`

Hai vector `v1` và `v3` có attention weights bằng nhau nên đóng góp đáng kể và ngang nhau về trọng số vào context vector.
