# LAB 06 - Prediction

## 1. Dự đoán Attention Weight

Nếu một key có tích vô hướng với query lớn hơn các key còn lại, key đó sẽ nhận attention weight lớn nhất sau khi áp dụng Softmax.

Điều này xảy ra vì hàm số mũ tăng theo giá trị đầu vào, khiến score lớn hơn tạo ra weight lớn hơn.

## 2. Các Scores bằng nhau

Cho attention scores:

`s = [1, 1, 1, 1]`

Vì tất cả scores bằng nhau nên Softmax phân phối attention weights đồng đều cho bốn phần tử.

Kết quả:

`alpha = [0.25, 0.25, 0.25, 0.25]`

Mỗi phần tử nhận 25% tổng attention weight.

## 3. Một Score lớn vượt trội

Cho attention scores:

`s = [1, 1, 10]`

Score thứ ba lớn hơn rất nhiều so với hai score còn lại. Vì vậy, sau khi áp dụng Softmax, phần tử thứ ba sẽ nhận gần như toàn bộ attention weight.

Kết quả xấp xỉ:

`alpha = [0.0001, 0.0001, 0.9998]`

Do đó, Attention tập trung gần như hoàn toàn vào phần tử thứ ba.

## 4. Ảnh hưởng của độ dài chuỗi

Giả sử độ dài chuỗi tăng từ 10 token lên 1000 token.

Với cơ chế Full Self-Attention, số phần tử trong ma trận attention tỷ lệ với bình phương độ dài chuỗi.

- Chuỗi 10 token: `10*10 = 100` phần tử.
- Chuỗi 1000 token: `1000*1000 = 1,000,000` phần tử.

Tỷ lệ tăng:

`1,000,000 / 100 = 10,000`

Vì vậy, số phần tử trong ma trận attention tăng gấp 10.000 lần.

Độ phức tạp tính toán của phép nhân ma trận query-key là `O(T^2 * d)`, trong đó `T` là độ dài chuỗi và `d` là chiều của vector key.

Điều này giải thích vì sao Full Self-Attention tốn nhiều tài nguyên khi xử lý chuỗi dài.

## 5. Attention có luôn chọn từ quan trọng nhất không?

Không. Attention không phải lúc nào cũng tập trung vào từ mà con người cho là quan trọng nhất.

Attention weights được tính dựa trên query và key trong mô hình. Chúng phản ánh mức độ liên quan giữa các biểu diễn theo cách mô hình học được, chứ không nhất thiết trùng với đánh giá của con người.

Ngoài ra, một từ có thể quan trọng đối với con người nhưng nhận attention weight thấp trong một head hoặc tại một lớp cụ thể.

Vì vậy, Attention là cơ chế giúp mô hình tập trung thông tin, nhưng không phải thước đo hoàn hảo về mức độ quan trọng của từ.
