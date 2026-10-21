# Reflection — LAB 05: Sequence Models

## 1. So sánh RNN, LSTM và GRU

Trong thực nghiệm, cả ba mô hình đều được sử dụng để phân loại cảm xúc của các review phim trên bộ dữ liệu IMDb.

Sau 3 epoch, LSTM đạt Test Accuracy cao nhất với 82,19%, tiếp theo là GRU với 81,16%. RNN đạt 63,68%.

Kết quả này cho thấy LSTM và GRU hoạt động tốt hơn RNN trong lần thực nghiệm hiện tại. Tuy nhiên, kết quả còn phụ thuộc vào cách tiền xử lý dữ liệu, tham số mô hình và quá trình huấn luyện.

## 2. Vai trò của các mô hình tuần tự

RNN xử lý dữ liệu theo từng bước tuần tự nhưng có thể gặp khó khăn khi cần ghi nhớ thông tin từ những bước trước đó.

LSTM sử dụng các cổng (gates) để điều chỉnh thông tin được lưu giữ hoặc loại bỏ, giúp mô hình xử lý các phụ thuộc dài hạn tốt hơn.

GRU cũng sử dụng cơ chế cổng nhưng có cấu trúc đơn giản hơn LSTM. Trong thực nghiệm này, GRU đạt kết quả gần với LSTM, nhưng thời gian huấn luyện được ghi nhận dài hơn đáng kể.

## 3. Hạn chế của thực nghiệm

- Các mô hình mới được huấn luyện trong 3 epoch nên chưa chắc đã đạt kết quả tối ưu.
- Việc sử dụng `test_loader` sau mỗi epoch khiến tập test được dùng để theo dõi quá trình huấn luyện, làm giảm tính độc lập của đánh giá cuối cùng.
- Các review được giới hạn ở 200 token, vì vậy một phần thông tin trong review dài có thể bị loại bỏ.
- Thời gian huấn luyện có thể thay đổi tùy thuộc vào thiết bị và môi trường thực thi.

## 4. Bài học rút ra

Qua bài thực hành, tôi hiểu rõ hơn cách xây dựng mô hình RNN, LSTM và GRU cho bài toán phân loại chuỗi văn bản. Tôi cũng thực hành được quy trình tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá bằng các chỉ số và phân tích những trường hợp dự đoán sai.

Kết quả thực nghiệm giúp tôi nhận ra rằng việc lựa chọn kiến trúc mô hình là quan trọng, nhưng cách tổ chức quá trình đánh giá và phân tích lỗi cũng ảnh hưởng lớn đến độ tin cậy của kết luận.
