# Phân tích lỗi — LAB 05: Sequence Models

## 1. Tổng quan

Mô hình LSTM dự đoán sai 4.453 trên tổng số 25.000 review trong tập test, tương ứng với tỷ lệ lỗi khoảng 17,81%. Độ chính xác đạt 82,19%.

## 2. Các dạng lỗi quan sát được

### 2.1. Cảm xúc trái chiều (Mixed Sentiment)

Một số review vừa thể hiện ý kiến tích cực vừa thể hiện ý kiến tiêu cực. Ví dụ, người viết có thể khen diễn xuất của diễn viên nhưng đồng thời chê kịch bản hoặc chất lượng tổng thể của bộ phim.

### 2.2. Từ ngữ tích cực trong review tiêu cực

Một số review có các từ mang sắc thái tích cực như "great", "excellent" hoặc "pleasure", nhưng kết luận chung vẫn là tiêu cực. Những từ này có thể khiến mô hình dự đoán nhầm nhãn Positive.

### 2.3. Review dài và nhiều chi tiết

Các review dài thường đề cập đến nhiều khía cạnh của bộ phim như diễn xuất, cốt truyện và kịch bản. Mô hình có thể gặp khó khăn khi tổng hợp các thông tin này để xác định cảm xúc chung của toàn bộ review.

### 2.4. Phủ định và ngữ cảnh (Negation and Context)

Một số cách diễn đạt, chẳng hạn "don't watch it", cần được hiểu theo ngữ cảnh. Nếu mô hình phụ thuộc nhiều vào các từ riêng lẻ mà chưa nắm bắt tốt ý nghĩa của cả câu, kết quả phân loại có thể không chính xác.

## 3. Hạn chế của phân tích

Các nhận xét trên được rút ra từ một số review bị phân loại sai, chưa đại diện cho toàn bộ 4.453 trường hợp. Cần kiểm tra thêm dữ liệu để xác định mức độ phổ biến của từng dạng lỗi.

## 4. Kết luận

Các ví dụ cho thấy bài toán phân loại cảm xúc không chỉ phụ thuộc vào việc nhận diện những từ tích cực và tiêu cực. Ngữ cảnh, câu phủ định, cảm xúc trái chiều và mối liên hệ giữa các phần trong review đều có thể ảnh hưởng đến kết quả dự đoán.
