# Prediction

## 1. Pretrained model và model train from scratch

**Dự đoán:** Pretrained Transformer sau khi fine-tuning có thể đạt kết quả tốt hơn model nhỏ được huấn luyện từ đầu trên IMDb.

**Lý do:** Pretrained model đã học được nhiều mẫu ngôn ngữ từ dữ liệu lớn trước khi thực hiện bài toán phân loại. Khi chuyển sang IMDb, model chỉ cần điều chỉnh những biểu diễn đã học cho phù hợp với việc nhận diện cảm xúc. Ngược lại, model train from scratch phải học cả biểu diễn ngôn ngữ lẫn cách phân loại từ tập dữ liệu IMDb, nên có thể gặp bất lợi khi dữ liệu và tài nguyên huấn luyện hạn chế.

**Điều cần kiểm tra:** So sánh Accuracy và các chỉ số đánh giá trên cùng tập test. Đồng thời quan sát tốc độ hội tụ và số epoch cần thiết của hai model.

## 2. Freeze encoder hay fine-tune toàn bộ model?

**Dự đoán:** Chọn **C — Kém hơn** đối với phương án freeze toàn bộ encoder.

**Lý do:** Khi encoder bị freeze, các biểu diễn mà encoder tạo ra không được điều chỉnh theo dữ liệu IMDb. Classification head phải tự học cách phân loại dựa trên những biểu diễn có sẵn. Khi fine-tuning toàn bộ model, encoder cũng được cập nhật nên có cơ hội học thêm các đặc trưng phù hợp hơn với bài toán sentiment classification.

**Điều cần kiểm tra:** So sánh kết quả trên cùng tập test và cùng điều kiện thực nghiệm. Phương án freeze thường tiết kiệm tài nguyên và thời gian huấn luyện hơn, nhưng có thể đạt chất lượng phân loại thấp hơn. Đây là dự đoán, không phải kết quả chắc chắn.

## 3. Khi chỉ sử dụng 10% training data

**Dự đoán:** Model train from scratch có khả năng giảm performance mạnh hơn pretrained model.

**Lý do:** Pretrained model đã tiếp thu kiến thức ngôn ngữ từ giai đoạn pretraining, nên không phải học mọi biểu diễn từ lượng dữ liệu IMDb còn lại. Khi chỉ có 10% dữ liệu, model train from scratch có ít ví dụ hơn để học cách biểu diễn câu và phân biệt cảm xúc. Pretrained model vẫn có nền tảng biểu diễn đã học trước đó, dù vẫn cần dữ liệu của task để thích nghi.

**Điều cần kiểm tra:** Huấn luyện và đánh giá hai phương pháp với các mức dữ liệu 100%, 50%, 25% và 10%. So sánh Accuracy hoặc F1 trên cùng tập test. Dự đoán là performance của model train from scratch sẽ suy giảm rõ hơn khi lượng dữ liệu giảm, trong khi pretrained model sẽ giữ kết quả ổn định hơn.

---

## Tổng kết

- **Train from scratch:** Khởi tạo model và học biểu diễn từ dữ liệu của bài toán.
- **Feature extraction:** Giữ encoder pretrained cố định, chỉ huấn luyện classification head.
- **Fine-tuning:** Tiếp tục cập nhật trọng số pretrained encoder cùng classification head.

Các dự đoán trên cần được đối chiếu với kết quả thực nghiệm; chúng không thay thế cho số liệu chạy model.
