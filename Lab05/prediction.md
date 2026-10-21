# Prediction — LAB 05: Sequence Models

## Prediction 1: TF-IDF + Logistic Regression và RNN + Embedding

**Dự đoán:** RNN + Embedding có khả năng mô hình hóa trực tiếp thứ tự từ tốt hơn.

**Lý do:** TF-IDF biểu diễn văn bản dựa trên các đặc trưng từ mà không trực tiếp giữ lại thứ tự xuất hiện của chúng. Trong khi đó, RNN xử lý từng token theo trình tự và cập nhật hidden state dựa trên đầu vào hiện tại cùng hidden state trước đó. Vì vậy, các câu có thứ tự từ khác nhau có thể tạo ra những biểu diễn hidden state khác nhau.

**Mức độ tự tin:** Cao.

---

## Prediction 2: RNN và LSTM trong việc xử lý phụ thuộc dài hạn

**Dự đoán:** LSTM có khả năng xử lý các phụ thuộc dài hạn hiệu quả hơn RNN cơ bản.

**Lý do:** RNN cơ bản có thể gặp hiện tượng vanishing gradient khi huấn luyện trên các chuỗi dài, khiến việc học thông tin từ những token ở xa trở nên khó khăn. LSTM sử dụng cell state và các cổng để kiểm soát thông tin được giữ lại, bổ sung hoặc đưa ra ngoài. Cơ chế này có thể giúp mô hình duy trì thông tin quan trọng qua khoảng cách dài hơn.

**Mức độ tự tin:** Cao.

---

## Prediction 3: Tăng sequence length từ 50 lên 200

**Dự đoán:** Khi tăng sequence length từ 50 lên 200, thời gian huấn luyện RNN có thể tăng.

**Lý do:** Sequence dài hơn yêu cầu mô hình thực hiện nhiều phép tính hồi quy qua các timestep. Quá trình lan truyền ngược cũng cần xử lý nhiều bước hơn, có thể làm tăng thời gian tính toán và bộ nhớ sử dụng. Mức độ ảnh hưởng thực tế phụ thuộc vào cách triển khai, batch size và phần cứng.

**Mức độ tự tin:** Cao.

---

## Prediction 4: Cắt ngắn mọi review còn 30 token

**Dự đoán:** Cắt ngắn mọi review còn 30 token có thể làm giảm hiệu quả phân loại cảm xúc.

**Lý do:** Một số review chứa thông tin quan trọng ở phía sau 30 token đầu tiên. Việc cắt ngắn có thể làm mất từ phủ định, nội dung thể hiện sự tương phản hoặc nhận xét cuối cùng của người đánh giá. Tuy nhiên, các review ngắn hoặc có từ thể hiện cảm xúc rõ ràng ở đầu câu vẫn có thể được phân loại chính xác.

**Mức độ tự tin:** Cao.

---

## Prediction 5: LSTM có luôn đạt accuracy cao hơn RNN không?

**Dự đoán:** Không. LSTM không nhất thiết đạt accuracy cao hơn RNN trên mọi tập dữ liệu.

**Lý do:** LSTM có cơ chế hỗ trợ xử lý phụ thuộc dài hạn, nhưng kết quả còn phụ thuộc vào đặc điểm tập dữ liệu, cấu hình mô hình, siêu tham số và quá trình huấn luyện. RNN đơn giản hơn vẫn có thể đạt kết quả tương đương hoặc tốt hơn trên một số tập dữ liệu, đặc biệt khi các chuỗi tương đối ngắn và không đòi hỏi nhiều thông tin dài hạn.

**Mức độ tự tin:** Cao.
