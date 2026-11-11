# Individual Learning Check

## 1. Pretraining khác fine-tuning ở đâu?

Pretraining là giai đoạn model học biểu diễn ngôn ngữ từ lượng văn bản lớn, thường qua nhiệm vụ tự giám sát như Masked Language Modeling. Fine-tuning là giai đoạn tiếp tục huấn luyện model đã pretrained trên dữ liệu của một nhiệm vụ cụ thể, ví dụ phân loại cảm xúc IMDb.

## 2. MLM là gì?

Masked Language Modeling là nhiệm vụ che một số token trong câu và yêu cầu model dự đoán token ban đầu dựa trên ngữ cảnh. Nhiệm vụ này giúp BERT học biểu diễn có xét đến ngữ cảnh hai chiều.

## 3. Tại sao cần tokenizer?

Tokenizer chuyển văn bản thành token rồi ánh xạ token thành ID trong vocabulary để model có thể xử lý bằng embedding layer.

## 4. Subword token giải quyết vấn đề gì?

Subword giúp xử lý từ hiếm hoặc từ chưa xuất hiện nguyên dạng trong vocabulary bằng cách chia từ thành đơn vị nhỏ hơn, giảm nhu cầu lưu một token riêng cho mọi từ.

## 5. [CLS] thường được sử dụng như thế nào?

`[CLS]` được đặt ở đầu sequence. Vector tại vị trí này thường được đưa vào classification head để dự đoán nhãn cho toàn sequence.

## 6. Attention mask dùng để làm gì?

Attention mask cho model biết vị trí nào là token thật và vị trí nào là padding. Thông thường 1 là token hợp lệ, 0 là padding.

## 7. Frozen encoder nghĩa là gì?

Encoder có trọng số không được cập nhật trong quá trình huấn luyện downstream. Thường chỉ classification head được học.

## 8. Fine-tuning thay đổi những tham số nào?

Trong full fine-tuning, tham số của pretrained encoder và classification head đều có thể được cập nhật.

## 9. Vì sao pretrained model hữu ích khi downstream dataset nhỏ?

Model đã học biểu diễn ngôn ngữ từ dữ liệu lớn trước đó. Dataset downstream chủ yếu giúp model thích nghi với nhiệm vụ mới thay vì phải học mọi kiến thức ngôn ngữ từ đầu.

## 10. Pretrained model có đảm bảo không bị domain shift không?

Không. Dữ liệu mới có thể khác về từ vựng, phong cách và ngữ cảnh so với dữ liệu pretraining hoặc fine-tuning, khiến performance giảm.

## 11. Vì sao tăng sequence length làm chi phí attention tăng?

Self-attention tính tương tác giữa các cặp vị trí. Với độ dài T, ma trận attention có kích thước T × T nên phần tính toán tăng xấp xỉ O(T²).

## 12. Confidence cao có đảm bảo prediction đúng không?

Không. Confidence là xác suất model gán cho nhãn dự đoán, không phải bảo đảm nhãn đó đúng. Model vẫn có thể rất tự tin nhưng dự đoán sai.
