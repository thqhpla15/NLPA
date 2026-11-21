# Prediction – LAB 09: Named Entity Recognition

## 1. Pretrained Transformer vs Simple Classifier

Dự đoán: Pretrained Transformer sẽ đạt hiệu suất NER tốt hơn classifier đơn giản sử dụng word features hoặc TF-IDF.

**Lý do:**

- Transformer sử dụng contextual representation, giúp biểu diễn mỗi token dựa trên ngữ cảnh của toàn bộ câu.
- Mô hình có khả năng phân biệt các từ có cùng cách viết nhưng mang ý nghĩa hoặc loại thực thể khác nhau trong những ngữ cảnh khác nhau.
- Pretrained Transformer đã được huấn luyện trên lượng lớn văn bản, do đó có thể tận dụng kiến thức ngôn ngữ đã học để nhận diện các thực thể.
- Classifier đơn giản chủ yếu dựa vào các đặc trưng từ vựng, vì vậy khả năng nắm bắt ngữ cảnh và mối quan hệ giữa các token còn hạn chế.

**Kết quả dự đoán:** Pretrained Transformer sẽ có Precision, Recall và F1-score cao hơn baseline, đặc biệt trong những trường hợp cần phân tích ngữ cảnh để xác định loại thực thể.

## 2. Entity Length

Dự đoán: Các entity có nhiều token sẽ khó nhận diện chính xác hơn entity chỉ có một token.

**Lý do:**

- Entity càng dài thì số lượng token cần dự đoán chính xác càng nhiều.
- Mô hình phải xác định đúng cả entity type và vị trí bắt đầu, kết thúc của thực thể.
- Chỉ cần một token bị gán nhãn sai cũng có thể khiến toàn bộ entity không được tính là một dự đoán chính xác.
- Những entity dài có thể chứa các từ thông thường, khiến việc xác định ranh giới thực thể trở nên khó khăn hơn.

Ví dụ:

- `Microsoft`: Chỉ cần nhận diện một token.
- `United Nations Security Council`: Cần xác định chính xác bốn token thuộc cùng một thực thể và gán nhãn BIO phù hợp.

**Kết quả dự đoán:** Entity một token sẽ có khả năng đạt F1-score cao hơn entity dài nhiều token. Tuy nhiên, kết quả thực tế còn phụ thuộc vào tần suất xuất hiện, mức độ quen thuộc và ngữ cảnh của từng entity.

## 3. Rare Entities

Dự đoán: Các entity hiếm xuất hiện trong training data có khả năng bị nhận diện sai nhiều hơn các entity phổ biến.

**Lý do:**

- Mô hình có ít ví dụ để học đặc điểm của những entity hiếm.
- Các đặc trưng từ vựng và cách biểu diễn của entity chưa được học đầy đủ.
- Một số entity hiếm có thể có cách viết không quen thuộc hoặc không xuất hiện trong quá trình pretrained model được huấn luyện.
- Mô hình có thể nhầm lẫn entity hiếm với các từ thông thường hoặc một loại entity khác.

**Kết quả dự đoán:** Rare entities sẽ có Recall và F1-score thấp hơn các entity phổ biến do mô hình khó ghi nhớ và nhận diện chính xác những thực thể ít xuất hiện.

## 4. Tổng kết dự đoán

| Nội dung | Dự đoán |
|---|---|
| Pretrained Transformer vs Baseline | Transformer có F1-score cao hơn |
| Entity một token vs nhiều token | Entity một token dễ nhận diện chính xác hơn |
| Common entities vs Rare entities | Common entities có hiệu suất nhận diện tốt hơn |

## 5. Expected Outcomes

Trước khi thực hiện thực nghiệm, dự kiến:

1. Pretrained Transformer sẽ cải thiện hiệu suất NER nhờ khả năng học contextual representation.
2. Entity có độ dài lớn sẽ gặp nhiều lỗi về entity boundary hơn.
3. Rare entities sẽ có tỷ lệ nhận diện chính xác thấp hơn do thiếu dữ liệu huấn luyện.
4. Context đóng vai trò quan trọng trong việc phân biệt các entity có cùng surface form.
5. Entity-level F1-score có thể thấp hơn token-level F1-score vì mô hình phải nhận diện chính xác toàn bộ thực thể thay vì chỉ dự đoán đúng từng token riêng lẻ.

Các dự đoán trên sẽ được kiểm chứng thông qua kết quả thực nghiệm trên tập CoNLL-2003.
