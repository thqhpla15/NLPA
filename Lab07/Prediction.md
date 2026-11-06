# LAB 07 – From Attention to Transformer

# 8. Prediction trước Experiment

## Prediction 1

**Câu hỏi:** Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?

**Prediction:** Có. Transformer có khả năng xử lý các token trong sequence song song hơn RNN.

**Lí do:** RNN phải xử lý từng token theo thứ tự, trong đó hidden state tại timestep hiện tại phụ thuộc vào timestep trước đó. Ngược lại, Transformer sử dụng self-attention để tính toán mối quan hệ giữa các token cùng lúc, không cần chờ kết quả từ token trước.

---

## Prediction 2

**Câu hỏi:** Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?

**Prediction:** Không. Transformer sẽ mất thông tin về vị trí và thứ tự ban đầu của các token.

**Lí do:** Self-attention tính toán mức độ liên quan giữa các token nhưng không tự nhận biết vị trí của chúng trong sequence. Positional encoding được thêm vào embedding để cung cấp thông tin về vị trí. Nếu loại bỏ thành phần này, mô hình khó phân biệt thứ tự các token.


---

## Prediction 3

**Câu hỏi:** Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?

**Prediction:** Không, Transformer không có positional information sẽ khó phân biệt hai câu chỉ khác nhau về thứ tự token.

**Lí do:** Self-attention không có positional information có tính chất permutation equivariant. Khi thứ tự đầu vào thay đổi, các biểu diễn đầu ra cũng thay đổi theo cùng phép hoán vị, nhưng mô hình không có thông tin để xác định vị trí ban đầu của từng token. Vì vậy, nếu sử dụng một phương pháp tổng hợp không phụ thuộc thứ tự, hai câu có thể tạo ra cùng biểu diễn cuối cùng.

---

## Prediction 4

**Câu hỏi:** Hai attention heads có nhất thiết học cùng một attention pattern không?

**Prediction:** Không. Hai attention heads không nhất thiết phải học cùng một attention pattern.

**Lí do:** Mỗi attention head sử dụng các ma trận trọng số riêng để tạo Query, Key và Value. Do đó, mỗi head có thể học những mối quan hệ khác nhau giữa các token. Ví dụ, một head có thể chú ý đến mối quan hệ giữa chủ ngữ và động từ, trong khi head khác có thể tập trung vào mối quan hệ giữa từ phủ định và từ biểu thị cảm xúc. Tuy nhiên, đây chỉ là những ví dụ trực quan, không phải quy tắc cố định.


---

## Prediction 5

**Câu hỏi:** Tăng số attention heads có chắc chắn làm accuracy tăng không?

**Prediction:** Không. Tăng số attention heads không đảm bảo accuracy sẽ tăng.

**Lí do:** Nhiều attention heads giúp mô hình có khả năng học nhiều loại quan hệ khác nhau hơn. Tuy nhiên, hiệu quả còn phụ thuộc vào dữ liệu, kiến trúc mô hình, hyperparameters và quá trình huấn luyện. Nếu số heads quá lớn so với lượng dữ liệu hoặc tài nguyên tính toán, mô hình có thể không cải thiện hiệu suất, thậm chí gây lãng phí tài nguyên.


---

## Prediction 6

**Câu hỏi:** Transformer có luôn nhanh hơn RNN trên mọi sequence length không?

**Prediction:** Không. Transformer không phải lúc nào cũng nhanh hơn RNN trên mọi sequence length.

**Reason:** Transformer có thể xử lý song song các token, giúp tận dụng GPU hiệu quả hơn. Tuy nhiên, self-attention có computational complexity theo sequence length là O(n²), do phải tính toán mối quan hệ giữa các cặp token. Trong khi đó, RNN có complexity khoảng O(n) theo sequence length nhưng phải xử lý tuần tự. Vì vậy, Transformer thường có lợi thế về khả năng song song hóa, nhưng với sequence ngắn hoặc tài nguyên hạn chế, RNN vẫn có thể nhanh hơn.

---


## Reflection trước Experiment

Các dự đoán trên cho thấy Transformer có lợi thế về khả năng xử lý song song và học nhiều mối quan hệ giữa các token thông qua Multi-Head Attention. Tuy nhiên, Positional Encoding vẫn là thành phần quan trọng để mô hình nhận biết thứ tự trong sequence.

Ngoài ra, việc tăng số attention heads không đảm bảo cải thiện accuracy, và chi phí tính toán của self-attention có thể trở thành hạn chế khi sequence quá dài.

Các experiment tiếp theo sẽ được sử dụng để kiểm chứng những dự đoán này bằng kết quả thực nghiệm.
