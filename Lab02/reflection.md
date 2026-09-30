# 24. Reflection

## Câu 1: Nếu tăng n, mô hình nhận thêm thông tin gì?
Khi tăng \(n\), mô hình nhận thêm thông tin về các từ trước đó trong context. Unigram không sử dụng context, bigram sử dụng 1 từ trước, còn trigram sử dụng 2 từ trước. Context dài hơn cho mô hình thêm thông tin để dự đoán từ tiếp theo.


## Câu 2: Tại sao tăng n lại làm sparsity tăng?
Khi \(n\) tăng, số lượng n-gram có thể có tăng rất nhanh, trong khi lượng dữ liệu vẫn cố định. Vì vậy, nhiều n-gram không xuất hiện hoặc chỉ xuất hiện rất ít lần trong corpus, làm dữ liệu trở nên sparse hơn. Kết quả Experiment 1 cũng cho thấy số lượng unique n-gram tăng mạnh từ unigram đến trigram.


## Câu 3: Tại sao smoothing cần thiết?
Nếu một n-gram chưa xuất hiện trong training corpus thì MLE cho xác suất bằng 0. Khi đó chỉ cần một n-gram có xác suất bằng 0 thì xác suất của cả câu cũng bằng 0 và log probability trở thành \(-\infty\). Smoothing giúp gán một xác suất dương cho những n-gram chưa quan sát thấy và tránh vấn đề zero probability.


## Câu 4: Perplexity đo điều gì?
Perplexity đo mức độ khó đoán của một sequence đối với language model. Perplexity thấp hơn nghĩa là model gán xác suất cao hơn cho dữ liệu đang được đánh giá. Khi so sánh perplexity cần sử dụng cùng dữ liệu và cùng cách xử lý dữ liệu.


## Câu 5: Perplexity thấp hơn có luôn tạo văn bản tốt hơn đối với con người không?
Không. Perplexity đo khả năng dự đoán dữ liệu của model, nhưng không trực tiếp đo chất lượng văn bản đối với con người như tính mạch lạc, ý nghĩa hay tính tự nhiên. Vì vậy, perplexity thấp hơn không đồng nghĩa với việc văn bản sinh ra luôn tốt hơn đối với con người.


## Câu 6: N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?
N-gram chủ yếu dựa trên thống kê của các chuỗi từ và chỉ sử dụng context có độ dài cố định. Vì vậy, nó khó nắm bắt các quan hệ xa trong câu, gặp vấn đề với những n-gram chưa từng xuất hiện và không thực sự hiểu ý nghĩa của ngôn ngữ như con người.


## Câu 7: Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?
Không. Trigram chỉ sử dụng 2 từ trước đó để dự đoán từ tiếp theo. Vì vậy, nếu context có 100 từ thì phần lớn context trước đó sẽ bị bỏ qua. Đây là một hạn chế của n-gram và là động lực để sử dụng các mô hình có khả năng xử lý context dài hơn như neural language models.
