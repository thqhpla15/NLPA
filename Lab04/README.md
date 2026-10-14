# Lab 04 — Phân loại cảm xúc đánh giá phim IMDb

## 1. Tổng quan dự án

Dự án xây dựng hệ thống phân loại văn bản nhằm dự đoán một bài đánh giá phim IMDb thể hiện cảm xúc tích cực hay tiêu cực.

Dự án tìm hiểu các phương pháp tiền xử lý văn bản, biểu diễn đặc trưng TF-IDF, vector từ Word2Vec, mô hình Logistic Regression, đánh giá mô hình, đặc trưng n-gram, phân tích lỗi và dự đoán cảm xúc.

## 2. Bộ dữ liệu

- **Tên bộ dữ liệu:** IMDb Movie Reviews
- **Nguồn:** Hugging Face Datasets (`stanfordnlp/imdb`)
- **Tập huấn luyện ban đầu:** 25.000 bài đánh giá
- **Tập kiểm thử chính thức:** 25.000 bài đánh giá
- **Tập không gán nhãn:** 50.000 bài đánh giá
- **Nhãn:** Tiêu cực (0), Tích cực (1)

Tập huấn luyện ban đầu được chia thành tập huấn luyện và tập xác thực theo tỷ lệ 80:20, sử dụng phương pháp lấy mẫu phân tầng để duy trì phân bố nhãn. Tập kiểm thử chính thức được giữ riêng để đánh giá cuối cùng.

## 3. Cấu trúc dự án

```text
Lab04/
├── data/
├── output/
├── src/
│   ├── config.py
│   ├── dataset_loader.py
│   ├── dataset_inspector.py
│   ├── data_splitter.py
│   ├── tfidf_classifier.py
│   ├── word2vec_classifier.py
│   ├── model_evaluator.py
│   ├── error_analyzer.py
│   ├── final_evaluator.py
│   ├── ngram_experiment.py
│   ├── model_interpreter.py
│   ├── sentiment_predictor.py
│   ├── word_order_experiment.py
│   └── main.py
├── requirements.txt
└── README.md
```

## 4. Các mô hình và thí nghiệm

### Thí nghiệm 1: Khảo sát bộ dữ liệu

Kiểm tra phân bố lớp, giá trị bị thiếu, các bài đánh giá trùng lặp và độ dài văn bản.

### Thí nghiệm 2: TF-IDF + Logistic Regression

Sử dụng đặc trưng TF-IDF kết hợp unigram và bigram để phân loại cảm xúc của bài đánh giá.

### Thí nghiệm 3: So sánh phương pháp biểu diễn văn bản

So sánh đặc trưng TF-IDF với vector Word2Vec trung bình, sử dụng Logistic Regression làm bộ phân loại.

### Thí nghiệm 4: Ảnh hưởng của thứ tự từ

So sánh biểu diễn Word2Vec trung bình của hai câu có và không có từ phủ định `not`.

### Thí nghiệm 5: So sánh đặc trưng n-gram

So sánh unigram với sự kết hợp giữa unigram và bigram.

### Thí nghiệm 6: Diễn giải mô hình

Trích xuất các đặc trưng có trọng số Logistic Regression cao nhất theo hướng tích cực và tiêu cực.

### Các nhiệm vụ bổ sung

Đánh giá mô hình trên tập kiểm thử chính thức, kiểm tra các trường hợp phân loại sai và xây dựng chức năng dự đoán cảm xúc cho văn bản do người dùng nhập.

## 5. Các chỉ số đánh giá

Các mô hình được đánh giá bằng những chỉ số sau:

- Accuracy (Độ chính xác tổng thể)
- Precision (Độ chính xác theo lớp)
- Recall (Độ bao phủ theo lớp)
- F1-score
- Ma trận nhầm lẫn (Confusion Matrix)

Kết quả thí nghiệm và hình ảnh trực quan hóa được lưu trong thư mục `output/`.

## 6. Cài đặt

Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```

## 7. Cách chạy dự án

Chạy các tệp sau bằng trình thông dịch Python đã cấu hình trong IntelliJ IDEA:

- `main.py` — chạy các thí nghiệm phân loại chính.
- `ngram_experiment.py` — so sánh đặc trưng unigram và bigram.
- `model_interpreter.py` — trích xuất các đặc trưng tích cực và tiêu cực nổi bật.
- `word_order_experiment.py` — thực hiện thí nghiệm về thứ tự từ với Word2Vec.
- `final_evaluator.py` — đánh giá mô hình trên tập kiểm thử chính thức.
- `sentiment_predictor.py` — dự đoán cảm xúc của văn bản đầu vào.

Một số thí nghiệm cần huấn luyện mô hình riêng nên có thể mất thêm thời gian.

## 8. Các tệp kết quả

Thư mục `output/` chứa các kết quả thí nghiệm, bao gồm:

- Biểu đồ phân bố lớp.
- Các ma trận nhầm lẫn.
- Kết quả so sánh mô hình.
- Kết quả so sánh n-gram.
- Kết quả phân tích lỗi.
- Kết quả đánh giá trên tập kiểm thử cuối cùng.
- Các đặc trưng có trọng số tích cực và tiêu cực cao nhất.

## 9. Khai báo sử dụng AI

Các công cụ AI được sử dụng để hỗ trợ lập trình, gỡ lỗi và giải thích trong quá trình phát triển dự án. Sinh viên chịu trách nhiệm kiểm tra mã nguồn, trực tiếp chạy thí nghiệm, xem xét kết quả và tự hoàn thành phần phân tích, nhận xét và phản ánh quá trình học tập.

Bản khai báo sử dụng AI chi tiết cần nêu rõ công cụ đã sử dụng, mục đích sử dụng, phần mã được tạo, những chỉnh sửa đã thực hiện và cách kiểm chứng kết quả.
