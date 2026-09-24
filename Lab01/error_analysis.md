# Part I — Error Analysis

## 1. Query: `transformer language model`

**Expected relevant documents:** `{25428}`

### Retrieved Documents

| Rank | Document ID | Similarity | Document Preview |
|---:|---:|---:|---|
| 1 | 27936 | 0.471849 | Transformer/circuit board, Hobby 720 model |
| 2 | 25428 | 0.279103 | Facebook language settings |
| 3 | 4075 | 0.222687 | Spanish language instructor |
| 4 | 701 | 0.216096 | French as Foreign Language program |
| 5 | 13690 | 0.213617 | Finnish language text |

### Analysis

1. **Vì sao document đứng đầu?**  
   Document 27936 đứng đầu với similarity `0.471849`, cao nhất trong Top-5. Preview cho thấy document có nội dung liên quan đến `"transformer"` và `"model"`.

2. **Những từ nào đóng góp nhiều vào similarity?**  
   Các từ `"transformer"` và `"model"` có khả năng đóng góp đáng kể vì chúng xuất hiện trong query và document đứng đầu.

3. **Có lexical overlap không?**  
   Có. Document 27936 có lexical overlap với query thông qua các từ `"transformer"` và `"model"`.

4. **Có relevant document nào bị bỏ sót không?**  
   Document relevant `25428` không bị bỏ sót khỏi Top-5, nhưng chỉ được xếp ở **rank 2**.

5. **Failure này xuất phát từ đâu?**  
   Kết quả cho thấy hạn chế chủ yếu nằm ở **lexical matching**: TF-IDF dựa trên sự xuất hiện của các từ nên có thể ưu tiên document có các từ `"transformer"` và `"model"` dù ngữ cảnh không hoàn toàn phù hợp với `"transformer language model"`.

---

## 2. Query: `deep learning healthcare`

**Expected relevant documents:** `{6123}`

### Retrieved Documents

| Rank | Document ID | Similarity | Document Preview |
|---:|---:|---:|---|
| 1 | 9252 | 0.278174 | West Health healthcare organization |
| 2 | 6123 | 0.275276 | Online bachelor's degree in health-related fields |
| 3 | 11119 | 0.274719 | Big Data in healthcare systems |
| 4 | 11979 | 0.273027 | Doctorate of Healthcare Organization |
| 5 | 7564 | 0.240487 | English grammar |

### Analysis

1. **Vì sao document đứng đầu?**  
   Document 9252 đứng đầu với similarity `0.278174`, cao hơn document relevant 6123 (`0.275276`).

2. **Những từ nào đóng góp nhiều vào similarity?**  
   Từ `"healthcare"` có khả năng đóng góp đáng kể vì nhiều document trong Top-5 đều liên quan đến healthcare.

3. **Có lexical overlap không?**  
   Có. Các document Top-5 có lexical overlap với query, đặc biệt thông qua từ `"healthcare"`.

4. **Có relevant document nào bị bỏ sót không?**  
   Document relevant `6123` được tìm thấy ở **rank 2**, nên không bị bỏ sót khỏi Top-5.

5. **Failure này xuất phát từ đâu?**  
   Kết quả cho thấy **lexical matching** đóng vai trò lớn. Các document có từ `"healthcare"` có thể nhận similarity cao ngay cả khi không chứa đầy đủ ngữ cảnh `"deep learning healthcare"`.

---

## 3. Query: `medical image classification`

**Expected relevant documents:** `{17794}`

### Retrieved Documents

| Rank | Document ID | Similarity | Document Preview |
|---:|---:|---:|---|
| 1 | 18971 | 0.400030 | Environmental classification system |
| 2 | 8527 | 0.349998 | History of maize classification |
| 3 | 19908 | 0.253427 | League of Legends wallpapers |
| 4 | 17794 | 0.240507 | WordPress image sizes |
| 5 | 12658 | 0.233145 | Pharmacists / medical devices |

### Analysis

1. **Vì sao document đứng đầu?**  
   Document 18971 có similarity cao nhất (`0.400030`). Preview cho thấy document liên quan đến **classification**, nên có sự trùng khớp từ khóa với query.

2. **Những từ nào đóng góp nhiều vào similarity?**  
   Từ `"classification"` có khả năng đóng góp lớn vì xuất hiện trong query và các document đứng đầu.

3. **Có lexical overlap không?**  
   Có. Các document 18971 và 8527 có lexical overlap với query, đặc biệt là từ `"classification"`.

4. **Có relevant document nào bị bỏ sót không?**  
   Không bị bỏ sót hoàn toàn, vì document relevant `17794` vẫn nằm trong Top-5. Tuy nhiên, nó chỉ đứng ở **rank 4**.

5. **Failure này xuất phát từ đâu?**  
   Đây chủ yếu là vấn đề **lexical matching**. TF-IDF nhận thấy các từ khóa như `"classification"` và `"image"` nhưng không hiểu đầy đủ quan hệ ngữ nghĩa giữa `"medical"`, `"image"` và `"classification"`. Vì vậy, các document có lexical overlap nhưng khác chủ đề vẫn được xếp cao.

---

## 4. Query: `natural language processing`

**Expected relevant documents:** `∅`

### Retrieved Documents

| Rank | Document ID | Similarity | Document Preview |
|---:|---:|---:|---|
| 1 | 25428 | 0.362266 | Facebook language settings |
| 2 | 8705 | 0.341789 | Food Safety regulations |
| 3 | 4075 | 0.289040 | Spanish language instructor |
| 4 | 5699 | 0.282915 | Truth-value gaps in natural language |
| 5 | 701 | 0.280485 | French as Foreign Language program |

### Analysis

1. **Vì sao document đứng đầu?**  
   Document 25428 đứng đầu với similarity `0.362266`, cao nhất trong các document được truy xuất.

2. **Những từ nào đóng góp nhiều vào similarity?**  
   Từ `"language"` có khả năng đóng góp đáng kể vì xuất hiện trong query và trong nhiều document được truy xuất.

3. **Có lexical overlap không?**  
   Có. Các document trong Top-5 có sự trùng lặp về từ khóa, đặc biệt với `"language"` và `"natural language"`.

4. **Có relevant document nào bị bỏ sót không?**  
   Theo relevance labels của evaluation set, **không có relevant document nào được gán cho query này**, nên không thể xác định một relevant document bị bỏ sót từ tập nhãn hiện tại.

5. **Failure này xuất phát từ đâu?**  
   Kết quả cho thấy vấn đề chủ yếu là **lexical matching**. TF-IDF có thể tìm được các document chứa những từ giống query như `"language"`, nhưng không hiểu rằng `"natural language processing"` là một khái niệm cụ thể.

---

# Failure Case Quan Trọng Nhất

## Query

`medical image classification`

## Relevant Document

`17794`

## Retrieved Ranking

| Rank | Document ID | Similarity |
|---:|---:|---:|
| 1 | 18971 | 0.400030 |
| 2 | 8527 | 0.349998 |
| 3 | 19908 | 0.253427 |
| **4** | **17794** | **0.240507** |

## Detailed Analysis

Đây là một failure case rõ ràng vì document relevant `17794` chỉ được xếp ở rank 4, trong khi các document không được gán nhãn relevant như `18971` và `8527` có similarity cao hơn.

Kết quả cho thấy TF-IDF phụ thuộc mạnh vào **lexical overlap** giữa query và document. Các từ như `"classification"` hoặc `"image"` có thể làm tăng similarity dù document không có đúng ngữ cảnh của `"medical image classification"`.

Vì TF-IDF chủ yếu dựa trên thống kê xuất hiện của token và không biểu diễn đầy đủ quan hệ ngữ nghĩa, hệ thống có thể ưu tiên các document có từ khóa giống query nhưng khác về ý nghĩa.
