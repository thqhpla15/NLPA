**Part I — Error Analysis:**

**Query 1: `transformer language model`**

Expected relevant document: `{25428}`

Retrieved:

| Rank | Document ID | Similarity | Preview |
|---|---|---|---|
| 1 | 27936 | 0.471849 | Transformer/circuit board, Hobby 720 model |
| 2 | 25428 | 0.279103 | Facebook language settings |
| 3 | 4075 | 0.222687 | Spanish language instructor |
| 4 | 701 | 0.216096 | French as Foreign Language program |
| 5 | 13690 | 0.213617 | Finnish language text |

- Document 27936 đứng đầu vì có similarity cao nhất: `0.471849`.
- Các từ `"transformer"` và `"model"` có thể đóng góp nhiều vào similarity.
- Có lexical overlap giữa query và document 27936.
- Document relevant `25428` vẫn nằm trong Top-5 nhưng chỉ ở rank 2.
- Đây là hạn chế của lexical matching. TF-IDF dựa nhiều vào các từ xuất hiện trong query và document nên có thể xếp một document khác chủ đề lên cao.

<br>

**Query 2: `deep learning healthcare`**

Expected relevant document: `{6123}`

Retrieved:

| Rank | Document ID | Similarity | Preview |
|---|---|---|---|
| 1 | 9252 | 0.278174 | West Health healthcare organization |
| 2 | 6123 | 0.275276 | Online bachelor's degree in health-related fields |
| 3 | 11119 | 0.274719 | Big Data in healthcare systems |
| 4 | 11979 | 0.273027 | Doctorate of Healthcare Organization |
| 5 | 7564 | 0.240487 | English grammar |

- Document 9252 đứng đầu với similarity `0.278174`, cao hơn document relevant 6123: `0.275276`.
- Từ `"healthcare"` có thể đóng góp nhiều vào similarity.
- Có lexical overlap giữa query và các documents trong Top-5.
- Document relevant `6123` nằm ở rank 2.
- Nguyên nhân chủ yếu là lexical matching. TF-IDF có thể ưu tiên documents có từ `"healthcare"` dù không có đầy đủ ngữ cảnh `"deep learning healthcare"`.

<br>

**Query 3: `medical image classification`**

Expected relevant document: `{17794}`

Retrieved:

| Rank | Document ID | Similarity | Preview |
|---|---|---|---|
| 1 | 18971 | 0.400030 | Environmental classification system |
| 2 | 8527 | 0.349998 | History of maize classification |
| 3 | 19908 | 0.253427 | League of Legends wallpapers |
| 4 | 17794 | 0.240507 | WordPress image sizes |
| 5 | 12658 | 0.233145 | Pharmacists / medical devices |

- Document 18971 đứng đầu với similarity `0.400030`.
- Từ `"classification"` có thể đóng góp nhiều vào similarity.
- Document 18971 và 8527 có lexical overlap với query thông qua `"classification"`.
- Document relevant `17794` vẫn nằm trong Top-5 nhưng chỉ ở rank 4.
- TF-IDF không hiểu đầy đủ ý nghĩa của `"medical image classification"`, nên có thể ưu tiên documents có từ khóa giống query nhưng khác chủ đề.

<br>

**Query 4: `natural language processing`**

Expected relevant documents: `∅`

Retrieved:

| Rank | Document ID | Similarity | Preview |
|---|---|---|---|
| 1 | 25428 | 0.362266 | Facebook language settings |
| 2 | 8705 | 0.341789 | Food Safety regulations |
| 3 | 4075 | 0.289040 | Spanish language instructor |
| 4 | 5699 | 0.282915 | Truth-value gaps in natural language |
| 5 | 701 | 0.280485 | French as Foreign Language program |

- Document 25428 đứng đầu với similarity `0.362266`.
- Từ `"language"` có thể đóng góp nhiều vào similarity.
- Có lexical overlap giữa query và các documents được truy xuất.
- Không có relevant document nào theo relevance labels của evaluation set.
- TF-IDF có thể tìm được documents chứa `"language"` nhưng không hiểu `"natural language processing"` là một khái niệm cụ thể.

<br>

**Failure case quan trọng nhất:**

Query: `medical image classification`

Relevant document: `17794`

| Rank | Document ID | Similarity |
|---|---|---|
| 1 | 18971 | 0.400030 |
| 2 | 8527 | 0.349998 |
| 3 | 19908 | 0.253427 |
| **4** | **17794** | **0.240507** |

Document relevant `17794` chỉ đứng ở rank 4, trong khi các documents `18971` và `8527` có similarity cao hơn.

Nguyên nhân có thể là TF-IDF phụ thuộc nhiều vào lexical overlap. Các từ như `"classification"` hoặc `"image"` có thể làm similarity tăng dù document không đúng với ngữ cảnh của query.

TF-IDF dựa trên thống kê xuất hiện của các token nên không hiểu đầy đủ quan hệ ngữ nghĩa giữa `"medical"`, `"image"` và `"classification"`. Vì vậy, document có nhiều từ giống query chưa chắc là document phù hợp nhất.
