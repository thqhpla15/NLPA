**Part B — Calculation Exercises:**

**Excersice 1: Count Vector:**

D1: [1,0,1,1,0]

D2: [0,1,1,1,0]

D3: [1,0,0,1,1]

<br>

**Excercise 2: TF:**

a. Tính tf(cat, D1):

$$
tf(cat,D1) = \frac{1}{3}
$$

b, Tính tf(eats, D1):

$$
tf(eats, D1) = \frac{1}{3}
$$

c, Tính tf(fish, D1):

$$
tf(fish, D1) = \frac{1}{3}
$$

**Kiểm tra:** True

$$
\frac{1}{3} + \frac{1}{3} + \frac{1}{3} = 1
$$

<br>

**Excercise 3: IDF:**
Corpus có:

$$
N = 3
$$

và:

$$
df(cat) = 2
$$

$$
df(dog) = 1
$$

$$
df(eats) = 2
$$

$$
df(fish) = 3
$$

$$
df(likes) = 1
$$


$$
idf(cat) = \log\frac{3}{2} \sim 0.405
$$

$$
idf(eats) = \log\frac{3}{2} \sim 0.405
$$

$$
idf(dog) = \log\frac{3}{1} \sim 1.099
$$

$$
idf(fish) = \log\frac{3}{3} \sim 0
$$

$$
idf(likes) = \log\frac{3}{1} \sim 1.099
$$

Vậy IDF thấp nhất là fish vì nó xuất hiện ở cả 3 documents

<br>

**Excercise 4: TF - IDF:**

$$
tfidf(cat,D1) = tf(cat,D1) \cdot idf(cat) = \frac{1}{3} \times \log\frac{3}{2} = 0.0586
$$

$$
tfidf(eats,D1) = tf(eats,D1) \cdot idf(eats) = \frac{1}{3} \times \log\frac{3}{2} = 0.0586
$$

$$
tfidf(fish,D1) = tf(fish,D1) \cdot idf(fish) = \frac{1}{3} \times 0 = 0
$$

<br>

**Excercise 5: Cosine Similarity:**

$$
x = [1,1,1]
$$

$$
y = [1,1,0]
$$

$$
\cos(x,y) = \frac{x^T \cdot y}{\|x\| \cdot \|y\|}
$$

Tích vô hướng: $$x^T \cdot y$$

$$
x^T \cdot y = 1 \times 1 + 1 \times 1 + 1 \times 0 = 2
$$

Độ dài x:

$$
\||x\|| = \sqrt{1^2 + 1^2 + 1^2} = \sqrt{3}
$$

Độ dài y:

$$
\||y\|| = \sqrt{1^2 + 1^2 + 0^2} = \sqrt{2}
$$

Vậy:

$$
\cos(x,y) = \frac{2}{\sqrt{6}} \sim 0.8165
$$

<br>

**Excercise 6: Prediction:**

D1 = "medical image classification"

D2 = "medical image analysis"

D3 = "natural language processing"

Query: "medical image classification"

Không dùng code, hãy dự đoán:
1. Document nào có similarity cao nhất? **D1**
2. Document nào có similarity thấp nhất? **D3**
3. Term nào có thể có giá trị IDF thấp? **medical và image**
4. Nếu bỏ IDF và chỉ sử dụng count vector thì ranking có thay đổi không? **không**

<br>
