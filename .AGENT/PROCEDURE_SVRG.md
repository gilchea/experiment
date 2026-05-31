Thuật toán **SVRG (Stochastic Variance Reduced Gradient)** - một thuật toán tối ưu hóa đột phá giúp khắc phục nhược điểm về phương sai của SGD truyền thống.

---

## Thuật toán SVRG (Stochastic Variance Reduced Gradient)

**Đầu vào:**
* Tốc độ học (Step size) $\eta$.
* Số lần lặp trong (Update frequency) $m$.
* Điểm bắt đầu (Initial weight) $\tilde{w}_0$.

---

### Cấu trúc thuật toán

**Cho mỗi vòng lặp (epoch) $s = 1, 2, \dots$:**
1.  **Tính Gradient toàn phần (Full Gradient):**
    $$\tilde{\mu} = \frac{1}{n} \sum_{i=1}^{n} \nabla \psi_i(\tilde{w}_{s-1})$$
    *(Đây là bước tính trung bình đạo hàm của tất cả dữ liệu tại điểm snapshot $\tilde{w}_{s-1}$)*.

2.  **Thiết lập giá trị ban đầu cho vòng lặp trong:**
    $$w_0 = \tilde{w}_{s-1}$$

3.  **Vòng lặp trong (Inner Loop) $t = 1, 2, \dots, m$:**
    * Chọn ngẫu nhiên một chỉ số dữ liệu $i_t \in \{1, \dots, n\}$.
    * **Cập nhật trọng số:**
        $$w_t = w_{t-1} - \eta \left( \nabla \psi_{i_t}(w_{t-1}) - \nabla \psi_{i_t}(\tilde{w}_{s-1}) + \tilde{\mu} \right)$$

4.  **Cập nhật Snapshot cho vòng lặp ngoài tiếp theo:**
    * Tùy chọn 1 (Trong hình): $\tilde{w}_s = w_m$
    * Tùy chọn 2 (Lý thuyết): $\tilde{w}_s = w_t$ với $t$ chọn ngẫu nhiên từ $\{0, \dots, m-1\}$.

---

### Giải thích chi tiết các thành phần (Annotations)

#### 1. Thành phần hiệu chỉnh phương sai (The Variance Reduction Core)
Trái tim của thuật toán nằm ở biểu thức cập nhật:
$$\mathbf{g}_t = \underbrace{\nabla \psi_{i_t}(w_{t-1})}_{\text{SGD thông thường}} - \underbrace{\left( \nabla \psi_{i_t}(\tilde{w}_{s-1}) - \tilde{\mu} \right)}_{\text{Thành phần hiệu chỉnh (Correction)}}$$

* **$\nabla \psi_{i_t}(w_{t-1})$**: Là gradient ngẫu nhiên tại điểm hiện tại (giống SGD). Nó có phương sai rất lớn.
* **$\nabla \psi_{i_t}(\tilde{w}_{s-1}) - \tilde{\mu}$**: Đây là "độ lệch" của mẫu $i_t$ so với trung bình toàn thể tại điểm snapshot.
* **Ý nghĩa:** Khi thuật toán dần hội tụ, $w_{t-1}$ và $\tilde{w}_{s-1}$ sẽ tiến lại gần nhau. Khi đó, $\nabla \psi_{i_t}(w_{t-1}) - \nabla \psi_{i_t}(\tilde{w}_{s-1})$ sẽ tiến về $0$. Lúc này, hướng cập nhật chỉ còn phụ thuộc vào $\tilde{\mu}$ (gradient chính xác), giúp triệt tiêu nhiễu (noise) và cho phép dùng tốc độ học $\eta$ cố định mà vẫn hội tụ.

#### 2. Vòng lặp kép (Double Loop Structure)
* **Vòng lặp ngoài (Outer Loop):** Thực hiện tính toán nặng (full gradient). Mục đích là để "neo" hướng đi đúng đắn cho các bước chạy ngẫu nhiên phía sau. Thường chỉ chạy khoảng 10-100 vòng.
* **Vòng lặp trong (Inner Loop):** Chạy nhanh với độ phức tạp $O(1)$ mỗi bước. Giá trị $m$ thường được chọn bằng $2n$ hoặc $5n$ (với $n$ là số lượng mẫu dữ liệu).

#### 3. Tại sao SVRG lại "tốt" hơn?
* **SGD truyền thống:** Phương sai không đổi, buộc phải giảm dần $\eta$ (learning rate decay) khiến tốc độ hội tụ chậm ($O(1/t)$).
* **SVRG:** Phương sai giảm dần về 0 khi tiến tới điểm tối ưu. Điều này cho phép hội tụ với tốc độ tuyến tính ($O(e^{-kt})$) đối với hàm lồi mạnh - nhanh hơn rất nhiều so với SGD.

---

### Lưu ý khi cài đặt thực nghiệm (Implementation Tips)
* **Lưu trữ:** SVRG chỉ cần lưu trữ $\tilde{w}$ và $\tilde{\mu}$, không cần lưu trữ toàn bộ $n$ gradient như các phương pháp khác (SAG/SAGA), cực kỳ tiết kiệm bộ nhớ cho các mô hình lớn.
* **Tính toán:** Mỗi vòng lặp ngoài tốn $n$ lần tính gradient, vòng lặp trong tốn $2m$ lần. Tổng chi phí cho một epoch (vòng lặp ngoài) là khoảng $n + 2m$ lần tính gradient.

Ngoài các thông tin cơ bản về cấu trúc và công thức, các nguồn tài liệu cung cấp thêm những góc nhìn chuyên sâu và các biến thể kỹ thuật quan trọng để làm rõ bản chất của thuật toán SVRG:

### 1. Cách nhìn nhận qua "Hàm bổ trợ" (Auxiliary Function)
Một cách thú vị để hiểu công thức cập nhật của SVRG là coi nó như việc áp dụng SGD tiêu chuẩn lên một biểu diễn mới của hàm mục tiêu. Tác giả định nghĩa hàm bổ trợ:
$$\tilde{\psi}_i(w) = \psi_i(w) - (\nabla \psi_i(\tilde{w}) - \tilde{\mu})^\top w$$
Vì tổng của các thành phần hiệu chỉnh $(\nabla \psi_i(\tilde{w}) - \tilde{\mu})$ trên toàn bộ $n$ mẫu bằng 0, nên hàm mục tiêu $P(w)$ không đổi. Khi đó, bước cập nhật của SVRG thực chất là **gradient của hàm bổ trợ này**, giúp ta thấy rõ cách thuật toán "lái" hướng đi của gradient ngẫu nhiên mà không làm thay đổi điểm tối ưu của bài toán gốc.

### 2. Tối ưu hóa bộ nhớ cho các bài toán lồi tuyến tính
Dù SVRG được biết đến là không cần lưu trữ gradient, nhưng đối với các bài toán **tiên đoán tuyến tính lồi** (như Logistic Regression), ta có thể tối ưu thêm:
*   Thay vì tính toán lại toàn bộ gradient $\nabla \psi_i(\tilde{w})$ trong vòng lặp con, ta có thể chỉ cần lưu trữ các giá trị **vô hướng (scalars)** $\phi'_i(\tilde{w}^\top x_i)$.
*   Điều này giúp SVRG có mức tiêu thụ bộ nhớ thấp tương đương với các phương pháp như SAG hay SDCA nhưng vẫn giữ được sự đơn giản trong phân tích.

### 3. Biến thể SVRG xác suất (Probabilistic SVRG)
Trong bài báo của Driggs et al. (2022), một biến thể của SVRG được giới thiệu để phục vụ khung tăng tốc:
*   Thay vì cập nhật snapshot $\tilde{w}$ một cách định kỳ sau đúng $m$ bước (như bài báo 2013), biến thể này thực hiện tính toán full gradient tại mỗi bước lặp với một **xác suất $1/p$**.
*   Cách tiếp cận này cho phép thuật toán tích hợp mượt mà hơn vào các kỹ thuật **khớp nối tuyến tính (linear coupling)** để đạt được tốc độ tăng tốc $O(1/T^2)$ mà không cần cấu trúc vòng lặp lồng nhau cứng nhắc.

### 4. Chiến lược khởi tạo (Warm-start)
Để thuật toán đạt hiệu quả cao nhất, đặc biệt là với các bài toán không lồi như mạng thần kinh, các tác giả khuyên nên có bước khởi tạo:
*   Nên chạy **SGD trong một vài vòng lặp (epoch)** (ví dụ: 1 epoch cho bài toán lồi, 10 epoch cho mạng thần kinh) để đưa trọng số $w$ vào vùng lân cận của điểm tối ưu cục bộ trước khi bắt đầu cơ chế giảm phương sai của SVRG.
*   Lý do là vì SVRG phát huy hiệu quả tốt nhất khi điểm snapshot $\tilde{w}$ đã tương đối gần với điểm tối ưu $w^*$, giúp thành phần hiệu chỉnh phương sai hoạt động chính xác ngay từ đầu.

### 5. Phân tích độ phức tạp khi điều kiện bài toán kém (Condition Number)
Lý thuyết của SVRG chỉ ra một kết quả ấn tượng khi so sánh với Gradient Descent (GD) truyền thống:
*   Với bài toán có hệ số điều kiện $L/\gamma = n$ (hệ số điều kiện bằng đúng số lượng mẫu dữ liệu), GD yêu cầu xử lý tổng cộng $n^2 \ln(1/\epsilon)$ mẫu dữ liệu để đạt độ chính xác $\epsilon$.
*   Trong khi đó, SVRG chỉ cần xử lý **$n \ln(1/\epsilon)$** mẫu dữ liệu, tương đương với các phương pháp giảm phương sai tiên tiến nhất như SAG hay SDCA nhưng với cách chứng minh đơn giản và trực quan hơn nhiều.

### 6. Mối liên hệ với SDCA (Variance Reduction view)
Bài báo 2013 cũng làm rõ rằng **SDCA thực chất cũng là một phương pháp giảm phương sai** cho SGD. Điểm khác biệt là SDCA giảm phương sai thông qua việc cập nhật các biến đối ngẫu (dual variables), trong khi SVRG làm điều đó một cách trực tiếp trên biến nguyên thủy (primal variables) bằng cách sử dụng snapshot gradient toàn phần. Sự tương đồng này giải thích tại sao hai thuật toán thường có đường cong hội tụ thực nghiệm rất sát nhau trong các bài toán lồi.


