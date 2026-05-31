Dựa trên tài liệu, phương pháp **SDCA (Stochastic Dual Coordinate Ascent)** được xem là một kỹ thuật **giảm phương sai (variance reduction)** cho Gradient Ngẫu nhiên (SGD), tương tự như SVRG nhưng sử dụng các biến đối ngẫu để đạt được hiệu ứng này.

Dưới đây là chi tiết về cơ chế giảm phương sai của SDCA theo bài báo:

### 1. Bài toán và Biểu diễn Đối ngẫu
SDCA xem xét bài toán tối ưu hóa với hàm mất mát lồi $\phi_i(w)$ và thành phần điều chỉnh (regularization) $L_2$:
$$w^* = \text{argmin } P(w), \quad P(w) = \frac{1}{n} \sum_{i=1}^{n} \phi_i(w) + \frac{1}{2}\lambda w^\top w$$

Tại điểm tối ưu $w^*$, ta có một biểu diễn "đối ngẫu" của trọng số:
*   **Công thức:** $w^* = \sum_{i=1}^{n} \alpha_i^*$, trong đó các biến đối ngẫu được định nghĩa là $\alpha_i^* = -\frac{1}{\lambda n} \nabla \phi_i(w^*)$.

### 2. Quy tắc cập nhật trong SDCA
Thay vì cập nhật trọng số trực tiếp như SGD truyền thống, SDCA duy trì đại diện $w^{(t)} = \sum_{i=1}^{n} \alpha_i^{(t)}$ và áp dụng quy tắc cập nhật **Tăng tọa độ đối ngẫu (Dual Coordinate Ascent)** cho một chỉ số $i$ được chọn ngẫu nhiên:
*   **Cập nhật biến đối ngẫu:** $\alpha_i^{(t)} = \alpha_i^{(t-1)} - \eta_t (\nabla \phi_i(w^{(t-1)}) + \lambda n \alpha_i^{(t-1)})$.
*   **Cập nhật trọng số:** $w^{(t)} = w^{(t-1)} + (\alpha_i^{(t)} - \alpha_i^{(t-1)})$.

### 3. Tại sao SDCA là phương pháp giảm phương sai?
Tài liệu giải thích rằng SDCA đạt được hiệu ứng giảm phương sai nhờ vào sự hội tụ đồng thời của các biến nguyên thủy và đối ngẫu:
*   **Triệt tiêu nhiễu:** Khi các tham số $(w, \alpha)$ tiến dần đến tối ưu $(w^*, \alpha^*)$, thành phần $(\nabla \phi_i(w) + \lambda n \alpha_i)$ sẽ tiến dần về 0.
*   **Phương sai tiến về 0:** Hệ quả là ngay cả khi **tốc độ học $\eta_t$ không giảm dần** (giữ hằng số), phương sai của quá trình cập nhật vẫn tự động giảm dần về 0 theo thời gian:
    $$\frac{1}{n} \sum_{i=1}^{n} (\nabla \phi_i(w) + \lambda n \alpha_i)^2 \to 0$$
*   **Tốc độ hội tụ:** Nhờ cơ chế này, SDCA đạt được **tốc độ hội tụ tuyến tính** cho các hàm lồi mạnh và trơn, nhanh hơn nhiều so với tốc độ hội tụ dưới tuyến tính ($O(1/t)$) của SGD truyền thống.

### 4. So sánh với SVRG
Mặc dù cùng là kỹ thuật giảm phương sai, bài báo chỉ ra những điểm khác biệt quan trọng:
*   **Lưu trữ:** SDCA yêu cầu **lưu trữ tất cả các biến đối ngẫu** (hoặc gradient cũ), điều này tốn kém bộ nhớ ($O(n)$) và không khả thi cho các mô hình lớn như mạng thần kinh. SVRG khắc phục điểm này bằng cách không yêu cầu lưu trữ như vậy.
*   **Tính trực quan:** Tác giả lập luận rằng cách tiếp cận giảm phương sai của SVRG đơn giản và trực quan hơn SDCA, giúp việc phân tích toán học dễ dàng hơn.
*   **Thực nghiệm:** Trong các bài toán lồi (như Logistic Regression), SVRG và SDCA cho thấy hiệu suất cạnh tranh và các đường cong hội tụ thực nghiệm thường nằm sát nhau.