Dựa trên bài báo NIPS 2013, dưới đây là thuật toán **SGD (Stochastic Gradient Descent)** hoàn chỉnh được thiết kế để thực hiện các thí nghiệm đối chứng với các phương pháp giảm phương sai như SVRG hay SDCA.

### Thuật toán SGD (Stochastic Gradient Descent) cho Thực nghiệm

**Đầu vào:**
*   **Tốc độ học ban đầu (Initial learning rate):** $\eta_0$.
*   **Chiến lược giảm tốc độ học (Learning rate scheduling):** Để đạt hiệu quả tốt nhất (phiên bản "SGD-best" trong bài báo), tốc độ học $\eta_t$ cần giảm dần theo thời gian để kiểm soát phương sai.
*   **Trọng số khởi tạo:** $w_0$.

---

#### 1. Các chiến lược giảm tốc độ học (Learning Rate Scheduling)
Theo thực nghiệm của bài báo, để có một baseline SGD mạnh, bạn nên chọn một trong hai lịch trình sau:
*   **Giảm theo hàm mũ (Exponential decay):** $\eta(t) = \eta_0 a^{bt/n}$ (trong đó $a, b$ là các tham số điều chỉnh).
*   **Nghịch đảo thời gian ($t$-inverse):** $\eta(t) = \eta_0(1 + b \cdot t/n)^{-1}$ (trong đó $b$ là tham số điều chỉnh).

*(Lưu ý: $n$ là tổng số mẫu dữ liệu, $t$ là chỉ số bước lặp hiện tại)*.

#### 2. Cấu trúc vòng lặp cập nhật
Cho mỗi bước lặp $t = 1, 2, \dots$:

1.  **Lấy mẫu ngẫu nhiên:** Chọn ngẫu nhiên một chỉ số dữ liệu $i_t$ từ tập $\{1, \dots, n\}$.
2.  **Tính Gradient ngẫu nhiên:** Tính đạo hàm của hàm mất mát trên mẫu đã chọn: $\nabla \psi_{i_t}(w^{(t-1)})$.
3.  **Cập nhật trọng số:**
    $$w^{(t)} = w^{(t-1)} - \eta_t \nabla \psi_{i_t}(w^{(t-1)})$$
    Trong đó $\eta_t$ là tốc độ học tại bước $t$ tuân theo chiến lược đã chọn ở bước 1.

---

### Thiết lập thí nghiệm để so sánh công bằng (Experimental Setup)

Để kết quả thí nghiệm có tính thuyết phục như trong bài báo, bạn cần tuân thủ các quy tắc sau:

*   **Đơn vị đo lường (X-axis):** Trục hoành của biểu đồ nên là **số lần tính toán gradient chia cho $n$** (tương đương với số lượt quét qua toàn bộ dữ liệu - passes through data). Điều này giúp so sánh công bằng giữa SGD (mỗi bước tốn 1 gradient) và SVRG (mỗi giai đoạn tốn $n + 2m$ gradient).
*   **Kích thước mẫu (Mini-batch):** 
    *   Đối với các bài toán lồi (Logistic Regression): Sử dụng kích thước mẫu bằng **1** (mỗi bước lặp chọn 1 mẫu).
    *   Đối với bài toán không lồi (Neural Networks): Sử dụng mini-batch kích thước **10**.
*   **Tiêu chí đánh giá (Metrics):**
    *   **Training loss residual:** $P(w) - P(w^*)$, trong đó $P(w^*)$ là giá trị tối ưu tìm được bằng cách chạy Gradient Descent trong thời gian rất dài.
    *   **Test error rate:** Tỷ lệ lỗi trên tập dữ liệu kiểm tra độc lập.
    *   **Phương sai cập nhật (Update Variance):** Tính toán $E \| \eta_t \nabla \psi_{i_t}(w^{(t-1)}) \|^2$ để quan sát mức độ dao động của thuật toán.
*   **Khởi tạo:** Trong thực nghiệm SVRG, tác giả khởi tạo bằng 1-10 epoch SGD. Vì vậy, khi làm thí nghiệm SGD thuần túy, bạn nên bắt đầu từ $w_0$ ngẫu nhiên hoặc từ cùng một điểm xuất phát để thấy rõ sự khác biệt ở giai đoạn sau.

**Tóm lại**, điểm quan trọng nhất của thí nghiệm SGD trong bài báo này là việc **tinh chỉnh lịch trình giảm tốc độ học** (learning rate scheduling) sao cho "SGD-best" đạt được kết quả tốt nhất có thể trước khi so sánh với khả năng dùng tốc độ học hằng số của SVRG.