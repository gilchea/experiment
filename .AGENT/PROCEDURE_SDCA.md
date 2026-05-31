Dựa trên phần 4 của bài báo NIPS 2013 ("SDCA as Variance Reduction"), phương pháp **SDCA** (Stochastic Dual Coordinate Ascent) có thể được mô tả như một thuật toán giảm phương sai bằng cách duy trì các biến đối ngẫu,.

Dưới đây là thuật toán hoàn chỉnh được thiết kế để thiết lập thí nghiệm dựa trên các công thức lý thuyết trong nguồn tài liệu:

### Thuật toán SDCA (Góc nhìn Giảm phương sai)

**Bài toán tối ưu hóa:**
Tìm $w^* = \text{argmin}_w P(w)$,
 với $P(w) = \frac{1}{n} \sum_{i=1}^{n} \phi_i(w) + \frac{1}{2}\lambda w^\top w$.

**Đầu vào:**
*   **Tham số điều chỉnh:** $\lambda > 0$.
*   **Tốc độ học:** $\eta_t$ (Trong SDCA, $\eta_t$ có thể giữ cố định hoặc không cần giảm dần về 0 như SGD truyền thống để đạt được hội tụ),.
*   **Số lượng mẫu:** $n$.

---

#### 1. Khởi tạo (Initialization)
*   Khởi tạo các **biến đối ngẫu** cho từng mẫu dữ liệu: $\alpha_i^{(0)} \in \mathbb{R}^d$ với $i = 1, \dots, n$ (thường đặt $\alpha_i^{(0)} = 0$).
*   Thiết lập trọng số ban đầu dựa trên tổng các biến đối ngẫu: 
    $w^{(0)} = \sum_{i=1}^{n} \alpha_i^{(0)}$,.

#### 2. Vòng lặp chính (Main Loop)
Cho mỗi bước lặp $t = 1, 2, \dots$:

*   **Bước A: Lấy mẫu ngẫu nhiên**
    Chọn ngẫu nhiên một chỉ số dữ liệu $i \in \{1, \dots, n\}$.

*   **Bước B: Cập nhật biến đối ngẫu cho mẫu $i$**
    Tính toán giá trị mới cho $\alpha_i$ dựa trên gradient của hàm mất mát $\phi_i$ và giá trị đối ngẫu hiện tại:
    $\alpha_i^{(t)} = \alpha_i^{(t-1)} - \eta_t \left( \nabla \phi_i(w^{(t-1)}) + \lambda n \alpha_i^{(t-1)} \right)$.
    *(Lưu ý: Các biến $\alpha_l$ với $l \neq i$ được giữ nguyên: $\alpha_l^{(t)} = \alpha_l^{(t-1)}$)*.

*   **Bước C: Cập nhật trọng số nguyên thủy (Primal weight update)**
    Cập nhật $w$ dựa trên sự thay đổi của biến đối ngẫu vừa tính:
    $w^{(t)} = w^{(t-1)} + \left( \alpha_i^{(t)} - \alpha_i^{(t-1)} \right)$.

---

### Các lưu ý để tạo thí nghiệm hiệu quả (theo nguồn tài liệu)

1.  **Cơ chế giảm phương sai:** Điểm mấu chốt của thí nghiệm là quan sát thành phần $(\nabla \phi_i(w) + \lambda n \alpha_i)$. Khi thuật toán hội tụ về điểm tối ưu $(w^*, \alpha^*)$, thành phần này sẽ tiến về 0, làm cho phương sai của bước cập nhật triệt tiêu dần.
2.  **Yêu cầu bộ nhớ:** Khác với SVRG, SDCA bắt buộc phải **lưu trữ toàn bộ $n$ biến đối ngẫu** $\alpha_i$ trong bộ nhớ,. Trong thí nghiệm với các bài toán tuyến tính (như Logistic Regression), bạn có thể tối ưu bằng cách chỉ lưu trữ các giá trị vô hướng nếu gradient có dạng tích của một số vô hướng và vector đặc trưng $x_i$,.
3.  **So sánh với SGD:** Trong thí nghiệm, bạn nên so sánh SDCA với **SGD-best** (phiên bản SGD được tinh chỉnh tốc độ học giảm dần tốt nhất). Nguồn tài liệu cho thấy SDCA thường hội tụ tuyến tính và có đường cong lỗi tương đương với SVRG trên các bài toán lồi như MNIST hay CIFAR-10,.
4.  **Đánh giá:** Trục hoành của biểu đồ thí nghiệm nên là **số lần tính gradient chia cho $n$** (#grad / n) để đảm bảo so sánh công bằng về chi phí tính toán giữa các phương pháp,.