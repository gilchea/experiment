# THIẾT LẬP THỰC NGHIỆM CHI TIẾT (THEO JOHNSON & ZHANG, 2013)

Để chứng minh tính hiệu quả của thuật toán SVRG, các thực nghiệm được thiết kế nhằm so sánh tốc độ hội tụ và độ ổn định của nó với các phương pháp tối ưu hóa ngẫu nhiên hàng đầu.

### 1. Danh mục bộ dữ liệu và Tiền xử lý
Các thực nghiệm sử dụng 5 bộ dữ liệu đa dạng về đặc điểm (số lượng mẫu $n$ và số chiều $d$) cũng như tính chất bài toán:

| Bộ dữ liệu | Loại bài toán | Đặc điểm nổi bật | Tiền xử lý |
| :--- | :--- | :--- | :--- |
| **MNIST** | Phân loại chữ số | 10 lớp, hình ảnh xám | Chia cho 255 để đưa về khoảng $[0, 1]$. |
| **CIFAR-10** | Phân loại vật thể | 10 lớp, hình ảnh màu | Chia cho 255 để đưa về khoảng $[0, 1]$. |
| **rcv1.binary** | Phân loại văn bản | Dữ liệu thưa (sparse), nhị phân | Giữ nguyên định dạng thưa để tối ưu bộ nhớ. |
| **covtype.binary**| Thảm thực vật | Nhị phân | Chia đôi ngẫu nhiên (Train/Test). |

### 2. Mô hình và Hàm mục tiêu
Tác giả thử nghiệm trên hai kịch bản quan trọng:

#### A. Bài toán Lồi (Convex): Hồi quy Logistic điều chỉnh $L_2$
Hàm mục tiêu có dạng:
$$P(w) = \frac{1}{n} \sum_{i=1}^{n} \log(1 + \exp(-y_i w^\top x_i)) + \frac{\lambda}{2} \|w\|^2$$
* **Tham số điều chỉnh ($\lambda$):** 
    * $10^{-4}$ cho MNIST.
    * $10^{-3}$ cho CIFAR-10.
    * $10^{-5}$ cho rcv1, covtype.

#### B. Bài toán Không lồi (Non-convex): Mạng thần kinh (Neural Networks)
Kiểm chứng khả năng của SVRG trên bề mặt lỗi phức tạp:
* **Kiến trúc:** 1 lớp ẩn (100 nút), hàm kích hoạt Sigmoid, lớp đầu ra Softmax (10 nút).
* **Tham số điều chỉnh ($\lambda$):** $10^{-4}$ (MNIST) và $10^{-3}$ (CIFAR-10).

### 3. Cấu hình tham số thuật toán (Hyper-parameters)
Điểm mạnh của SVRG là sự đơn giản trong việc chọn tham số:

* **Tốc độ học ($\eta$):** Sử dụng tốc độ học **hằng số (constant)**. 
    * *Kinh nghiệm:* Chọn $\eta$ sao cho thuật toán ổn định (ví dụ 0.025 - 0.1 tùy dữ liệu).
* **Độ dài vòng lặp con ($m$):** * Bài toán lồi: $m = 2n$ (mỗi vòng lặp ngoài quét qua dữ liệu tương đương 2 lần).
    * Mạng thần kinh: $m = 5n$.
* **Khởi tạo (Warm-start):** * Lồi: Chạy **1 vòng lặp SGD** để lấy điểm xuất phát tốt.
    * Không lồi: Chạy **10 vòng lặp SGD**.
* **Kích thước Mini-batch:** * SVRG gốc dùng mini-batch = 1 (cập nhật từng mẫu).
    * Với mạng thần kinh, sử dụng mini-batch = 10 để tận dụng tính toán song song.

### 4. Phương pháp đối chứng (Baselines)
SVRG được đặt lên bàn cân cùng các "đối thủ" mạnh nhất:
1.  **SGD (Constant $\eta$):** SGD với tốc độ học không đổi.
2.  **SGD (Best):** Phiên bản SGD được tối ưu hóa cực hạn với lịch trình giảm tốc độ học $\eta_t = \eta_0 (1 + a \eta_0 t)^{-1}$ hoặc tương đương.
3.  **SDCA:** Phương pháp tối ưu hóa đối ngẫu, vốn rất mạnh cho các bài toán lồi có cấu trúc tổng hữu hạn.
4.  **SAG:** Phương pháp trung bình gradient ngẫu nhiên (cần lưu trữ gradient).

### 5. Tiêu chí và Thước đo đánh giá
Để đảm bảo tính khách quan, các tác giả sử dụng các đơn vị đo lường sau:

1.  **Số lượt quét dữ liệu (Effective Passes):** Tính bằng tổng số lần tính toán gradient chia cho $n$. Đây là thước đo công bằng nhất vì SVRG tốn thêm chi phí tính gradient toàn phần ở đầu mỗi vòng lặp ngoài.
    * *Công thức tính cho SVRG:* $1 \text{ (vòng ngoài)} + m/n \text{ (vòng trong)}$. Nếu $m=2n$, mỗi epoch SVRG tốn chi phí tương đương 3 lần quét dữ liệu.
2.  **Hàm mất mát dư thừa (Loss Residual):** $P(w) - P(w^*)$. Trong đó $P(w^*)$ là giá trị tối ưu đạt được bằng cách chạy Gradient Descent truyền thống trong rất nhiều vòng lặp.
3.  **Tỷ lệ lỗi kiểm tra (Test Error):** Khả năng tổng quát hóa của mô hình trên dữ liệu chưa thấy.
4.  **Phương sai của Gradient:** Đo lường mức độ "nhiễu" của hướng cập nhật khi tiến gần đến điểm hội tụ.

### 6. Ghi chú về triển khai kỹ thuật (Implementation Notes)
* **Tối ưu bộ nhớ:** Với hồi quy logistic, thay vì lưu toàn bộ vector gradient $\nabla \psi_i(\tilde{w})$, ta chỉ cần lưu giá trị vô hướng là đạo hàm của hàm tổn thất đối với tích vô hướng ($\phi'_i(\tilde{w}^\top x_i)$). Điều này giúp bộ nhớ của SVRG chỉ là $O(n)$, tương đương với SDCA.
* **Dáng điệu hội tụ:** Thực nghiệm cho thấy SVRG hội tụ theo đường thẳng (trên thang log), minh chứng cho tốc độ hội tụ tuyến tính, trong khi các đường của SGD thường bị "khựng" lại hoặc dao động mạnh do phương sai không triệt tiêu.

