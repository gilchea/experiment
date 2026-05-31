Trong bài báo NIPS 2013 (Johnson & Zhang), các tác giả đã thực hiện thực nghiệm trên **5 bộ dữ liệu chính** để chứng minh tính hiệu quả của thuật toán SVRG,. Các thực nghiệm này được chia thành hai nhóm: bài toán lồi (Hồi quy Logistic) và bài toán không lồi (Mạng thần kinh),.

Dưới đây là chi tiết về 5 bộ dữ liệu và các thực nghiệm tương ứng:

### 1. MNIST
*   **Mô tả:** Bộ dữ liệu chữ số viết tay gồm 10 lớp.
*   **Thực nghiệm lồi:** Tác giả thực hiện hồi quy Logistic đa lớp có điều chỉnh $L_2$ với tham số $\lambda = 10^{-4}$. Kết quả cho thấy SVRG hội tụ nhanh và ổn định hơn SGD ngay cả khi dùng tốc độ học hằng số.
*   **Thực nghiệm không lồi:** Huấn luyện mạng thần kinh có một lớp ẩn (100 nút) sử dụng hàm kích hoạt sigmoid. SVRG giúp giảm phương sai và hội tụ nhanh hơn so với SGD-best.

### 2. rcv1.binary
*   **Mô tả:** Bộ dữ liệu phân loại văn bản nhị phân từ trang LIBSVM.
*   **Thực nghiệm:** Thực hiện hồi quy Logistic có điều chỉnh $L_2$ với $\lambda = 10^{-5}$.
*   **Kết quả:** SVRG cho thấy sự vượt trội rõ rệt so với SGD được tinh chỉnh tốt nhất (SGD-best) về cả độ giảm hàm mất mát (loss residual) và tỷ lệ lỗi trên tập kiểm tra (test error rate).

### 3. covtype.binary (Covertype)
*   **Mô tả:** Bộ dữ liệu dự đoán loại thảm thực vật.
*   **Thực nghiệm:** Vì bộ dữ liệu này không có nhãn kiểm tra sẵn, tác giả đã chia ngẫu nhiên dữ liệu thành hai nửa để làm tập huấn luyện và tập kiểm tra. Tham số điều chỉnh $\lambda = 10^{-5}$.
*   **Kết quả:** SVRG duy trì hiệu suất cạnh tranh với SDCA và tốt hơn nhiều so với SGD truyền thống,.

### 4. protein
*   **Mô tả:** Bộ dữ liệu về cấu trúc protein.
*   **Thực nghiệm:** Tương tự như *covtype*, dữ liệu được chia đôi để huấn luyện và kiểm tra. Dữ liệu này đã được chuẩn hóa (standardized) trước khi chạy thuật toán. Tham số $\lambda = 10^{-5}$.
*   **Kết quả:** SVRG đạt được mức giảm hàm mất mát xuống gần mức tối ưu ($10^{-6}$) nhanh hơn các baseline khác.

### 5. CIFAR-10
*   **Mô tả:** Bộ dữ liệu hình ảnh vật thể gồm 10 lớp,.
*   **Thực nghiệm lồi:** Hồi quy Logistic đa lớp với $\lambda = 10^{-3}$,. Dữ liệu được chuẩn hóa về khoảng $$ bằng cách chia cho 255.
*   **Thực nghiệm không lồi:** Huấn luyện mạng thần kinh với mini-batch kích thước 10.
*   **Kết quả:** Trong cả hai trường hợp lồi và không lồi, SVRG đều cho thấy khả năng giảm phương sai hiệu quả, giúp đường cong hội tụ mượt mà và nhanh hơn SGD.

### Tóm tắt các thiết lập chung của thực nghiệm:
*   **Baseline đối chứng:** So sánh với **SGD** (tốc độ học cố định và tốc độ học giảm dần được tinh chỉnh tốt nhất) và **SDCA**,,.
*   **Tham số $m$ (độ dài vòng lặp con):** Thiết lập $m = 2n$ cho bài toán lồi và $m = 5n$ cho mạng thần kinh.
*   **Khởi tạo:** Chạy **1 vòng lặp SGD** cho bài toán lồi và **10 vòng lặp SGD** cho bài toán không lồi trước khi bắt đầu SVRG để có điểm bắt đầu tốt.
*   **Tiêu chí đánh giá:** Sử dụng số lần tính gradient chia cho $n$ làm trục hoành để đảm bảo so sánh công bằng về chi phí tính toán. Các chỉ số đánh giá gồm: Training loss residual ($P(w) - P(w^*)$), Test error rate và phương sai cập nhật (Update variance),,.