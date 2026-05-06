# Kế hoạch Triển khai – Tái lập Thí nghiệm NIPS-2013 (SVRG)

> **Paper**: *Accelerating Stochastic Gradient Descent using Predictive Variance Reduction* – Johnson & Zhang, NIPS 2013

---

## 0. Tổng quan Thực nghiệm của Paper

Paper có **2 nhóm thực nghiệm** trên **4 bộ dữ liệu**:

| Nhóm | Mô hình | Datasets |
|---|---|---|
| **Lồi (Convex)** | L2-regularized Logistic Regression | MNIST, rcv1, covtype,  CIFAR-10 |
| **Không lồi (Non-convex)** | Neural Network 1 lớp ẩn (100 nút), Sigmoid + Softmax | MNIST, CIFAR-10 |

---

## 1. Cấu trúc thư mục

```
experiment/
├── data/                         ✅ đã có
│   ├── rcv1/rcv1_train.binary/
│   ├── covtype/covtype.libsvm.binary/
│   ├── mnist/
│   └── cifar-10-python/
│
├── models/
│   └── logistic.py               ← L2 logistic regression (convex)
│
├── optimizers/
│   ├── sgd.py                    ← SGD (constant η + decaying η)
│   └── svrg.py                   ← SVRG Algorithm 1 (paper gốc)
│
├── utils/
│   ├── data_loader.py            ← load 4 datasets → numpy/sparse
│   └── metrics.py                ← tính P(w) và P(w*) trên toàn dataset
│
├── config.py                     ← TẤT CẢ hyperparameters từ paper
├── train.py                      ← runner chính: chạy tất cả thuật toán
├── compute_optimal.py            ← chạy GD để tìm P(w*)
└── plot_results.py               ← vẽ Figure 1, 2, 3 giống paper
```

---

## 2. Bảng Hyperparameters Chính Xác Từ Paper

### 2A. Logistic Regression (Convex)

| Dataset | λ | η (SVRG) | m | Warm-start |
|---|---|---|---|---|
| MNIST | 10⁻⁴ | ~0.025 | 2n | 1 epoch SGD |
| rcv1.binary | 10⁻⁵ | tuned | 2n | 1 epoch SGD |
| covtype.binary | 10⁻⁵ | tuned | 2n | 1 epoch SGD |
| CIFAR-10 | 10⁻³ | tuned | 2n | 1 epoch SGD |

### 2B. Neural Network (Non-convex)

| Dataset | λ | m | Mini-batch | Warm-start |
|---|---|---|---|---|
| MNIST | 10⁻⁴ | 5n | 10 | **10 epoch SGD** |
| CIFAR-10 | 10⁻³ | 5n | 10 | **10 epoch SGD** |

### 2C. Tiền xử lý Dataset

| Bộ dữ liệu | Loại bài toán | Đặc điểm nổi bật | Tiền xử lý |
| :--- | :--- | :--- | :--- |
| **MNIST** | Phân loại chữ số | 10 lớp, hình ảnh xám | Chia cho 255 để đưa về khoảng $[0, 1]$. |
| **CIFAR-10** | Phân loại vật thể | 10 lớp, hình ảnh màu | Chia cho 255 để đưa về khoảng $[0, 1]$. |
| **rcv1.binary** | Phân loại văn bản | Dữ liệu thưa (sparse), nhị phân | Giữ nguyên định dạng thưa để tối ưu bộ nhớ. |
| **covtype.binary**| Thảm thực vật | Nhị phân | Chia đôi ngẫu nhiên (Train/Test). |

---

## 3. Baselines Phải So Sánh

| Baseline | Mô tả |
|---|---|
| **SGD (Constant η)** | SGD với learning rate cố định tốt nhất tìm được |
| **SGD (Best / Decaying)** | SGD với schedule $\eta_t = \eta_0(1 + a\eta_0 t)^{-1}$, tinh chỉnh kỹ |
| **SDCA** | Stochastic Dual Coordinate Ascent (cho bài toán lồi) |
| **SVRG** | Thuật toán của paper |

> **Chú ý**: "SGD-best" trong paper nghĩa là SGD được tune CỰC KỲ kỹ. Cần cài cả 2 variant (constant + decaying).

---

## 4. Metrics Cần Đo

| Metric | Công thức | Ghi chú |
|---|---|---|
| **Loss Residual** (chính) | $P(w) - P(w^*)$ | $P(w^*)$ ước tính bằng GD rất lâu |
| **Test Error Rate** | % lỗi trên test set | |
| **Update Variance** | $\text{Var}[\nabla\psi_{i_t}(w) - \nabla\psi_{i_t}(\tilde{w}) + \tilde{\mu}]$ | Hình 4 trong paper |
| **Trục X** | Tổng gradient evaluations / n | **Effective passes** |

> **Critical**: Loss residual đòi hỏi phải biết $P(w^*)$. Chạy GD nhiều iterations trước để estimate.

---

## 5. Chi tiết Cài đặt Từng Thành Phần

### Bước 1: `utils/data_loader.py`

```python
# Interface
X, y = load_rcv1()          # X: sparse (n, 47236), y: {-1, +1}
X, y = load_covtype()       # X: dense (n, 54), y: {-1, +1}, đã split 50/50
X, y = load_mnist()         # X: (60000, 784) / 255, y: {0..9}
X, y = load_cifar10()       # X: (50000, 3072) / 255, y: {0..9}
```

- [ ] RCV1: `load_svmlight_file`, labels ∈ {-1, +1} → giữ nguyên sparse
- [ ] Covtype: load, binarize (class 1 → +1, rest → -1), split 50/50 random seed 42
- [ ] MNIST: load binary files từ `data/mnist/`, chia 255
- [ ] CIFAR-10: load pickle batches, chia 255

### Bước 2: `models/logistic.py`

Hàm mục tiêu paper:
$$P(w) = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i w^\top x_i)) + \frac{\lambda}{2}\|w\|^2$$

```python
def loss(w, X, y, lam):
    """Full objective P(w) trên toàn bộ dataset"""
    margins = y * (X @ w)
    return np.mean(np.log1p(np.exp(-margins))) + 0.5 * lam * np.dot(w, w)

def full_grad(w, X, y, lam):
    """Gradient ∇P(w) = (1/n) Σ ∇ψ_i(w) + λw"""
    n = len(y)
    margins = y * (X @ w)
    coefs = -y * sigmoid(-margins)          # shape (n,)
    return X.T @ coefs / n + lam * w

def stoch_grad(w, xi, yi, lam):
    """Gradient ∇ψ_i(w) tại sample i (bao gồm regularizer)"""
    margin = yi * (xi @ w)
    coef = -yi * sigmoid(-margin)
    return coef * xi + lam * w
```

> **Lưu ý**: Dùng `np.log1p(np.exp(-x))` thay `log(1+exp(-x))` để tránh overflow.

### Bước 3: `optimizers/sgd.py`

```python
def sgd_epoch(w, X, y, lr, lam):
    """1 epoch SGD (constant lr), trả về w mới"""
    n = len(y)
    indices = np.random.permutation(n)
    for i in indices:
        g = stoch_grad(w, X[i], y[i], lam)
        w = w - lr * g
    return w

def sgd_epoch_decay(w, X, y, lr0, lam, t, a):
    """1 epoch SGD với decaying lr: η_t = η_0 / (1 + a*η_0*t/n)"""
    ...
```

### Bước 4: `optimizers/svrg.py` ← **Quan trọng nhất**

Implement chính xác **Algorithm 1** của paper:

```python
def svrg_outer_loop(w_tilde, X, y, lr, lam, m):
    """
    1 outer iteration của SVRG.
    Chi phí: n (full grad) + 2m (inner loop) gradient evaluations
    = (1 + 2m/n) effective passes
    """
    n = len(y)

    # --- Bước 1: Tính full gradient tại snapshot ---
    mu = full_grad(w_tilde, X, y, lam)        # O(n) evals

    # --- Bước 2: Inner loop m bước ---
    w = w_tilde.copy()
    for t in range(m):
        i = np.random.randint(n)
        g_current  = stoch_grad(w, X[i], y[i], lam)
        g_snapshot = stoch_grad(w_tilde, X[i], y[i], lam)
        # Bước cập nhật variance-reduced
        w = w - lr * (g_current - g_snapshot + mu)

    # Option I: w̃_next = w_m
    return w
```

**Tính toán effective passes** cho mỗi outer iteration:
$$\text{passes} = \underbrace{1}_{\text{full grad}} + \underbrace{m/n}_{\text{inner steps}} = 1 + 2 = 3 \quad (\text{khi } m=2n)$$

### Bước 5: `compute_optimal.py`

Chạy GD (full batch) nhiều iteration để ước tính $P(w^*)$:

```python
# Gradient Descent để tìm w*
w_star = np.zeros(d)
for _ in range(1000):
    g = full_grad(w_star, X, y, lam)
    w_star -= lr_gd * g
P_star = loss(w_star, X, y, lam)
```

### Bước 6: `train.py`

```python
results = {}

# 1. Warm-start: 1 epoch SGD
w0 = np.zeros(d)
w0 = sgd_epoch(w0, X, y, lr=0.01, lam=lam)
effective_passes = 1.0

# 2. Chạy SVRG
w = w0.copy()
log_passes, log_loss = [effective_passes], [loss(w, X, y, lam) - P_star]
for s in range(n_outer):
    w = svrg_outer_loop(w, X, y, lr=lr_svrg, lam=lam, m=m)
    effective_passes += 1 + m/n    # ← đếm ĐÚNG
    log_loss.append(loss(w, X, y, lam) - P_star)
    log_passes.append(effective_passes)

# 3. Chạy SGD baselines (cùng số passes)
...
```

### Bước 7: `plot_results.py`

```python
plt.semilogy(passes_svrg, loss_svrg, label='SVRG')
plt.semilogy(passes_sgd_const, loss_sgd_const, label='SGD (const η)')
plt.semilogy(passes_sgd_best, loss_sgd_best, label='SGD-best')
plt.semilogy(passes_sdca, loss_sdca, label='SDCA')
plt.xlabel('Number of effective passes')
plt.ylabel('Training loss residual P(w) - P(w*)')
plt.legend(); plt.savefig('figures/figure1_rcv1.png')
```

---

## 6. Thứ tự Triển khai

```
Bước 1: utils/data_loader.py
   → verify: load rcv1, shape (20242, 47236), labels in {-1,+1}

Bước 2: models/logistic.py
   → verify: numerical gradient check pass

Bước 3: compute_optimal.py
   → verify: GD loss giảm đều, lưu P_star

Bước 4: optimizers/sgd.py
   → verify: loss giảm sau 5 epochs trên rcv1

Bước 5: optimizers/svrg.py
   → verify: loss giảm NHANH hơn SGD, đường thẳng trên log scale

Bước 6: train.py (runner đầy đủ)
   → verify: effective_passes đếm đúng, kết quả lưu vào results/

Bước 7: plot_results.py
   → verify: figure trông giống Figure 1 & 2 trong paper PDF
```

---

## 7. Những Điểm Hay Bị Sai

| Bẫy | Giải thích | Fix |
|---|---|---|
| **Không warm-start** | SVRG hội tụ chậm nếu bắt đầu xa w* | Chạy 1 epoch SGD trước |
| **Đếm sai effective passes** | Mỗi outer iteration SVRG = 3 passes (m=2n), không phải 1 | `passes += 1 + m/n` |
| **Không tính P(w*)** | Loss residual cần P(w*) để vẽ | Chạy GD riêng để estimate |
| **Sparse matrix với numpy** | RCV1 sparse, `X @ w` cần dùng `.dot()` hoặc scipy | Kiểm tra type trước |
| **Gradient regularizer** | λw phải có trong MỌI gradient call (full và stochastic) | Không bỏ sót |
| **Option I vs II** | Cài Option I (w̃ = w_m) cho convex theo paper | `return w` ở cuối inner loop |

---

## 8. Checklist Kết quả (Verify giống paper)

- [ ] **rcv1**: SVRG loss residual giảm theo đường thẳng trên log scale
- [ ] **covtype**: SVRG cạnh tranh với SDCA
- [ ] SGD constant η → loss "plateau" (không về 0 vì variance không giảm)
- [ ] SGD-best → giảm nhưng chậm hơn SVRG rõ rệt
- [ ] Trục X = effective passes, không phải epochs
- [ ] Y-axis = $P(w) - P(w^*)$ (log scale)

