"""
svrg.py — SVRG Algorithm (Core Contribution)

Implements Algorithm 1 from Johnson & Zhang (NIPS 2013):
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"

Uses scalar storage optimization for linear prediction models:
precompute phi'(w_tilde^T x_i) for all i to reduce inner-loop cost.
"""

import numpy as np
from models.logistic import sigmoid


# ---------------------------------------------------------------------------
# SVRG for Binary Logistic Regression
# ---------------------------------------------------------------------------

def svrg_outer_loop_binary(w_tilde, X, y, lr, lam, m, option='I',
                           track_variance=True):
    """One outer iteration of SVRG for binary logistic regression.

    Uses scalar storage optimization: precompute phi'(w_tilde^T x_i) for all i
    to avoid recomputing inner products in the inner loop.

    Args:
        w_tilde: snapshot weights (d,)
        X: feature matrix (n, d) — dense or sparse CSR
        y: label vector (n,) with values in {-1, +1}
        lr: step size eta (constant)
        lam: L2 regularization strength
        m: number of inner loop iterations (typically 2n)
        option: 'I' (w_tilde = w_m) or 'II' (random w_t from history)
        track_variance: if True, compute and return gradient variance estimate

    Returns:
        updated w_tilde for next outer iteration
        (if track_variance, also returns variance estimate)
    """
    n = len(y)

    # ── Step 1: Compute full gradient mu_tilde ──
    # Precompute z_i = w_tilde^T x_i for all i
    z_tilde = X @ w_tilde                              # (n,)

    # phi'(z) for logistic loss: phi(z) = log(1 + exp(-y*z))
    # phi'(z) = -y * sigmoid(-y*z)
    phi_prime_tilde = -y * sigmoid(-y * z_tilde)       # (n,)

    # Full gradient: mu = (1/n) sum phi'_i(z_i) * x_i + lam * w_tilde
    mu = (X.T @ phi_prime_tilde) / n + lam * w_tilde   # (d,)

    # ── Step 2: Inner loop ──
    w = w_tilde.copy()

    if option == 'II':
        w_history = [w.copy()]

    # Variance tracking
    variance_sum = 0.0
    variance_count = 0

    for t in range(m):
        i = np.random.randint(n)
        xi = X[i]          # (d,) — dense 1D array
        yi = y[i]

        # nabla psi_i(w) = -y_i * sigmoid(-y_i * w^T x_i) * x_i + lam * w
        margin_w = yi * (xi @ w)
        g_current = (-yi * sigmoid(-margin_w)) * xi + lam * w

        # nabla psi_i(w_tilde) — using precomputed scalar phi'_i(z_i)
        # = phi'_i(z_i) * x_i + lam * w_tilde
        g_snapshot = phi_prime_tilde[i] * xi + lam * w_tilde

        # 1. Tính toán SVRG update direction 'v' như bạn đã làm
        v = g_current - g_snapshot + mu

        if track_variance:
            # Tính chính xác full gradient tại vị trí w_{t-1} hiện tại để làm kỳ vọng E[v]
            z_current_full = X @ w
            phi_prime_current_full = -y * sigmoid(-y * z_current_full)
            E_v = (X.T @ phi_prime_current_full) / n + lam * w  # Đây mới là E[v] chuẩn
            
            # Tính Variance của update (bao gồm cả learning rate lr giống bài báo)
            # Var(lr * v) = lr^2 * E[||v - E[v]||^2]
            diff = v - E_v
            # actual_update_variance = (lr ** 2) * np.dot(diff, diff)
            actual_update_variance = (lr ** 2) * np.sum(diff * diff)
            
            variance_sum += actual_update_variance
            variance_count += 1

        # SVRG update: w = w - lr * v
        w = w - lr * v

        if option == 'II':
            w_history.append(w.copy())

    # ── Step 3: Update snapshot ──
    if option == 'I':
        result = w
    else:  # Option II: random pick from {w_0, ..., w_{m-1}} per spec
        idx = np.random.randint(m)
        result = w_history[idx]

    if track_variance:
        variance_estimate = variance_sum / max(variance_count, 1)
        return result, variance_estimate
    return result


# ---------------------------------------------------------------------------
# SVRG for Multi-class Logistic Regression
# ---------------------------------------------------------------------------

# def svrg_outer_loop_multiclass(W_tilde, X, y, lr, lam, m, option='I',
#                                track_variance=True):
#     """One outer iteration of SVRG for multi-class logistic regression.

#     For multi-class with K classes, W is (d, K).
#     The scalar optimization still applies per class via precomputed probs.

#     Args:
#         W_tilde: snapshot weight matrix (d, K)
#         X: feature matrix (n, d)
#         y: label vector (n,) with values in {0, ..., K-1}
#         lr: step size eta
#         lam: L2 regularization
#         m: inner loop length
#         option: 'I' or 'II'
#         track_variance: if True, compute and return gradient variance estimate

#     Returns:
#         updated W_tilde
#         (if track_variance, also returns variance estimate)
#     """
def svrg_outer_loop_multiclass(W_tilde, X, y, lr, lam, m, track_variance=True, var_sample_rate=2000):
    """
    Một vòng lặp ngoài SVRG tối ưu hóa theo Chương 5 bài báo gốc.
    Áp dụng cho bài toán Multi-class (MNIST).
    
    var_sample_rate: Cứ sau N bước nhỏ mới tính phương sai chuẩn một lần (Tránh nghẽn cổ chai).
    """
    n, d = X.shape
    K = W_tilde.shape[1] # Số lượng class (MNIST = 10)

    # -------------------------------------------------------------------------
    # BƯỚC 1: VÒNG LẶP NGOÀI (Tính Full Gradient & Bộ đệm Scalar)
    # -------------------------------------------------------------------------
    logits_tilde = X @ W_tilde                                     # (n, K)
    logits_tilde -= np.max(logits_tilde, axis=1, keepdims=True)    # Ổn định số học
    exp_logits = np.exp(logits_tilde)
    
    # Bộ đệm xác suất tại snapshot (Scalar/Vector Storage Optimization theo Chương 5)
    probs_tilde = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (n, K)

    # Tạo ma trận One-hot để tính gradient nhanh
    one_hot = np.zeros((n, K))
    one_hot[np.arange(n), y] = 1.0

    # Full gradient mu chuẩn của vòng ngoài
    mu = (X.T @ (probs_tilde - one_hot)) / n + lam * W_tilde       # (d, K)

    # -------------------------------------------------------------------------
    # BƯỚC 2: VÒNG LẶP NỘI (Inner Loop - Cập nhật nhanh)
    # -------------------------------------------------------------------------
    W = W_tilde.copy()
    
    variance_sum = 0.0
    variance_count = 0

    for t in range(m):
        # Chọn mẫu ngẫu nhiên
        i = np.random.randint(n)
        xi = X[i]          # (d,)
        yi = y[i]

        # 2a. Tính nabla psi_i(W) tại trọng số hiện tại (Thời điểm t)
        logits_i = xi @ W  # (K,)
        logits_i -= np.max(logits_i)
        probs_i = np.exp(logits_i) / np.sum(np.exp(logits_i)) # (K,)
        
        e_yi = np.zeros(K)
        e_yi[yi] = 1.0
        g_current = np.outer(xi, probs_i - e_yi) + lam * W    # (d, K)

        # 2b. Tính nabla psi_i(W_tilde) - KHÔNG dùng phép nhân ma trận xi @ W_tilde
        # Lấy trực tiếp từ bộ đệm probs_tilde đã tính ở vòng ngoài (Tối ưu theo bài báo)
        g_snapshot = np.outer(xi, probs_tilde[i] - e_yi) + lam * W_tilde  # (d, K)

        # Hướng cập nhật SVRG
        v = g_current - g_snapshot + mu

        # 2c. Đo phương sai tối ưu (Theo yêu cầu thực nghiệm của bạn)
        if track_variance:
            # THAY ĐỔI CỐT LÕI: Chỉ tính Full Gradient để làm E[v] nếu thỏa mãn sample_rate
            # Giúp giảm chi phí tính toán từ 120,000 lần xuống còn 60 lần!
            if t % var_sample_rate == 0:
                logits_current_full = X @ W
                logits_current_full -= np.max(logits_current_full, axis=1, keepdims=True)
                exp_logits_c = np.exp(logits_current_full)
                probs_current_full = exp_logits_c / np.sum(exp_logits_c, axis=1, keepdims=True)
                
                # E_v thực tế tại thời điểm t
                E_v = (X.T @ (probs_current_full - one_hot)) / n + lam * W
                
                # Phương sai cập nhật thực tế: Var(lr * v)
                diff = v - E_v
                actual_update_variance = (lr ** 2) * np.sum(diff * diff)
                
                variance_sum += actual_update_variance
                variance_count += 1

        # Cập nhật trọng số SVRG
        W = W - lr * v

    # Theo chương 5: Trả về W cuối cùng hoạt động rất tốt (Option I)
    if track_variance:
        variance_estimate = variance_sum / max(variance_count, 1)
        return W, variance_estimate
    return W

# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def svrg_outer_loop(w, X, y, lr, lam, m, multiclass=False, option='I',
                    track_variance=True):
    """One outer iteration of SVRG.

    Args:
        w: snapshot weights (d,) for binary or (d, K) for multi-class
        X: feature matrix (n, d)
        y: label vector (n,)
        lr: step size
        lam: regularization
        m: inner loop length
        multiclass: multi-class flag
        option: 'I' or 'II'
        track_variance: if True, compute and return gradient variance estimate

    Returns:
        updated weights
        (if track_variance, also returns variance estimate)
    """
    if multiclass:
        return svrg_outer_loop_multiclass(w, X, y, lr, lam, m, option,
                                          track_variance)
    return svrg_outer_loop_binary(w, X, y, lr, lam, m, option,
                                  track_variance)


def effective_passes_svrg(n, m):
    """Compute effective passes for one SVRG outer iteration.

    Each outer iteration costs:
    - 1 pass for full gradient computation
    - m/n passes for inner loop

    Total = 1 + m/n

    Args:
        n: number of samples
        m: inner loop length

    Returns:
        effective passes for one outer iteration
    """
    return 1.0 + m / n
