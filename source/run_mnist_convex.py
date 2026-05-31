"""
run_mnist_convex.py — MNIST Convex Experiment Runner

Chạy 4 nhóm thuật toán trên MNIST (logistic regression, convex):
  1. SVRG              (lr = 0.025)
  2. SGD constant      (lr = 0.001, 0.0025, 0.005)
  3. SGD-best          (decaying schedule)
  4. SDCA

Metrics được lưu lại cho mỗi thuật toán:
  - training_loss      : P(w) tại mỗi effective pass
  - loss_residual      : P(w) - P(w*)
  - grad_variance      : E[||∇ψ_i(w) - ∇P(w)||²] — variance stochastic gradient (tất cả thuật toán)

Kết quả lưu vào: results/mnist_convex_results.json
"""

import os
import sys
import json
import numpy as np

np.random.seed(42)

# Đảm bảo import từ source/
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_dataset
from models.logistic import loss, full_grad, stoch_grad
from optimizers.sgd import sgd_epoch_constant, sgd_epoch_decay, warm_start
from optimizers.svrg import svrg_outer_loop, effective_passes_svrg
from optimizers.sdca import sdca_train, count_effective_passes_sdca

# ---------------------------------------------------------------------------
# Cấu hình thí nghiệm
# ---------------------------------------------------------------------------

DATASET = 'mnist'
LAM = 1e-4
MULTICLASS = True

# SVRG
SVRG_LR = 0.025
SVRG_M_FACTOR = 2       # m = 2 * n (convex)
SVRG_N_OUTER = 30

# SGD constant — 3 learning rates
SGD_CONST_LRS = [0.001, 0.0025, 0.005]
SGD_N_EPOCHS = 90

# SGD-best (decaying)
SGD_BEST_LR0 = 0.1
SGD_BEST_B = 1.0

# SDCA
SDCA_N_EPOCHS = 90

# Warm-start
WARM_START_EPOCHS = 1
WARM_START_LR = 0.01

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'mnist_convex_results.json')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_test_error(w, X_test, y_test):
    """Tính test error rate (%) cho multiclass MNIST."""
    preds = np.argmax(X_test @ w, axis=1)
    return float(np.mean(preds != y_test) * 100)


def _log_entry(results_key, results, passes, train_loss, P_star, variance=None):
    """Ghi một điểm log vào results dict."""
    results[results_key]['passes'].append(float(passes))
    results[results_key]['training_loss'].append(float(train_loss))
    results[results_key]['loss_residual'].append(float(train_loss - P_star))
    if variance is not None:
        results[results_key]['grad_variance'].append(float(variance))


VAR_N_SAMPLE = 500  # số điểm sample để ước tính variance


def estimate_grad_variance(w, X, y, lam, multiclass, n_sample=VAR_N_SAMPLE):
    """Ước tính E[||∇ψ_i(w) - ∇P(w)||²] bằng cách sample ngẫu nhiên.

    Đây là định nghĩa gradient variance chuẩn dùng để so sánh
    giữa các thuật toán (SGD có variance cao, SVRG giảm được variance).

    Args:
        w        : weight hiện tại
        X, y     : tập train
        lam      : regularization
        multiclass: flag
        n_sample : số sample để ước lượng (mặc định 500)

    Returns:
        scalar variance estimate
    """
    n = len(y)
    mu = full_grad(w, X, y, lam, multiclass)          # ∇P(w)
    indices = np.random.choice(n, size=min(n_sample, n), replace=False)
    var_sum = 0.0
    for i in indices:
        g_i = stoch_grad(w, X[i], y[i], lam, multiclass)  # ∇ψ_i(w)
        diff = g_i - mu
        var_sum += np.sum(diff * diff)
    return var_sum / len(indices)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_mnist_convex():
    print("=" * 60)
    print("MNIST Convex Experiment")
    print("=" * 60)

    # ── Load P(w*) ──
    optimal_path = os.path.join(RESULTS_DIR, 'optimal_loss.json')
    if not os.path.exists(optimal_path):
        print(f"ERROR: {optimal_path} not found. Hãy chạy compute_optimal.py trước.")
        return

    with open(optimal_path, 'r') as f:
        optimal = json.load(f)
    P_star = float(optimal[DATASET]['P_star'])
    print(f"P(w*) = {P_star:.10f}")

    # ── Load data ──
    X_train, y_train, X_test, y_test = load_dataset(DATASET)
    n, d = X_train.shape
    K = len(np.unique(y_train))
    print(f"Dataset: n={n}, d={d}, K={K}")

    # ── Warm-start (dùng chung cho tất cả thuật toán) ──
    print(f"\nWarm-start: {WARM_START_EPOCHS} epoch SGD (lr={WARM_START_LR})")
    w_init = warm_start(X_train, y_train, LAM, MULTICLASS,
                        n_epochs=WARM_START_EPOCHS, lr=WARM_START_LR)

    # ── Khởi tạo cấu trúc lưu kết quả ──
    results = {
        'dataset': DATASET,
        'P_star': P_star,
        'config': {
            'lam': LAM,
            'svrg_lr': SVRG_LR,
            'svrg_m_factor': SVRG_M_FACTOR,
            'sgd_const_lrs': SGD_CONST_LRS,
            'sgd_best_lr0': SGD_BEST_LR0,
            'sgd_best_b': SGD_BEST_B,
            'warm_start_epochs': WARM_START_EPOCHS,
        },
        # SVRG
        'svrg': {
            'passes': [], 'training_loss': [], 'loss_residual': [],
            'grad_variance': [],
        },
        # SGD constant — một entry cho mỗi lr
        'sgd_0.001':  {'passes': [], 'training_loss': [], 'loss_residual': [], 'grad_variance': []},
        'sgd_0.0025': {'passes': [], 'training_loss': [], 'loss_residual': [], 'grad_variance': []},
        'sgd_0.005':  {'passes': [], 'training_loss': [], 'loss_residual': [], 'grad_variance': []},
        # SGD-best
        'sgd_best': {'passes': [], 'training_loss': [], 'loss_residual': [], 'grad_variance': []},
        # SDCA
        'sdca': {'passes': [], 'training_loss': [], 'loss_residual': [], 'grad_variance': []},
    }

    # ====================================================================
    # 1. SVRG (lr = 0.025)
    # ====================================================================
    m = SVRG_M_FACTOR * n
    print(f"\n[1/4] SVRG  lr={SVRG_LR}  m={SVRG_M_FACTOR}*n={m}  outer={SVRG_N_OUTER}")

    w_svrg = w_init.copy()
    ep = float(WARM_START_EPOCHS)

    # Log điểm khởi đầu (sau warm-start)
    tl = loss(w_svrg, X_train, y_train, LAM, MULTICLASS)
    _log_entry('svrg', results, ep, tl, P_star, variance=0.0)

    for s in range(SVRG_N_OUTER):
        w_svrg, var = svrg_outer_loop(
            w_svrg, X_train, y_train,
            lr=SVRG_LR, lam=LAM, m=m,
            multiclass=MULTICLASS, option='I', track_variance=True,
        )
        ep += effective_passes_svrg(n, m)
        tl = loss(w_svrg, X_train, y_train, LAM, MULTICLASS)
        _log_entry('svrg', results, ep, tl, P_star, variance=var)

        if (s + 1) % 5 == 0:
            print(f"  outer {s+1:2d}: loss={tl:.6f}  residual={tl-P_star:.2e}  var={var:.2e}")

    # ====================================================================
    # 2. SGD constant  (lr = 0.001, 0.0025, 0.005)
    # ====================================================================
    for lr in SGD_CONST_LRS:
        key = f'sgd_{lr}'
        print(f"\n[2/4] SGD constant  lr={lr}  epochs={SGD_N_EPOCHS}")

        w_sgd = w_init.copy()
        ep = float(WARM_START_EPOCHS)

        tl = loss(w_sgd, X_train, y_train, LAM, MULTICLASS)
        var = estimate_grad_variance(w_sgd, X_train, y_train, LAM, MULTICLASS)
        _log_entry(key, results, ep, tl, P_star, variance=var)

        for epoch in range(SGD_N_EPOCHS):
            w_sgd = sgd_epoch_constant(w_sgd, X_train, y_train,
                                       lr=lr, lam=LAM, multiclass=MULTICLASS)
            ep += 1.0
            tl = loss(w_sgd, X_train, y_train, LAM, MULTICLASS)
            var = estimate_grad_variance(w_sgd, X_train, y_train, LAM, MULTICLASS)
            _log_entry(key, results, ep, tl, P_star, variance=var)

        print(f"  done. final residual = {results[key]['loss_residual'][-1]:.2e}")

    # ====================================================================
    # 3. SGD-best (decaying schedule)
    # ====================================================================
    print(f"\n[3/4] SGD-best  lr0={SGD_BEST_LR0}  b={SGD_BEST_B}  epochs={SGD_N_EPOCHS}")

    w_sgd_best = w_init.copy()
    ep = float(WARM_START_EPOCHS)
    t = 0   # total gradient steps (for decay schedule)

    tl = loss(w_sgd_best, X_train, y_train, LAM, MULTICLASS)
    var = estimate_grad_variance(w_sgd_best, X_train, y_train, LAM, MULTICLASS)
    _log_entry('sgd_best', results, ep, tl, P_star, variance=var)

    for epoch in range(SGD_N_EPOCHS):
        w_sgd_best, t = sgd_epoch_decay(
            w_sgd_best, X_train, y_train,
            lr0=SGD_BEST_LR0, lam=LAM, n=n,
            t_start=t, b=SGD_BEST_B, multiclass=MULTICLASS,
        )
        ep += 1.0
        tl = loss(w_sgd_best, X_train, y_train, LAM, MULTICLASS)
        var = estimate_grad_variance(w_sgd_best, X_train, y_train, LAM, MULTICLASS)
        _log_entry('sgd_best', results, ep, tl, P_star, variance=var)

    print(f"  done. final residual = {results['sgd_best']['loss_residual'][-1]:.2e}")

    # ====================================================================
    # 4. SDCA
    # ====================================================================
    print(f"\n[4/4] SDCA  epochs={SDCA_N_EPOCHS}")

    ep_sdca = float(WARM_START_EPOCHS)
    # Log điểm khởi đầu bằng w_init
    tl_init = loss(w_init.copy(), X_train, y_train, LAM, MULTICLASS)
    var_init = estimate_grad_variance(w_init, X_train, y_train, LAM, MULTICLASS)
    _log_entry('sdca', results, ep_sdca, tl_init, P_star, variance=var_init)

    def sdca_callback(w_curr, epoch):
        ep_curr = float(WARM_START_EPOCHS) + count_effective_passes_sdca(epoch + 1)
        tl = loss(w_curr, X_train, y_train, LAM, MULTICLASS)
        var = estimate_grad_variance(w_curr, X_train, y_train, LAM, MULTICLASS)
        _log_entry('sdca', results, ep_curr, tl, P_star, variance=var)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}: loss={tl:.6f}  residual={tl-P_star:.2e}  var={var:.2e}")

    sdca_train(X_train, y_train, LAM, SDCA_N_EPOCHS,
               multiclass=MULTICLASS, callback=sdca_callback)

    print(f"  done. final residual = {results['sdca']['loss_residual'][-1]:.2e}")

    # ── Lưu kết quả ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Kết quả đã lưu vào {RESULTS_FILE}")

    return results


if __name__ == '__main__':
    run_mnist_convex()
