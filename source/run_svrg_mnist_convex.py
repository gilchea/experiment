"""
run_svrg_mnist_convex.py — SVRG Runner for MNIST Convex
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_dataset
from models.logistic import loss, full_grad, stoch_grad
from optimizers.svrg import svrg_outer_loop, effective_passes_svrg

# --- Cấu hình ---
DATASET = 'mnist'
LAM = 1e-4
MULTICLASS = True
WARM_START_EPOCHS = 1
SVRG_LR = 0.025
SVRG_M_FACTOR = 2       # m = 2 * n (convex)
SVRG_N_OUTER = 30
VAR_N_SAMPLE = 500

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'svrg_mnist_convex.json')
W_INIT_PATH = os.path.join(RESULTS_DIR, 'w_init_mnist.npy')
OPTIMAL_PATH = os.path.join(RESULTS_DIR, 'optimal_loss.json')

# --- Helpers ---
def _log_entry(results_key, results, passes, train_loss, P_star, variance=None):
    results[results_key]['passes'].append(float(passes))
    results[results_key]['training_loss'].append(float(train_loss))
    results[results_key]['loss_residual'].append(float(train_loss - P_star))
    if variance is not None:
        results[results_key]['grad_variance'].append(float(variance))

def estimate_grad_variance(w, X, y, lam, multiclass, n_sample=VAR_N_SAMPLE):
    n = len(y)
    mu = full_grad(w, X, y, lam, multiclass)
    indices = np.random.choice(n, size=min(n_sample, n), replace=False)
    var_sum = 0.0
    for i in indices:
        g_i = stoch_grad(w, X[i], y[i], lam, multiclass)
        diff = g_i - mu
        var_sum += np.sum(diff * diff)
    return var_sum / len(indices)

def main():
    print("=" * 60)
    print("MNIST Convex: SVRG Experiment")
    print("=" * 60)

    # 1. Load P*
    if not os.path.exists(OPTIMAL_PATH):
        print(f"ERROR: {OPTIMAL_PATH} not found.")
        return
    with open(OPTIMAL_PATH, 'r') as f:
        optimal = json.load(f)
    P_star = float(optimal[DATASET]['P_star'])

    # 2. Load data
    X_train, y_train, _, _ = load_dataset(DATASET)
    n, _ = X_train.shape

    # 3. Load w_init
    if not os.path.exists(W_INIT_PATH):
        print(f"ERROR: {W_INIT_PATH} not found. Run prepare_mnist_convex.py first.")
        return
    w_init = np.load(W_INIT_PATH)

    # 4. Cấu trúc kết quả
    results = {
        'svrg': {
            'passes': [], 'training_loss': [], 'loss_residual': [],
            'grad_variance': [],
        }
    }

    # 5. Run SVRG
    np.random.seed(42) # Set seed for fair comparison within algorithms
    m = SVRG_M_FACTOR * n
    print(f"\nSVRG  lr={SVRG_LR}  m={SVRG_M_FACTOR}*n={m}  outer={SVRG_N_OUTER}")

    w_svrg = w_init.copy()
    ep = float(WARM_START_EPOCHS)

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

    # 6. Lưu kết quả
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Kết quả đã lưu vào {RESULTS_FILE}")

if __name__ == '__main__':
    main()
