"""
run_sgd_const_mnist_convex.py — SGD Constant Runner for MNIST Convex
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_dataset
from models.logistic import loss, full_grad, stoch_grad
from optimizers.sgd import sgd_epoch_constant

# --- Cấu hình ---
DATASET = 'mnist'
LAM = 1e-4
MULTICLASS = True
WARM_START_EPOCHS = 1
SGD_CONST_LRS = [0.001, 0.0025, 0.005]
SGD_N_EPOCHS = 90
VAR_N_SAMPLE = 500

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'sgd_const_mnist_convex.json')
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
    print("MNIST Convex: SGD Constant Experiment")
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
    results = {}

    # 5. Run SGD Constant cho từng LR
    for lr in SGD_CONST_LRS:
        np.random.seed(42) # Set seed for fair comparison within algorithms
        key = f'sgd_{lr}'
        results[key] = {'passes': [], 'training_loss': [], 'loss_residual': [], 'grad_variance': []}
        
        print(f"\nSGD constant  lr={lr}  epochs={SGD_N_EPOCHS}")

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

    # 6. Lưu kết quả
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Kết quả đã lưu vào {RESULTS_FILE}")

if __name__ == '__main__':
    main()
