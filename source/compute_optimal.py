"""
compute_optimal.py — Estimate P(w*) via Full Gradient Descent

Computes a high-precision estimate of the optimal loss P(w*) for each
(dataset, λ) combination. Used as baseline for loss residual P(w) - P(w*)
in all plots.
"""

import os
import json
import numpy as np
from utils.data_loader import load_dataset
from models.logistic import loss_binary, full_grad_binary, loss_multiclass, full_grad_multiclass


def gd_solve(X, y, lam, multiclass=False, max_iter=10000, tol=1e-14, verbose=True):
    """Run full-batch Gradient Descent to find P(w*).

    Args:
        X: feature matrix (n, d)
        y: label vector (n,)
        lam: L2 regularization strength
        multiclass: whether to use multi-class logistic regression
        max_iter: maximum GD iterations
        tol: stop when relative loss change < tol
        verbose: print progress

    Returns:
        w_star: optimal weights
        P_star: optimal loss value
        loss_history: list of loss values per iteration
    """
    n, d = X.shape
    if multiclass:
        K = len(np.unique(y))
        w = np.zeros((d, K))
        loss_fn = loss_multiclass
        grad_fn = full_grad_multiclass
    else:
        w = np.zeros(d)
        loss_fn = loss_binary
        grad_fn = full_grad_binary

    loss_history = []
    grad_norm_history = []
    eta = 1.0  # initial guess
    c = 0.5   # Armijo parameter

    for t in range(max_iter):
        g = grad_fn(w, X, y, lam)
        grad_norm = np.linalg.norm(g)
        current_loss = loss_fn(w, X, y, lam)

        # Armijo line search
        eta_current = eta
        for _ in range(30):
            w_new = w - eta_current * g
            new_loss = loss_fn(w_new, X, y, lam)
            # Armijo condition: f(w - ηg) ≤ f(w) - c·η·||g||²
            if new_loss <= current_loss - c * eta_current * (grad_norm ** 2):
                w = w_new
                eta = eta_current * 1.01  # Slightly increase for next step
                break
            eta_current *= 0.5
        else:
            # Line search failed completely
            if verbose:
                print(f"  Warning: line search failed at iter {t}")
            break

        if t % 100 == 0 or t == max_iter - 1:
            loss_history.append(current_loss)
            grad_norm_history.append(grad_norm)
            if verbose:
                print(f"  iter {t:6d}: loss = {current_loss:.12f}, |grad| = {grad_norm:.2e}, eta = {eta_current:.2e}")

        if grad_norm < tol:
            if verbose:
                print(f"  Converged at iter {t}")
            break

    P_star = loss_fn(w, X, y, lam)
    return w, P_star, loss_history


def compute_all_optimal():
    """Compute P(w*) for all 4 datasets with their respective λ values."""
    configs = [
        # (dataset_name, lam, multiclass)
        ('mnist',   1e-4, True),
        ('cifar10', 1e-3, True),
        ('rcv1',    1e-5, False),
        ('covtype', 1e-5, False),
    ]

    results = {}

    for name, lam, multiclass in configs:
        print(f"\n{'='*60}")
        print(f"Computing P(w*) for {name} (lam={lam}, multiclass={multiclass})")
        print(f"{'='*60}")

        X_train, y_train, _, _ = load_dataset(name)
        n, d = X_train.shape
        print(f"  Dataset: n={n}, d={d}")

        # Adjust learning rate based on dataset size
        if name == 'covtype':
            lr = 0.01   # Large n, need smaller step
        elif name == 'cifar10':
            lr = 0.01   # Multiclass on CIFAR-10 can be unstable with 0.1
        elif name == 'rcv1':
            lr = 0.05
        else:
            lr = 0.1

        w_star, P_star, loss_hist = gd_solve(
            X_train, y_train, lam,
            multiclass=multiclass,
            max_iter=10000,
            verbose=True,
        )

        results[name] = {
            'P_star': float(P_star),
            'lam': lam,
            'multiclass': multiclass,
            'n': n,
            'd': d,
            'loss_history': [float(v) for v in loss_hist],
            'final_loss': float(loss_hist[-1]),
        }

        print(f"  [V] P(w*) = {P_star:.12f}")

    return results


def save_optimal(results, filepath='results/optimal_loss.json'):
    """Save P(w*) results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved optimal losses to {filepath}")


def load_optimal(filepath='results/optimal_loss.json'):
    """Load P(w*) results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_one(name, lam, multiclass):
    X_train, y_train, _, _ = load_dataset(name)

    if name == 'covtype':
        lr = 0.01
    elif name == 'cifar10':
        lr = 0.01
    elif name == 'rcv1':
        lr = 0.05
    else:
        lr = 0.1

    w_star, P_star, loss_hist = gd_solve(
        X_train, y_train, lam,
        multiclass=multiclass,
        max_iter=10000,
        verbose=True,
    )

    return {
        'P_star': float(P_star),
        'lam': lam,
        'multiclass': multiclass,
        'n': X_train.shape[0],
        'd': X_train.shape[1],
        'loss_history': [float(v) for v in loss_hist],
        'final_loss': float(loss_hist[-1]),
    }

if __name__ == '__main__':
    results = compute_all_optimal()
    save_optimal(results)

# if __name__ == '__main__':
#     filepath = 'results/optimal_loss.json'

#     # Load file cũ (nếu có)
#     if os.path.exists(filepath):
#         results = load_optimal(filepath)
#     else:
#         results = {}

#     # 👇 CHỈ ĐỔI TÊN DATASET BỊ SAI Ở ĐÂY
#     name = 'covtype'   # ví dụ: mnist / cifar10 / rcv1 / covtype

#     configs = {
#         'mnist':   (1e-4, True),
#         'cifar10': (1e-3, True),
#         'rcv1':    (1e-5, False),
#         'covtype': (1e-5, False),
#     }

#     lam, multiclass = configs[name]

#     # 👇 chỉ chạy lại 1 dataset
#     results[name] = compute_one(name, lam, multiclass)

#     # save lại file
#     save_optimal(results, filepath)