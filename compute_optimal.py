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


def gd_solve(X, y, lam, multiclass=False, max_iter=2000, lr=0.1, tol=1e-14, verbose=True):
    """Run full-batch Gradient Descent to find P(w*).

    Args:
        X: feature matrix (n, d)
        y: label vector (n,)
        lam: L2 regularization strength
        multiclass: whether to use multi-class logistic regression
        max_iter: maximum GD iterations
        lr: learning rate (constant)
        tol: stop when relative loss change < tol
        verbose: print progress

    Returns:
        w_star: optimal weights
        P_star: optimal loss value
        loss_history: list of loss values per iteration
    """
    n, d = X.shape
    K = len(np.unique(y)) if multiclass else 1

    if multiclass:
        w = np.zeros((d, K))
        loss_fn = loss_multiclass
        grad_fn = full_grad_multiclass
    else:
        w = np.zeros(d)
        loss_fn = loss_binary
        grad_fn = full_grad_binary

    loss_history = []

    for t in range(max_iter):
        g = grad_fn(w, X, y, lam)
        w = w - lr * g

        if t % 100 == 0 or t == max_iter - 1:
            current_loss = loss_fn(w, X, y, lam)
            loss_history.append(current_loss)

            if verbose:
                print(f"  iter {t:4d}: loss = {current_loss:.12f}")

            # Check convergence
            if len(loss_history) >= 2:
                rel_change = abs(loss_history[-1] - loss_history[-2]) / max(1, abs(loss_history[-2]))
                if rel_change < tol:
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
        print(f"Computing P(w*) for {name} (λ={lam}, multiclass={multiclass})")
        print(f"{'='*60}")

        X_train, y_train, _, _ = load_dataset(name)
        n, d = X_train.shape
        print(f"  Dataset: n={n}, d={d}")

        # Adjust learning rate based on dataset size
        if name == 'covtype':
            lr = 0.01   # Large n, need smaller step
        elif name == 'rcv1':
            lr = 0.05
        else:
            lr = 0.1

        w_star, P_star, loss_hist = gd_solve(
            X_train, y_train, lam,
            multiclass=multiclass,
            max_iter=2000,
            lr=lr,
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

        print(f"  ✓ P(w*) = {P_star:.12f}")

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


if __name__ == '__main__':
    results = compute_all_optimal()
    save_optimal(results)
