"""
compute_optimal.py — Estimate P(w*) via Full-Batch Gradient Descent

Computes a high-precision estimate of the optimal loss P(w*) for each
(dataset, λ) combination using full-batch GD with Armijo line search.
The resulting values are saved to JSON and used as the baseline for
loss-residual plots: P(w) - P(w*).

Usage:
    # Recompute a single dataset (edit `name` at the bottom):
    python compute_optimal.py
"""

import os
import json

import numpy as np

from utils.data_loader import load_dataset
from models.logistic import (
    loss_binary, full_grad_binary,
    loss_multiclass, full_grad_multiclass,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    'mnist':   {'lam': 1e-4, 'multiclass': True},
    'cifar10': {'lam': 1e-3, 'multiclass': True},
    'rcv1':    {'lam': 1e-5, 'multiclass': False},
    'covtype': {'lam': 1e-5, 'multiclass': False},
}

DEFAULT_OUTPUT_PATH = 'results/optimal_loss.json'


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def gd_solve(X, y, lam, multiclass=False, max_iter=10_000, tol=1e-14, verbose=True):
    """Run full-batch gradient descent with Armijo line search to find P(w*).

    Args:
        X:          feature matrix (n, d)
        y:          label vector (n,)
        lam:        L2 regularization strength
        multiclass: use multi-class logistic regression if True
        max_iter:   maximum number of GD iterations
        tol:        stop when gradient norm < tol
        verbose:    print progress every 100 iterations

    Returns:
        w_star:       optimal weight vector/matrix
        P_star:       optimal loss value
        loss_history: list of loss values recorded every 100 iterations
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
    eta = 1.0           # initial step size
    armijo_c = 0.5      # Armijo sufficient-decrease constant

    for t in range(max_iter):
        g = grad_fn(w, X, y, lam)
        grad_norm = np.linalg.norm(g)
        current_loss = loss_fn(w, X, y, lam)

        # Armijo backtracking line search
        eta_t = eta
        for _ in range(30):
            w_new = w - eta_t * g
            if loss_fn(w_new, X, y, lam) <= current_loss - armijo_c * eta_t * grad_norm ** 2:
                w = w_new
                eta = eta_t * 1.01   # slightly increase for the next step
                break
            eta_t *= 0.5
        else:
            if verbose:
                print(f"  Warning: Armijo line search failed at iter {t}")
            break

        if t % 100 == 0 or t == max_iter - 1:
            loss_history.append(current_loss)
            if verbose:
                print(
                    f"  iter {t:6d}: loss = {current_loss:.12f}, "
                    f"|grad| = {grad_norm:.2e}, eta = {eta_t:.2e}"
                )

        if grad_norm < tol:
            if verbose:
                print(f"  Converged at iter {t}")
            break

    P_star = loss_fn(w, X, y, lam)
    return w, P_star, loss_history


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def compute_one(name):
    """Compute P(w*) for a single dataset.

    Args:
        name: dataset key in DATASET_CONFIGS

    Returns:
        dict with P_star, lam, multiclass, n, d, loss_history, final_loss
    """
    cfg = DATASET_CONFIGS[name]
    lam, multiclass = cfg['lam'], cfg['multiclass']

    print(f"\n{'='*60}")
    print(f"Computing P(w*) for {name}  (lam={lam}, multiclass={multiclass})")
    print(f"{'='*60}")

    X_train, y_train, _, _ = load_dataset(name)
    n, d = X_train.shape
    print(f"  Dataset: n={n}, d={d}")

    _, P_star, loss_hist = gd_solve(
        X_train, y_train, lam,
        multiclass=multiclass,
        max_iter=10_000,
        verbose=True,
    )
    print(f"  [✓] P(w*) = {P_star:.12f}")

    return {
        'P_star':       float(P_star),
        'lam':          lam,
        'multiclass':   multiclass,
        'n':            n,
        'd':            d,
        'loss_history': [float(v) for v in loss_hist],
        'final_loss':   float(loss_hist[-1]),
    }


def compute_all():
    """Compute P(w*) for all datasets defined in DATASET_CONFIGS."""
    return {name: compute_one(name) for name in DATASET_CONFIGS}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_optimal(results, filepath=DEFAULT_OUTPUT_PATH):
    """Save P(w*) results to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved optimal losses → {filepath}")


def load_optimal(filepath=DEFAULT_OUTPUT_PATH):
    """Load P(w*) results from a JSON file."""
    with open(filepath) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point — recompute a single dataset
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Load existing results (if any) so other datasets are preserved.
    results = load_optimal() if os.path.exists(DEFAULT_OUTPUT_PATH) else {}

    # ── Change this to the dataset you want to recompute ──
    name = 'covtype'   # one of: mnist / cifar10 / rcv1 / covtype

    results[name] = compute_one(name)
    save_optimal(results)