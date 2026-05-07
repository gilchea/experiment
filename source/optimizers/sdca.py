"""
sdca.py — Stochastic Dual Coordinate Ascent (SDCA)

Implements SDCA as described in PROCEDURE_SDCA.md (Section 4, NIPS 2013 SVRG paper),
viewed as a variance-reduction method maintaining per-sample dual variables.

Primal problem:
    min_w P(w) = (1/n) Σ φ_i(w) + (λ/2) ||w||²

Dual variables: α_i ∈ ℝ^d  for i = 1, ..., n
Primal relationship:
    w⁽⁰⁾ = Σ α_i⁽⁰⁾  (= 0 at initialization)

Main loop (per step t, sample i):
    α_i⁽ᵗ⁾ = α_i⁽ᵗ⁻¹⁾ - η_t * (∇φ_i(w⁽ᵗ⁻¹⁾) + λn · α_i⁽ᵗ⁻¹⁾)
    w⁽ᵗ⁾  = w⁽ᵗ⁻¹⁾ + (α_i⁽ᵗ⁾ - α_i⁽ᵗ⁻¹⁾)
"""

import numpy as np
from models.logistic import sigmoid


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def _grad_logistic_binary(w, xi, yi):
    """∇φ_i(w) for binary logistic loss: φ_i(w) = log(1 + exp(-y_i w^T x_i)).

    ∇φ_i(w) = -y_i * (1 - σ(y_i w^T x_i)) * x_i
             = (σ(w^T x_i) - [y_i == +1]) * x_i   (equivalent form)
    """
    margin = yi * (xi @ w)
    # gradient of logistic loss w.r.t. w
    return -(1.0 - sigmoid(margin)) * yi * xi


def _grad_logistic_multiclass(W, xi, yi, K):
    """∇φ_i(W) for multi-class logistic loss (cross-entropy).

    Returns gradient of shape (d, K).
    """
    logits = xi @ W                        # (K,)
    logits -= np.max(logits)               # numerical stability
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum()            # (K,)

    # one-hot target
    target = np.zeros(K)
    target[yi] = 1.0

    # gradient: outer product of x_i and (probs - target)
    # shape (d, K)
    return np.outer(xi, probs - target)


# ---------------------------------------------------------------------------
# SDCA Epoch — Binary
# ---------------------------------------------------------------------------

def sdca_epoch_binary(alpha, w, X, y, lam, n, lr):
    """Run 1 epoch of SDCA for binary logistic regression.

    Per PROCEDURE_SDCA.md:
        α_i ← α_i - η * (∇φ_i(w) + λn · α_i)
        w   ← w + (α_i_new - α_i_old)

    Args:
        alpha : dual variables (n, d)
        w     : primal weights (d,)
        X     : feature matrix (n, d)
        y     : labels (n,) in {-1, +1}
        lam   : L2 regularization λ
        n     : number of samples
        lr    : learning rate η

    Returns:
        (updated alpha, updated w)
    """
    indices = np.random.permutation(n)

    for i in indices:
        xi = X[i]          # (d,)
        yi = y[i]

        # ∇φ_i(w): shape (d,)
        grad = _grad_logistic_binary(w, xi, yi)

        # Dual update: α_i ← α_i - η*(∇φ_i(w) + λn·α_i)
        alpha_old = alpha[i].copy()
        alpha[i] -= lr * (grad + lam * n * alpha[i])

        # Primal update: w ← w + Δα_i
        w += alpha[i] - alpha_old

    return alpha, w


def sdca_binary(X, y, lam, n_epochs, lr=None, callback=None):
    """Run SDCA for binary logistic regression.

    Args:
        X        : feature matrix (n, d)
        y        : labels (n,) in {-1, +1}
        lam      : L2 regularization λ
        n_epochs : number of epochs
        lr       : learning rate η (default: 1/(λn))
        callback : optional function(w, epoch) called after each epoch

    Returns:
        final primal weights w (d,)
    """
    n, d = X.shape
    alpha = np.zeros((n, d))   # α_i ∈ ℝ^d
    w = np.zeros(d)             # w = Σ α_i = 0 at init

    if lr is None:
        lr = 1.0 / (lam * n)

    for epoch in range(n_epochs):
        alpha, w = sdca_epoch_binary(alpha, w, X, y, lam, n, lr)
        if callback:
            callback(w, epoch)

    return w


# ---------------------------------------------------------------------------
# SDCA Epoch — Multi-class
# ---------------------------------------------------------------------------

def sdca_epoch_multiclass(alpha, W, X, y, lam, n, K, lr):
    """Run 1 epoch of SDCA for multi-class logistic regression.

    Per PROCEDURE_SDCA.md:
        α_i ← α_i - η * (∇φ_i(W) + λn · α_i)    [shape (d, K)]
        W   ← W + (α_i_new - α_i_old)

    Args:
        alpha : dual variables (n, d, K)
        W     : primal weights (d, K)
        X     : feature matrix (n, d)
        y     : labels (n,) in {0, ..., K-1}
        lam   : L2 regularization λ
        n     : number of samples
        K     : number of classes
        lr    : learning rate η

    Returns:
        (updated alpha, updated W)
    """
    indices = np.random.permutation(n)

    for i in indices:
        xi = X[i]   # (d,)
        yi = y[i]

        # ∇φ_i(W): shape (d, K)
        grad = _grad_logistic_multiclass(W, xi, yi, K)

        # Dual update: α_i ← α_i - η*(∇φ_i(W) + λn·α_i)
        alpha_old = alpha[i].copy()   # (d, K)
        alpha[i] -= lr * (grad + lam * n * alpha[i])

        # Primal update: W ← W + Δα_i
        W += alpha[i] - alpha_old

    return alpha, W


def sdca_multiclass(X, y, lam, n_epochs, lr=None, callback=None):
    """Run SDCA for multi-class logistic regression.

    Args:
        X        : feature matrix (n, d)
        y        : labels (n,) in {0, ..., K-1}
        lam      : L2 regularization λ
        n_epochs : number of epochs
        lr       : learning rate η (default: 1/(λn))
        callback : optional function(W, epoch) called after each epoch

    Returns:
        final primal weights W (d, K)
    """
    n, d = X.shape
    K = len(np.unique(y))
    alpha = np.zeros((n, d, K))   # α_i ∈ ℝ^(d×K)
    W = np.zeros((d, K))           # W = Σ α_i = 0 at init

    if lr is None:
        lr = 1.0 / (lam * n)

    for epoch in range(n_epochs):
        alpha, W = sdca_epoch_multiclass(alpha, W, X, y, lam, n, K, lr)
        if callback:
            callback(W, epoch)

    return W


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def sdca_train(X, y, lam, n_epochs, lr=None, multiclass=False, callback=None):
    """Run SDCA for multiple epochs.

    Args:
        X          : feature matrix (n, d)
        y          : labels (n,)
        lam        : L2 regularization λ
        n_epochs   : number of epochs
        lr         : learning rate η (default: 1/(λn))
        multiclass : multi-class flag
        callback   : optional function(w, epoch) called after each epoch

    Returns:
        final primal weights
    """
    if multiclass:
        return sdca_multiclass(X, y, lam, n_epochs, lr, callback)
    else:
        return sdca_binary(X, y, lam, n_epochs, lr, callback)
