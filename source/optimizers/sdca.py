"""
sdca.py — Stochastic Dual Coordinate Ascent (SDCA)

Implements SDCA for L2-regularized logistic regression, as described in:
"Stochastic Dual Coordinate Ascent Methods for Regularized Loss Minimization"
(Shalev-Shwartz & Zhang, 2013).

SDCA solves the dual problem and maintains the primal-dual relationship:
    w = (1/(λn)) Σ α_i y_i x_i

Key advantage: O(n) memory (store dual variables α_i), linear convergence.
"""

import numpy as np
from models.logistic import sigmoid


# ---------------------------------------------------------------------------
# SDCA for Binary Logistic Regression
# ---------------------------------------------------------------------------

def sdca_epoch_binary(alpha, w, X, y, lam, n):
    """Run 1 epoch of SDCA for binary logistic regression.

    Updates each dual variable α_i once per epoch using the SDCA update:
        Δα = clip( (1 - σ(y_i w^T x_i) - α_i) / (||x_i||²/(λn) + 1), -α_i, 1-α_i )

    Args:
        alpha: dual variables (n,)
        w: primal weights (d,)
        X: feature matrix (n, d)
        y: labels (n,) in {-1, +1}
        lam: L2 regularization
        n: number of samples

    Returns:
        (updated alpha, updated w)
    """
    indices = np.random.permutation(n)

    for i in indices:
        xi = X[i]
        yi = y[i]

        # Current margin and probability
        margin = yi * (xi @ w)
        prob = sigmoid(margin)

        # Precompute ||x_i||² / (λn)
        xi_norm_sq = xi @ xi
        q_i = xi_norm_sq / (lam * n)

        # SDCA dual update
        # Δα = (1 - prob - α_i) / (q_i + 1)
        delta = (1.0 - prob - alpha[i]) / (q_i + 1.0)
        delta = np.clip(delta, -alpha[i], 1.0 - alpha[i])

        # Update dual variable
        alpha[i] += delta

        # Update primal: w += (Δα * yi / (λn)) * xi
        w += (delta * yi / (lam * n)) * xi

    return alpha, w


def sdca_binary(X, y, lam, n_epochs, callback=None):
    """Run SDCA for binary logistic regression with per-epoch logging.

    Args:
        X: feature matrix (n, d)
        y: labels (n,) in {-1, +1}
        lam: L2 regularization
        n_epochs: number of epochs
        callback: optional function(w, epoch) called after each epoch

    Returns:
        final primal weights w
    """
    n, d = X.shape
    alpha = np.zeros(n)
    w = np.zeros(d)

    for epoch in range(n_epochs):
        alpha, w = sdca_epoch_binary(alpha, w, X, y, lam, n)
        if callback:
            callback(w, epoch)

    return w


# ---------------------------------------------------------------------------
# SDCA for Multi-class Logistic Regression
# ---------------------------------------------------------------------------

def sdca_epoch_multiclass(alpha, W, X, y, lam, n, K):
    """Run 1 epoch of SDCA for multi-class logistic regression.

    For each sample i, updates the K-dimensional dual variable α_i.

    Args:
        alpha: dual variables (n, K)
        W: primal weights (d, K)
        X: feature matrix (n, d)
        y: labels (n,) in {0, ..., K-1}
        lam: L2 regularization
        n: number of samples
        K: number of classes

    Returns:
        (updated alpha, updated W)
    """
    indices = np.random.permutation(n)

    for i in indices:
        xi = X[i]
        yi = y[i]

        # Current logits and probabilities
        logits = xi @ W
        logits_stable = logits - np.max(logits)
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / np.sum(exp_logits)

        # Precompute ||x_i||² / (λn)
        xi_norm_sq = xi @ xi
        q_i = xi_norm_sq / (lam * n)

        # Update each class coordinate
        for k in range(K):
            target = 1.0 if k == yi else 0.0
            delta = (target - probs[k] - alpha[i, k]) / (q_i + 1.0)
            delta = np.clip(delta, -alpha[i, k], 1.0 - alpha[i, k])

            alpha[i, k] += delta
            W[:, k] += (delta / (lam * n)) * xi

    return alpha, W


def sdca_multiclass(X, y, lam, n_epochs, callback=None):
    """Run SDCA for multi-class logistic regression with per-epoch logging.

    Args:
        X: feature matrix (n, d)
        y: labels (n,) in {0, ..., K-1}
        lam: L2 regularization
        n_epochs: number of epochs
        callback: optional function(w, epoch) called after each epoch

    Returns:
        final primal weights W (d, K)
    """
    n, d = X.shape
    K = len(np.unique(y))
    alpha = np.zeros((n, K))
    W = np.zeros((d, K))

    for epoch in range(n_epochs):
        alpha, W = sdca_epoch_multiclass(alpha, W, X, y, lam, n, K)
        if callback:
            callback(W, epoch)

    return W


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def sdca_train(X, y, lam, n_epochs, multiclass=False, callback=None):
    """Run SDCA for multiple epochs.

    Args:
        X: feature matrix (n, d)
        y: labels (n,)
        lam: L2 regularization
        n_epochs: number of epochs
        multiclass: multi-class flag
        callback: optional function(w, epoch) called after each epoch

    Returns:
        final primal weights
    """
    if multiclass:
        return sdca_multiclass(X, y, lam, n_epochs, callback)
    else:
        return sdca_binary(X, y, lam, n_epochs, callback)
