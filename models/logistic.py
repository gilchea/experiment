"""
logistic.py — L2-Regularized Logistic Regression (Binary + Multi-class)

Provides loss, full gradient, and stochastic gradient for both binary
(y in {-1, +1}) and multi-class (y in {0..K-1}) logistic regression.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Numerically stable sigmoid function."""
    x = np.clip(x, -100, 100)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits):
    """Numerically stable softmax.

    Args:
        logits: (n, K) or (K,) array of logits

    Returns:
        probabilities of same shape
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Binary Logistic Regression  (y in {-1, +1})
# ---------------------------------------------------------------------------

def loss_binary(w, X, y, lam):
    """P(w) for binary logistic regression.

    Args:
        w: weight vector (d,)
        X: feature matrix (n, d) — dense or sparse
        y: label vector (n,) with values in {-1, +1}
        lam: L2 regularization strength

    Returns:
        scalar loss value
    """
    n = len(y)
    margins = y * (X @ w)               # (n,)
    log_part = np.log1p(np.exp(-margins))
    return np.mean(log_part) + 0.5 * lam * np.dot(w, w)


def full_grad_binary(w, X, y, lam):
    """Full gradient ∇P(w).

    ∇P(w) = (1/n) Σ ∇ψ_i(w) + λw
    ∇ψ_i(w) = -y_i * σ(-y_i w^T x_i) * x_i

    Args:
        w: weight vector (d,)
        X: feature matrix (n, d) — dense or sparse
        y: label vector (n,) with values in {-1, +1}
        lam: L2 regularization strength

    Returns:
        gradient vector (d,)
    """
    n = len(y)
    margins = y * (X @ w)               # (n,)
    coefs = -y * sigmoid(-margins)      # (n,)
    grad_data = X.T @ coefs / n         # (d,)
    return grad_data + lam * w


def stoch_grad_binary(w, xi, yi, lam):
    """Stochastic gradient ∇ψ_i(w) for a single sample.

    Args:
        w: weight vector (d,)
        xi: single sample features (d,) — dense or sparse
        yi: single label in {-1, +1}
        lam: L2 regularization strength

    Returns:
        gradient vector (d,)
    """
    margin = yi * (xi @ w)
    coef = -yi * sigmoid(-margin)
    return coef * xi + lam * w


# ---------------------------------------------------------------------------
# Multi-class Logistic Regression  (y in {0..K-1})
# ---------------------------------------------------------------------------

def loss_multiclass(W, X, y, lam):
    """P(W) for multi-class logistic regression (softmax + cross-entropy).

    Args:
        W: weight matrix (d, K)
        X: feature matrix (n, d)
        y: label vector (n,) with values in {0, ..., K-1}
        lam: L2 regularization strength

    Returns:
        scalar loss value
    """
    n = len(y)
    K = W.shape[1]
    logits = X @ W                          # (n, K)

    # Log-softmax for numerical stability
    logits_max = np.max(logits, axis=1, keepdims=True)
    log_probs = logits - logits_max
    log_probs -= np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))

    correct_log_probs = log_probs[np.arange(n), y]
    loss = -np.mean(correct_log_probs)
    reg = 0.5 * lam * np.sum(W * W)
    return loss + reg


def full_grad_multiclass(W, X, y, lam):
    """Full gradient for multi-class logistic regression.

    ∇P(W) = (1/n) Σ (softmax(W^T x_i) - e_{y_i}) ⊗ x_i + λW

    Args:
        W: weight matrix (d, K)
        X: feature matrix (n, d)
        y: label vector (n,) with values in {0, ..., K-1}
        lam: L2 regularization strength

    Returns:
        gradient matrix (d, K)
    """
    n = len(y)
    K = W.shape[1]
    logits = X @ W                          # (n, K)
    probs = softmax(logits)                 # (n, K)

    one_hot = np.zeros((n, K))
    one_hot[np.arange(n), y] = 1.0

    grad_data = X.T @ (probs - one_hot) / n  # (d, K)
    return grad_data + lam * W


def stoch_grad_multiclass(W, xi, yi, lam):
    """Stochastic gradient for a single sample.

    Args:
        W: weight matrix (d, K)
        xi: single sample (d,)
        yi: label in {0, ..., K-1}
        lam: L2 regularization strength

    Returns:
        gradient matrix (d, K)
    """
    K = W.shape[1]
    logits = xi @ W                         # (K,)
    probs = softmax(logits)                 # (K,)

    one_hot = np.zeros(K)
    one_hot[yi] = 1.0

    grad = np.outer(xi, probs - one_hot)    # (d, K)
    return grad + lam * W


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def loss(w, X, y, lam, multiclass=False):
    """Unified loss function."""
    if multiclass:
        return loss_multiclass(w, X, y, lam)
    return loss_binary(w, X, y, lam)


def full_grad(w, X, y, lam, multiclass=False):
    """Unified full gradient."""
    if multiclass:
        return full_grad_multiclass(w, X, y, lam)
    return full_grad_binary(w, X, y, lam)


def stoch_grad(w, xi, yi, lam, multiclass=False):
    """Unified stochastic gradient."""
    if multiclass:
        return stoch_grad_multiclass(w, xi, yi, lam)
    return stoch_grad_binary(w, xi, yi, lam)
