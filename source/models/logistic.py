"""
logistic.py — L2-Regularized Logistic Regression (Binary + Multi-class)

Provides loss, full gradient, and stochastic gradient for:
  - Binary logistic regression:    y in {-1, +1}
  - Multi-class logistic regression: y in {0, ..., K-1}
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    x = np.clip(x, -100, 100)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits):
    """Numerically stable softmax.

    Args:
        logits: array of shape (n, K) or (K,)

    Returns:
        probabilities of the same shape
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Binary Logistic Regression  (y in {-1, +1})
# ---------------------------------------------------------------------------

def loss_binary(w, X, y, lam):
    """Compute loss P(w) for binary logistic regression.

    P(w) = (1/n) sum log(1 + exp(-y_i * w^T x_i)) + (lam/2) ||w||^2

    Args:
        w:   weight vector (d,)
        X:   feature matrix (n, d), dense or sparse
        y:   label vector (n,) with values in {-1, +1}
        lam: L2 regularization strength

    Returns:
        scalar loss value
    """
    margins = y * (X @ w)
    log_loss = np.log1p(np.exp(-np.abs(margins))) + np.maximum(0.0, -margins)
    return np.mean(log_loss) + 0.5 * lam * np.dot(w, w)


def full_grad_binary(w, X, y, lam):
    """Compute full gradient ∇P(w) for binary logistic regression.

    ∇P(w) = (1/n) sum [-y_i * sigmoid(-y_i w^T x_i) * x_i] + lam * w

    Args:
        w:   weight vector (d,)
        X:   feature matrix (n, d), dense or sparse
        y:   label vector (n,) with values in {-1, +1}
        lam: L2 regularization strength

    Returns:
        gradient vector (d,)
    """
    n = len(y)
    margins = y * (X @ w)
    coefs = -y * sigmoid(-margins)
    return X.T @ coefs / n + lam * w


def stoch_grad_binary(w, xi, yi, lam):
    """Compute stochastic gradient ∇ψ_i(w) for a single sample.

    Args:
        w:   weight vector (d,)
        xi:  feature vector for sample i (d,), dense or sparse
        yi:  label for sample i in {-1, +1}
        lam: L2 regularization strength

    Returns:
        gradient vector (d,)
    """
    margin = yi * (xi @ w)
    coef = -yi * sigmoid(-margin)
    return coef * xi + lam * w


# ---------------------------------------------------------------------------
# Multi-class Logistic Regression  (y in {0, ..., K-1})
# ---------------------------------------------------------------------------

def loss_multiclass(W, X, y, lam):
    """Compute loss P(W) for multi-class logistic regression (softmax cross-entropy).

    P(W) = (1/n) sum [-logits[y_i] + log(sum exp(logits))] + (lam/2) ||W||^2

    Args:
        W:   weight matrix (d, K)
        X:   feature matrix (n, d)
        y:   label vector (n,) with values in {0, ..., K-1}
        lam: L2 regularization strength

    Returns:
        scalar loss value
    """
    n = len(y)
    logits = X @ W                                                       # (n, K)
    logits_max = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = logits_max + np.log(
        np.sum(np.exp(logits - logits_max), axis=1, keepdims=True)
    )
    loss_data = np.mean(log_sum_exp.ravel() - logits[np.arange(n), y])
    return loss_data + 0.5 * lam * np.sum(W * W)


def full_grad_multiclass(W, X, y, lam):
    """Compute full gradient ∇P(W) for multi-class logistic regression.

    ∇P(W) = (1/n) sum (softmax(W^T x_i) - e_{y_i}) ⊗ x_i + lam * W

    Args:
        W:   weight matrix (d, K)
        X:   feature matrix (n, d)
        y:   label vector (n,) with values in {0, ..., K-1}
        lam: L2 regularization strength

    Returns:
        gradient matrix (d, K)
    """
    n = len(y)
    K = W.shape[1]
    probs = softmax(X @ W)                                               # (n, K)

    one_hot = np.zeros((n, K))
    one_hot[np.arange(n), y] = 1.0

    return X.T @ (probs - one_hot) / n + lam * W


def stoch_grad_multiclass(W, xi, yi, lam):
    """Compute stochastic gradient ∇ψ_i(W) for a single sample.

    Args:
        W:   weight matrix (d, K)
        xi:  feature vector for sample i (d,)
        yi:  label for sample i in {0, ..., K-1}
        lam: L2 regularization strength

    Returns:
        gradient matrix (d, K)
    """
    K = W.shape[1]
    probs = softmax(xi @ W)                                              # (K,)

    one_hot = np.zeros(K)
    one_hot[yi] = 1.0

    return np.outer(xi, probs - one_hot) + lam * W


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def loss(w, X, y, lam, multiclass=False):
    """Compute loss P(w). Dispatches to binary or multi-class variant."""
    if multiclass:
        return loss_multiclass(w, X, y, lam)
    return loss_binary(w, X, y, lam)


def full_grad(w, X, y, lam, multiclass=False):
    """Compute full gradient ∇P(w). Dispatches to binary or multi-class variant."""
    if multiclass:
        return full_grad_multiclass(w, X, y, lam)
    return full_grad_binary(w, X, y, lam)


def stoch_grad(w, xi, yi, lam, multiclass=False):
    """Compute stochastic gradient ∇ψ_i(w). Dispatches to binary or multi-class variant."""
    if multiclass:
        return stoch_grad_multiclass(w, xi, yi, lam)
    return stoch_grad_binary(w, xi, yi, lam)