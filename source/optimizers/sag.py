"""
sag.py — Stochastic Average Gradient (SAG)

Implements SAG for L2-regularized logistic regression, as described in:
"Minimizing Finite Sums with the Stochastic Average Gradient"
(Le Roux, Schmidt & Bach, 2012).

SAG maintains a table of stored gradients g_i for each sample i, and uses:
    w_{t+1} = w_t - η * ( (1/n) Σ g_i + λw_t )

Key property: O(nd) memory for storing all gradients.
For logistic regression with scalar storage optimization, we store only
the scalar phi'_i(w^T x_i) and reconstruct the gradient on the fly.
"""

import numpy as np
from models.logistic import sigmoid


# ---------------------------------------------------------------------------
# SAG for Binary Logistic Regression
# ---------------------------------------------------------------------------

def sag_epoch_binary(w, X, y, lam, lr, stored_grads, sum_grad):
    """Run 1 epoch of SAG for binary logistic regression.

    Uses scalar storage: stores phi'_i(w^T x_i) for each sample i.
    The full gradient contribution of sample i is:
        g_i = phi'_i * x_i + λw

    But we store only the scalar phi'_i and reconstruct g_i when needed.

    Args:
        w: current weights (d,)
        X: feature matrix (n, d)
        y: labels (n,) in {-1, +1}
        lam: L2 regularization
        lr: learning rate
        stored_grads: array of stored scalar values (n,) — phi'_i for each i
        sum_grad: current sum of data part of gradients (d,)

    Returns:
        (updated w, updated stored_grads, updated sum_grad)
    """
    n = len(y)
    indices = np.random.permutation(n)

    for i in indices:
        xi = X[i]
        yi = y[i]

        # Current margin and gradient scalar
        margin = yi * (xi @ w)
        new_scalar = -yi * sigmoid(-margin)  # phi'_i(w^T x_i)

        old_scalar = stored_grads[i]

        # Update sum_grad: add new contribution, remove old
        # g_i_data = scalar * x_i
        # sum_grad = Σ stored_scalar_j * x_j
        sum_grad += (new_scalar - old_scalar) * xi

        # Store new scalar
        stored_grads[i] = new_scalar

        # Full gradient: (1/n) * sum_grad + λ * w
        full_g = sum_grad / n + lam * w

        # SAG update: w = w - lr * full_g
        w = w - lr * full_g

    return w, stored_grads, sum_grad


def sag_binary(X, y, lam, n_epochs, lr, callback=None):
    """Run SAG for binary logistic regression.

    Args:
        X: feature matrix (n, d)
        y: labels (n,) in {-1, +1}
        lam: L2 regularization
        n_epochs: number of epochs
        lr: learning rate
        callback: optional function(w, epoch) called after each epoch

    Returns:
        final weights w
    """
    n, d = X.shape
    w = np.zeros(d)

    # Initialize stored scalars and sum_grad
    # Start with all zeros (w=0 => margin=0 => phi' = -y * sigmoid(0) = -y/2)
    stored_grads = np.zeros(n)
    sum_grad = np.zeros(d)

    # Initialize with gradient at w=0
    for i in range(n):
        yi = y[i]
        xi = X[i]
        scalar = -yi * sigmoid(0)  # = -yi * 0.5
        stored_grads[i] = scalar
        sum_grad += scalar * xi

    for epoch in range(n_epochs):
        w, stored_grads, sum_grad = sag_epoch_binary(
            w, X, y, lam, lr, stored_grads, sum_grad
        )
        if callback:
            callback(w, epoch)

    return w


# ---------------------------------------------------------------------------
# SAG for Multi-class Logistic Regression
# ---------------------------------------------------------------------------

def sag_epoch_multiclass(W, X, y, lam, lr, stored_grads, sum_grad):
    """Run 1 epoch of SAG for multi-class logistic regression.

    For multi-class, we store the probability vector (K,) for each sample.
    The gradient contribution is: g_i = (probs_i - e_{y_i}) ⊗ x_i + λW

    We store only the probability vector (K scalars) per sample.

    Args:
        W: current weights (d, K)
        X: feature matrix (n, d)
        y: labels (n,) in {0, ..., K-1}
        lam: L2 regularization
        lr: learning rate
        stored_grads: stored probability vectors (n, K)
        sum_grad: current sum of data gradients (d, K)

    Returns:
        (updated W, updated stored_grads, updated sum_grad)
    """
    n = len(y)
    K = W.shape[1]
    indices = np.random.permutation(n)

    for i in indices:
        xi = X[i]
        yi = y[i]

        # Current probabilities
        logits = xi @ W
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        new_probs = exp_logits / np.sum(exp_logits)  # (K,)

        old_probs = stored_grads[i]  # (K,)

        # One-hot
        e_yi = np.zeros(K)
        e_yi[yi] = 1.0

        # Gradient data part: (probs - e_yi) ⊗ xi
        # Update sum_grad: add new, remove old
        new_grad_data = np.outer(xi, new_probs - e_yi)  # (d, K)
        old_grad_data = np.outer(xi, old_probs - e_yi)  # (d, K)

        sum_grad += new_grad_data - old_grad_data

        # Store new probabilities
        stored_grads[i] = new_probs

        # Full gradient: (1/n) * sum_grad + λ * W
        full_g = sum_grad / n + lam * W

        # SAG update
        W = W - lr * full_g

    return W, stored_grads, sum_grad


def sag_multiclass(X, y, lam, n_epochs, lr, callback=None):
    """Run SAG for multi-class logistic regression.

    Args:
        X: feature matrix (n, d)
        y: labels (n,) in {0, ..., K-1}
        lam: L2 regularization
        n_epochs: number of epochs
        lr: learning rate
        callback: optional function(w, epoch) called after each epoch

    Returns:
        final weights W (d, K)
    """
    n, d = X.shape
    K = len(np.unique(y))
    W = np.zeros((d, K))

    # Initialize stored probabilities and sum_grad
    # At w=0, all classes have equal probability 1/K
    init_probs = np.ones(K) / K
    stored_grads = np.tile(init_probs, (n, 1))  # (n, K)
    sum_grad = np.zeros((d, K))

    # Initialize sum_grad with initial gradients
    for i in range(n):
        xi = X[i]
        yi = y[i]
        e_yi = np.zeros(K)
        e_yi[yi] = 1.0
        sum_grad += np.outer(xi, init_probs - e_yi)

    for epoch in range(n_epochs):
        W, stored_grads, sum_grad = sag_epoch_multiclass(
            W, X, y, lam, lr, stored_grads, sum_grad
        )
        if callback:
            callback(W, epoch)

    return W


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def sag_epoch(w, X, y, lam, lr, multiclass=False, state=None):
    """Run 1 epoch of SAG.

    Args:
        w: current weights
        X: feature matrix
        y: labels
        lam: L2 regularization
        lr: learning rate
        multiclass: multi-class flag
        state: dict with 'stored_grads' and 'sum_grad'

    Returns:
        (updated w, updated state)
    """
    if state is None:
        raise ValueError("SAG requires state dict with 'stored_grads' and 'sum_grad'")

    if multiclass:
        w_out, stored, sum_g = sag_epoch_multiclass(
            w, X, y, lam, lr, state['stored_grads'], state['sum_grad']
        )
    else:
        w_out, stored, sum_g = sag_epoch_binary(
            w, X, y, lam, lr, state['stored_grads'], state['sum_grad']
        )

    return w_out, {'stored_grads': stored, 'sum_grad': sum_g}


def sag_train(X, y, lam, n_epochs, multiclass=False, lr=0.01, callback=None):
    """Run SAG for multiple epochs.

    Args:
        X: feature matrix (n, d)
        y: labels (n,)
        lam: L2 regularization
        n_epochs: number of epochs
        multiclass: multi-class flag
        lr: learning rate
        callback: optional function(w, epoch) called after each epoch

    Returns:
        final weights
    """
    if multiclass:
        return sag_multiclass(X, y, lam, n_epochs, lr, callback)
    else:
        return sag_binary(X, y, lam, n_epochs, lr, callback)
