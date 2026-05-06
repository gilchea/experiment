"""
svrg.py — SVRG Algorithm (Core Contribution)

Implements Algorithm 1 from Johnson & Zhang (NIPS 2013):
"Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"

Uses scalar storage optimization for linear prediction models:
precompute phi'(w_tilde^T x_i) for all i to reduce inner-loop cost.
"""

import numpy as np
from models.logistic import sigmoid


# ---------------------------------------------------------------------------
# SVRG for Binary Logistic Regression
# ---------------------------------------------------------------------------

def svrg_outer_loop_binary(w_tilde, X, y, lr, lam, m, option='I'):
    """One outer iteration of SVRG for binary logistic regression.

    Uses scalar storage optimization: precompute phi'(w_tilde^T x_i) for all i
    to avoid recomputing inner products in the inner loop.

    Args:
        w_tilde: snapshot weights (d,)
        X: feature matrix (n, d) — dense or sparse CSR
        y: label vector (n,) with values in {-1, +1}
        lr: step size eta (constant)
        lam: L2 regularization strength
        m: number of inner loop iterations (typically 2n)
        option: 'I' (w_tilde = w_m) or 'II' (random w_t from history)

    Returns:
        updated w_tilde for next outer iteration
    """
    n = len(y)

    # ── Step 1: Compute full gradient mu_tilde ──
    # Precompute z_i = w_tilde^T x_i for all i
    z_tilde = X @ w_tilde                              # (n,)

    # phi'(z) for logistic loss: phi(z) = log(1 + exp(-y*z))
    # phi'(z) = -y * sigmoid(-y*z)
    phi_prime_tilde = -y * sigmoid(-y * z_tilde)       # (n,)

    # Full gradient: mu = (1/n) sum phi'_i(z_i) * x_i + lam * w_tilde
    mu = (X.T @ phi_prime_tilde) / n + lam * w_tilde   # (d,)

    # ── Step 2: Inner loop ──
    w = w_tilde.copy()

    if option == 'II':
        w_history = [w.copy()]

    for t in range(m):
        i = np.random.randint(n)
        xi = X[i]          # (d,) — dense 1D array
        yi = y[i]

        # nabla psi_i(w) = -y_i * sigmoid(-y_i * w^T x_i) * x_i + lam * w
        margin_w = yi * (xi @ w)
        g_current = (-yi * sigmoid(-margin_w)) * xi + lam * w

        # nabla psi_i(w_tilde) — using precomputed scalar phi'_i(z_i)
        # = phi'_i(z_i) * x_i + lam * w_tilde
        g_snapshot = phi_prime_tilde[i] * xi + lam * w_tilde

        # SVRG update: w = w - lr * (g_current - g_snapshot + mu)
        w = w - lr * (g_current - g_snapshot + mu)

        if option == 'II':
            w_history.append(w.copy())

    # ── Step 3: Update snapshot ──
    if option == 'I':
        return w
    else:  # Option II: random pick from {w_0, ..., w_m}
        idx = np.random.randint(m + 1)
        return w_history[idx]


# ---------------------------------------------------------------------------
# SVRG for Multi-class Logistic Regression
# ---------------------------------------------------------------------------

def svrg_outer_loop_multiclass(W_tilde, X, y, lr, lam, m, option='I'):
    """One outer iteration of SVRG for multi-class logistic regression.

    For multi-class with K classes, W is (d, K).
    The scalar optimization still applies per class via precomputed probs.

    Args:
        W_tilde: snapshot weight matrix (d, K)
        X: feature matrix (n, d)
        y: label vector (n,) with values in {0, ..., K-1}
        lr: step size eta
        lam: L2 regularization
        m: inner loop length
        option: 'I' or 'II'

    Returns:
        updated W_tilde
    """
    n = len(y)
    K = W_tilde.shape[1]

    # ── Step 1: Full gradient ──
    logits_tilde = X @ W_tilde                          # (n, K)

    # Numerically stable softmax
    logits_tilde -= np.max(logits_tilde, axis=1, keepdims=True)
    exp_logits = np.exp(logits_tilde)
    probs_tilde = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (n, K)

    # One-hot encoding
    one_hot = np.zeros((n, K))
    one_hot[np.arange(n), y] = 1.0

    # Gradient: (1/n) sum (probs_i - e_{y_i}) otimes x_i + lam * W
    mu = (X.T @ (probs_tilde - one_hot)) / n + lam * W_tilde  # (d, K)

    # ── Step 2: Inner loop ──
    W = W_tilde.copy()

    if option == 'II':
        W_history = [W.copy()]

    for t in range(m):
        i = np.random.randint(n)
        xi = X[i]          # (d,)
        yi = y[i]

        # nabla psi_i(W) for current w
        logits_i = xi @ W                                # (K,)
        logits_i -= np.max(logits_i)
        probs_i = np.exp(logits_i) / np.sum(np.exp(logits_i))  # (K,)

        e_yi = np.zeros(K)
        e_yi[yi] = 1.0

        g_current = np.outer(xi, probs_i - e_yi) + lam * W  # (d, K)

        # nabla psi_i(W_tilde) — using precomputed probabilities
        g_snapshot = np.outer(xi, probs_tilde[i] - e_yi) + lam * W_tilde  # (d, K)

        # SVRG update
        W = W - lr * (g_current - g_snapshot + mu)

        if option == 'II':
            W_history.append(W.copy())

    if option == 'I':
        return W
    else:
        idx = np.random.randint(m + 1)
        return W_history[idx]


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def svrg_outer_loop(w, X, y, lr, lam, m, multiclass=False, option='I'):
    """One outer iteration of SVRG.

    Args:
        w: snapshot weights (d,) for binary or (d, K) for multi-class
        X: feature matrix (n, d)
        y: label vector (n,)
        lr: step size
        lam: regularization
        m: inner loop length
        multiclass: multi-class flag
        option: 'I' or 'II'

    Returns:
        updated weights
    """
    if multiclass:
        return svrg_outer_loop_multiclass(w, X, y, lr, lam, m, option)
    return svrg_outer_loop_binary(w, X, y, lr, lam, m, option)


def effective_passes_svrg(n, m):
    """Compute effective passes for one SVRG outer iteration.

    Each outer iteration costs:
    - 1 pass for full gradient computation
    - m/n passes for inner loop

    Total = 1 + m/n

    Args:
        n: number of samples
        m: inner loop length

    Returns:
        effective passes for one outer iteration
    """
    return 1.0 + m / n
