"""
svrg.py — SVRG Algorithm (Johnson & Zhang, NIPS 2013)

Implements Algorithm 1 from:
  "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"

Key optimization: precompute phi'(w_tilde^T x_i) for all i before the inner
loop to avoid redundant inner products (scalar storage trick).
"""

import numpy as np
import scipy.sparse as sp
from models.logistic import sigmoid


# ---------------------------------------------------------------------------
# SVRG — Binary Logistic Regression
# ---------------------------------------------------------------------------

def svrg_outer_loop_binary(w_tilde, X, y, lr, lam, m, option='I',
                           track_variance=False):
    """Run one outer iteration of SVRG for binary logistic regression.

    Uses the scalar storage optimization: phi'(w_tilde^T x_i) is precomputed
    for all i so the inner loop avoids recomputing full inner products.

    Args:
        w_tilde:        snapshot weights (d,)
        X:              feature matrix (n, d), dense or sparse CSR
        y:              label vector (n,) with values in {-1, +1}
        lr:             step size (constant)
        lam:            L2 regularization strength
        m:              inner loop length (typically 2n)
        option:         'I'  → next snapshot = w_m (last inner iterate)
                        'II' → next snapshot = random w_t from inner history
        track_variance: if True, also return gradient variance estimate

    Returns:
        w_tilde:    updated snapshot for the next outer iteration
        variance:   gradient variance estimate (only if track_variance=True)
    """
    n = len(y)

    # ── Step 1: Full gradient at snapshot ──
    z_tilde = X @ w_tilde                               # (n,)  inner products
    phi_prime = -y * sigmoid(-y * z_tilde)              # (n,)  phi'(z_i)
    mu = X.T @ phi_prime / n + lam * w_tilde            # (d,)  full gradient

    # ── Step 2: Inner loop ──
    w = w_tilde.copy()
    w_history = [w.copy()] if option == 'II' else None

    variance_sum = 0.0

    for _ in range(m):
        i = np.random.randint(n)
        xi = X[i]

        # Ensure xi is a flat dense array (handles sparse CSR rows)
        if sp.issparse(xi):
            xi = xi.toarray().ravel()
        else:
            xi = np.asarray(xi).ravel()

        # Stochastic gradient at current w
        margin_w = y[i] * (xi @ w)
        g_current = (-y[i] * sigmoid(-margin_w)) * xi + lam * w

        # Stochastic gradient at snapshot (uses precomputed phi')
        g_snapshot = phi_prime[i] * xi + lam * w_tilde

        # SVRG variance-reduced direction
        v = g_current - g_snapshot + mu

        if track_variance:
            # Estimate variance as squared deviation of v from the full gradient mu
            diff = np.asarray(v - mu).ravel()
            variance_sum += (lr ** 2) * np.sum(diff ** 2)

        w = w - lr * v

        if option == 'II':
            w_history.append(w.copy())

    # ── Step 3: Update snapshot ──
    if option == 'I':
        result = w
    else:
        result = w_history[np.random.randint(m)]     # random pick from {w_0, ..., w_{m-1}}

    if track_variance:
        return result, variance_sum / m
    return result


# ---------------------------------------------------------------------------
# SVRG — Multi-class Logistic Regression
# ---------------------------------------------------------------------------

def svrg_outer_loop_multiclass(W_tilde, X, y, lr, lam, m, option='I',
                               track_variance=False):
    """Run one outer iteration of SVRG for multi-class logistic regression.

    Weight matrix W has shape (d, K). The scalar storage trick applies
    per-class via precomputed softmax probabilities at the snapshot.

    Args:
        W_tilde:        snapshot weight matrix (d, K)
        X:              feature matrix (n, d)
        y:              label vector (n,) with values in {0, ..., K-1}
        lr:             step size
        lam:            L2 regularization strength
        m:              inner loop length
        option:         'I' or 'II' (same semantics as binary version)
        track_variance: if True, also return gradient variance estimate

    Returns:
        W_tilde:    updated snapshot for the next outer iteration
        variance:   gradient variance estimate (only if track_variance=True)
    """
    n = len(y)
    K = W_tilde.shape[1]

    # ── Step 1: Full gradient at snapshot ──
    logits_tilde = X @ W_tilde                                           # (n, K)
    logits_tilde -= np.max(logits_tilde, axis=1, keepdims=True)          # stability
    exp_logits = np.exp(logits_tilde)
    probs_tilde = exp_logits / exp_logits.sum(axis=1, keepdims=True)     # (n, K)

    one_hot_all = np.zeros((n, K))
    one_hot_all[np.arange(n), y] = 1.0

    mu = X.T @ (probs_tilde - one_hot_all) / n + lam * W_tilde          # (d, K)

    # ── Step 2: Inner loop ──
    W = W_tilde.copy()
    W_history = [W.copy()] if option == 'II' else None

    variance_sum = 0.0

    for _ in range(m):
        i = np.random.randint(n)
        xi = X[i]
        yi = y[i]

        # Stochastic gradient at current W
        logits_i = xi @ W
        logits_i -= np.max(logits_i)
        probs_i = np.exp(logits_i) / np.exp(logits_i).sum()             # (K,)

        e_yi = np.zeros(K)
        e_yi[yi] = 1.0

        g_current  = np.outer(xi, probs_i - e_yi) + lam * W            # (d, K)
        g_snapshot = np.outer(xi, probs_tilde[i] - e_yi) + lam * W_tilde  # (d, K)

        v = g_current - g_snapshot + mu

        if track_variance:
            diff = v - mu
            variance_sum += lr * np.sum(diff ** 2)

        W = W - lr * v

        if option == 'II':
            W_history.append(W.copy())

    # ── Step 3: Update snapshot ──
    if option == 'I':
        result = W
    else:
        result = W_history[np.random.randint(m)]

    if track_variance:
        return result, variance_sum / m
    return result


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

def svrg_outer_loop(w, X, y, lr, lam, m, multiclass=False, option='I',
                    track_variance=False):
    """Run one outer iteration of SVRG.

    Dispatches to the binary or multi-class implementation based on `multiclass`.

    Args:
        w:              snapshot weights — (d,) for binary, (d, K) for multi-class
        X:              feature matrix (n, d)
        y:              label vector (n,)
        lr:             step size
        lam:            L2 regularization strength
        m:              inner loop length
        multiclass:     multi-class flag
        option:         'I' or 'II'
        track_variance: if True, also return gradient variance estimate

    Returns:
        updated weights (and variance estimate if track_variance=True)
    """
    if multiclass:
        return svrg_outer_loop_multiclass(w, X, y, lr, lam, m, option, track_variance)
    return svrg_outer_loop_binary(w, X, y, lr, lam, m, option, track_variance)


def effective_passes_svrg(n, m):
    """Compute the number of effective dataset passes for one SVRG outer iteration.

    Cost breakdown:
      - 1 full pass to compute the snapshot gradient
      - m/n passes for the inner loop

    Args:
        n: number of training samples
        m: inner loop length

    Returns:
        effective passes (float): 1 + m/n
    """
    return 1.0 + m / n