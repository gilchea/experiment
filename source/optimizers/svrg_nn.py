"""
svrg_nn.py — SVRG for Neural Networks (Non-convex)

Implements SVRG for the 1-hidden-layer neural network.
Unlike logistic regression, the scalar storage optimization does NOT apply
here because the gradient depends on the full forward pass through the
hidden layer.

The algorithm follows the same structure as the convex SVRG:
  Algorithm 1 from Johnson & Zhang (NIPS 2013)

Per IMPLEMENT.md:
  - Non-convex: m = 5n, mini-batch size = 10 for inner loop.
  - Warm-start: 10 epochs SGD before SVRG.
"""

import numpy as np
from models.neural_net import (
    full_grad, stoch_grad, copy_params, add_params, scale_params
)


def svrg_nn_outer_loop(params, X, y, lr, lam, m, option='I',
                       track_variance=False, batch_size=10):
    """One outer iteration of SVRG for the neural network.

    Per IMPLEMENT.md §3: inner loop uses mini-batch size = 10 for NN.

    Args:
        params     : dict with 'W1', 'b1', 'W2', 'b2'
        X          : feature matrix (n, d)
        y          : label vector (n,) with values in {0, ..., K-1}
        lr         : step size eta (constant)
        lam        : L2 regularization strength
        m          : number of inner loop iterations (paper: 5n for non-convex)
        option     : 'I' (params = w_m) or 'II' (random w_t from {w_0,...,w_{m-1}})
        track_variance: if True, compute and return gradient variance estimate
        batch_size : mini-batch size for inner loop (10 per IMPLEMENT.md)

    Returns:
        updated params for next outer iteration
        (if track_variance, also returns variance estimate)
    """
    n = len(y)

    # ── Step 1: Compute full gradient mu_tilde ──
    # Full gradient is always computed on the entire dataset
    mu = full_grad(params, X, y, lam)

    # ── Step 2: Inner loop ──
    w = copy_params(params)

    if option == 'II':
        w_history = [copy_params(w)]

    # Variance tracking
    variance_sum = 0.0
    variance_count = 0

    for t in range(m):
        # Sample mini-batch of size batch_size (per IMPLEMENT.md §3: size=10 for NN)
        batch_idx = np.random.randint(n, size=batch_size)
        xi = X[batch_idx]    # (batch_size, d)
        yi = y[batch_idx]    # (batch_size,)

        # ∇ψ_batch(w) — stochastic gradient at current iterate
        g_current = stoch_grad(w, xi, yi, lam)

        # ∇ψ_batch(w_tilde) — stochastic gradient at snapshot
        g_snapshot = stoch_grad(params, xi, yi, lam)

        # SVRG update direction: v = g_current - g_snapshot + mu
        v = add_params(add_params(g_current, scale_params(g_snapshot, -1.0)), mu)

        # Track variance: E[||v - E[v]||²]
        if track_variance:
            diff = add_params(v, scale_params(mu, -1.0))
            sq_norm = sum(np.sum(d * d) for d in diff.values())
            variance_sum += sq_norm
            variance_count += 1

        # SVRG update: w = w - lr * v
        w = add_params(w, scale_params(v, -lr))

        if option == 'II':
            w_history.append(copy_params(w))

    # ── Step 3: Update snapshot ──
    if option == 'I':
        result = w
    else:  # Option II: random pick from {w_0, ..., w_{m-1}} per spec
        idx = np.random.randint(m)
        result = w_history[idx]

    if track_variance:
        variance_estimate = variance_sum / max(variance_count, 1)
        return result, variance_estimate
    return result


def effective_passes_svrg_nn(n, m, batch_size=10):
    """Compute effective passes for one SVRG outer iteration.

    Each outer iteration costs:
    - 1 pass for full gradient computation
    - m/n passes for inner loop (each step uses batch_size samples,
      but cost is counted in gradient evaluations / n)

    Total = 1 + m/n

    Args:
        n: number of samples
        m: inner loop length

    Returns:
        effective passes for one outer iteration
    """
    # return 1.0 + m / n
    return 1.0 + 2.0 * m * batch_size / n
