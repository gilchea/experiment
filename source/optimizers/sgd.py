"""
sgd.py — SGD Baselines (Constant & Decaying Learning Rates)

Implements two SGD variants used as baselines (NIPS 2013):
  1. SGD-const : fixed learning rate throughout training.
  2. SGD-best  : exponential decay schedule eta(t) = eta_0 * b^(epoch).

Per paper conventions:
  - Batch size = 1 for convex problems (logistic regression).
  - X-axis in plots = #grad_evals / n  (effective passes over the dataset).
"""

import numpy as np
from models.logistic import stoch_grad, full_grad


# ---------------------------------------------------------------------------
# SGD — Constant Learning Rate
# ---------------------------------------------------------------------------

def sgd_epoch_constant(w, X, y, lr, lam, multiclass=False, track_variance=True):
    """Run one epoch of SGD with a constant learning rate.

    Update rule: w <- w - lr * grad_psi_i(w)  for i drawn uniformly at random.

    Args:
        w:              weight vector/matrix
        X:              feature matrix (n, d)
        y:              label vector (n,)
        lr:             constant learning rate
        lam:            L2 regularization strength
        multiclass:     whether to use multi-class logistic regression
        track_variance: whether to estimate gradient variance

    Returns:
        w:                updated weights after one epoch
        epoch_variance:   average squared step norm (0.0 if track_variance=False)
    """
    n = len(y)
    variance_sum = 0.0

    for i in np.random.permutation(n):
        g = stoch_grad(w, X[i], y[i], lam, multiclass)

        if track_variance:
            # Proxy for update variance: ||lr * g||^2
            variance_sum += np.sum((lr * g) ** 2)

        w = w - lr * g

    epoch_variance = variance_sum / n if track_variance else 0.0
    return w, epoch_variance


def sgd_constant(w, X, y, lr, lam, n_epochs, multiclass=False,
                 callback=None, track_variance=True):
    """Run multiple epochs of SGD with a constant learning rate.

    Args:
        w:              initial weights
        X:              feature matrix (n, d)
        y:              label vector (n,)
        lr:             constant learning rate
        lam:            L2 regularization strength
        n_epochs:       number of training epochs
        multiclass:     multi-class flag
        callback:       optional function(w, epoch) called after each epoch
        track_variance: whether to track per-epoch gradient variance

    Returns:
        w:          final weights
        variances:  list of per-epoch variance estimates (only if track_variance=True)
    """
    variances = []

    for epoch in range(n_epochs):
        w, epoch_var = sgd_epoch_constant(w, X, y, lr, lam, multiclass, track_variance)

        if track_variance:
            variances.append(epoch_var)
        if callback:
            callback(w, epoch)

    if track_variance:
        return w, variances
    return w


# ---------------------------------------------------------------------------
# SGD — Decaying Learning Rate (SGD-best)
# ---------------------------------------------------------------------------

def sgd_epoch_decay(w, X, y, lr0, lam, n, t_start, b, multiclass=False,
                    track_variance=True):
    """Run one epoch of SGD with an exponential decaying learning rate.

    Decay schedule: eta(t) = lr0 * b^(epoch)
    where epoch = t // n  (integer number of complete passes so far).

    Args:
        w:              weight vector/matrix
        X:              feature matrix (n, d)
        y:              label vector (n,)
        lr0:            initial learning rate
        lam:            L2 regularization strength
        n:              dataset size (used to compute the current epoch index)
        t_start:        total gradient steps taken before this epoch
        b:              decay multiplier per epoch (0 < b < 1)
        multiclass:     multi-class flag
        track_variance: whether to estimate gradient variance

    Returns:
        w:              updated weights after one epoch
        t_end:          total gradient steps after this epoch
        epoch_variance: average squared step norm (0.0 if track_variance=False)
    """
    variance_sum = 0.0
    t = t_start

    for i in np.random.permutation(n):
        lr_t = lr0 * (b ** (t // n))
        g = stoch_grad(w, X[i], y[i], lam, multiclass)

        if track_variance:
            variance_sum += np.sum((lr_t * g) ** 2)

        w = w - lr_t * g
        t += 1

    epoch_variance = variance_sum / n if track_variance else 0.0
    return w, t, epoch_variance


def sgd_decay(w, X, y, lr0, lam, n_epochs, b, multiclass=False,
              callback=None, track_variance=True):
    """Run multiple epochs of SGD with an exponential decaying learning rate.

    Args:
        w:              initial weights
        X:              feature matrix (n, d)
        y:              label vector (n,)
        lr0:            initial learning rate
        lam:            L2 regularization strength
        n_epochs:       number of training epochs
        b:              decay multiplier per epoch
        multiclass:     multi-class flag
        callback:       optional function(w, epoch) called after each epoch
        track_variance: whether to track per-epoch gradient variance

    Returns:
        w:          final weights
        variances:  list of per-epoch variance estimates (only if track_variance=True)
    """
    n = len(y)
    t = 0
    variances = []

    for epoch in range(n_epochs):
        w, t, epoch_var = sgd_epoch_decay(
            w, X, y, lr0, lam, n, t, b, multiclass, track_variance
        )

        if track_variance:
            variances.append(epoch_var)
        if callback:
            callback(w, epoch)

    if track_variance:
        return w, variances
    return w


# ---------------------------------------------------------------------------
# Warm-start Helper
# ---------------------------------------------------------------------------

def warm_start(X, y, lam, multiclass=False, n_epochs=1, lr=0.01):
    """Run a short SGD warm-start before the main optimizer.

    Args:
        X:          feature matrix (n, d)
        y:          label vector (n,)
        lam:        L2 regularization strength
        multiclass: multi-class flag
        n_epochs:   number of warm-start epochs (1 for convex, 10 for NN)
        lr:         learning rate for warm-start

    Returns:
        warmed-up weight vector/matrix (initialized at zero before warm-start)
    """
    d = X.shape[1]
    if multiclass:
        K = len(np.unique(y))
        w = np.zeros((d, K))
    else:
        w = np.zeros(d)

    return sgd_constant(w, X, y, lr, lam, n_epochs, multiclass, track_variance=False)


# ---------------------------------------------------------------------------
# Effective Passes Counter
# ---------------------------------------------------------------------------

def count_effective_passes_sgd(n_epochs):
    """Return effective passes for SGD (1 epoch = 1 pass = n gradient evals)."""
    return n_epochs