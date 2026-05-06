"""
sgd.py — SGD Baselines (Constant & Decaying Learning Rates)

Implements two SGD variants used as baselines in the paper:
1. SGD (Constant eta): Fixed learning rate throughout training
2. SGD (Best / Decaying): eta_t = eta_0 / (1 + a * eta_0 * t)
"""

import numpy as np
from models.logistic import stoch_grad_binary, stoch_grad_multiclass


def _get_stoch_grad_fn(multiclass):
    """Return appropriate stochastic gradient function."""
    return stoch_grad_multiclass if multiclass else stoch_grad_binary


# ---------------------------------------------------------------------------
# SGD with Constant Learning Rate
# ---------------------------------------------------------------------------

def sgd_epoch_constant(w, X, y, lr, lam, multiclass=False, batch_size=1):
    """Run 1 epoch of SGD with constant learning rate.

    Args:
        w: weight vector/matrix
        X: feature matrix (n, d)
        y: label vector (n,)
        lr: constant learning rate
        lam: L2 regularization strength
        multiclass: whether multi-class
        batch_size: mini-batch size (1 for pure SGD)

    Returns:
        updated w after 1 epoch
    """
    n = len(y)
    stoch_grad = _get_stoch_grad_fn(multiclass)

    indices = np.random.permutation(n)

    for i in indices:
        xi = X[i]
        yi = y[i]
        g = stoch_grad(w, xi, yi, lam)
        w = w - lr * g

    return w


def sgd_constant(w, X, y, lr, lam, n_epochs, multiclass=False,
                 batch_size=1, callback=None):
    """Run multiple epochs of SGD with constant learning rate.

    Args:
        w: initial weights
        X: feature matrix
        y: labels
        lr: constant learning rate
        lam: regularization
        n_epochs: number of epochs to run
        multiclass: multi-class flag
        batch_size: mini-batch size
        callback: optional function(w, epoch) called after each epoch

    Returns:
        final weights
    """
    for epoch in range(n_epochs):
        w = sgd_epoch_constant(w, X, y, lr, lam, multiclass, batch_size)
        if callback:
            callback(w, epoch)
    return w


# ---------------------------------------------------------------------------
# SGD with Decaying Learning Rate  (SGD-best)
# ---------------------------------------------------------------------------

def sgd_epoch_decay(w, X, y, lr0, lam, t_start, a, multiclass=False, batch_size=1):
    """Run 1 epoch of SGD with decaying learning rate.

    Learning rate schedule: eta(t) = eta_0 / (1 + a * eta_0 * t)
    where t = total gradient evaluations so far.

    Args:
        w: weight vector/matrix
        X: feature matrix (n, d)
        y: label vector (n,)
        lr0: initial learning rate eta_0
        lam: L2 regularization strength
        t_start: total gradient evaluations before this epoch
        a: decay parameter
        multiclass: whether multi-class
        batch_size: mini-batch size

    Returns:
        (updated w, t_end) where t_end = t_start + n/batch_size
    """
    n = len(y)
    stoch_grad = _get_stoch_grad_fn(multiclass)

    indices = np.random.permutation(n)
    t = t_start

    for i in indices:
        # Current learning rate
        lr_t = lr0 / (1.0 + a * lr0 * t)

        xi = X[i]
        yi = y[i]
        g = stoch_grad(w, xi, yi, lam)
        w = w - lr_t * g

        t += 1

    return w, t


def sgd_decay(w, X, y, lr0, lam, n_epochs, a, multiclass=False,
              batch_size=1, callback=None):
    """Run multiple epochs of SGD with decaying learning rate.

    Args:
        w: initial weights
        X: feature matrix
        y: labels
        lr0: initial learning rate
        lam: regularization
        n_epochs: number of epochs
        a: decay parameter
        multiclass: multi-class flag
        batch_size: mini-batch size
        callback: optional function(w, epoch, lr_current)

    Returns:
        final weights
    """
    t = 0
    for epoch in range(n_epochs):
        w, t = sgd_epoch_decay(w, X, y, lr0, lam, t, a, multiclass, batch_size)
        if callback:
            lr_current = lr0 / (1.0 + a * lr0 * t)
            callback(w, epoch, lr_current)
    return w


# ---------------------------------------------------------------------------
# Warm-start Helper
# ---------------------------------------------------------------------------

def warm_start(X, y, lam, multiclass=False, n_epochs=1, lr=0.01):
    """Run SGD warm-start.

    Args:
        X: feature matrix
        y: labels
        lam: regularization
        multiclass: multi-class flag
        n_epochs: number of warm-start epochs (1 for convex, 10 for NN)
        lr: learning rate for warm-start

    Returns:
        warmed-up weights
    """
    d = X.shape[1]
    if multiclass:
        K = len(np.unique(y))
        w = np.zeros((d, K))
    else:
        w = np.zeros(d)

    w = sgd_constant(w, X, y, lr, lam, n_epochs, multiclass)
    return w


# ---------------------------------------------------------------------------
# Effective Passes Counting
# ---------------------------------------------------------------------------

def count_effective_passes_sgd(n_epochs, n_samples):
    """For SGD, 1 epoch = 1 effective pass."""
    return n_epochs
