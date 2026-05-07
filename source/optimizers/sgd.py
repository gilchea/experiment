"""
sgd.py — SGD Baselines (Constant & Decaying Learning Rates)

Implements two SGD variants used as baselines in the paper (NIPS 2013):
1. SGD (Constant): Fixed learning rate throughout training.
2. SGD-best (Decaying): t-inverse schedule eta(t) = eta_0 / (1 + b * t/n)

Per PROCEDURE_SGD_.md:
    - Batch size = 1 for convex problems (logistic regression).
    - X-axis in plots = #grad / n (effective passes).
"""

import numpy as np
from models.logistic import stoch_grad_binary, stoch_grad_multiclass


def _get_stoch_grad_fn(multiclass):
    """Return appropriate stochastic gradient function."""
    return stoch_grad_multiclass if multiclass else stoch_grad_binary


# ---------------------------------------------------------------------------
# SGD with Constant Learning Rate
# ---------------------------------------------------------------------------

def sgd_epoch_constant(w, X, y, lr, lam, multiclass=False):
    """Run 1 epoch of SGD with constant learning rate.

    Per spec: batch_size = 1, w^(t) = w^(t-1) - eta * grad_psi_i(w^(t-1))

    Args:
        w          : weight vector/matrix
        X          : feature matrix (n, d)
        y          : label vector (n,)
        lr         : constant learning rate eta
        lam        : L2 regularization strength
        multiclass : whether multi-class

    Returns:
        updated w after 1 epoch
    """
    n = len(y)
    stoch_grad = _get_stoch_grad_fn(multiclass)
    indices = np.random.permutation(n)

    for i in indices:
        g = stoch_grad(w, X[i], y[i], lam)
        w = w - lr * g

    return w


def sgd_constant(w, X, y, lr, lam, n_epochs, multiclass=False, callback=None):
    """Run multiple epochs of SGD with constant learning rate.

    Args:
        w          : initial weights
        X          : feature matrix (n, d)
        y          : labels (n,)
        lr         : constant learning rate
        lam        : regularization
        n_epochs   : number of epochs
        multiclass : multi-class flag
        callback   : optional function(w, epoch) called after each epoch

    Returns:
        final weights
    """
    for epoch in range(n_epochs):
        w = sgd_epoch_constant(w, X, y, lr, lam, multiclass)
        if callback:
            callback(w, epoch)
    return w


# ---------------------------------------------------------------------------
# SGD with Decaying Learning Rate  (SGD-best)
# ---------------------------------------------------------------------------

def sgd_epoch_decay(w, X, y, lr0, lam, n, t_start, b, multiclass=False):
    """Run 1 epoch of SGD with t-inverse decaying learning rate.

    Per PROCEDURE_SGD_.md, t-inverse schedule:
        eta(t) = eta_0 / (1 + b * t / n)
    where t = total gradient steps so far, n = dataset size.

    Args:
        w          : weight vector/matrix
        X          : feature matrix (n, d)
        y          : label vector (n,)
        lr0        : initial learning rate eta_0
        lam        : L2 regularization strength
        n          : dataset size (used for normalizing t in schedule)
        t_start    : total gradient steps before this epoch
        b          : decay parameter
        multiclass : whether multi-class

    Returns:
        (updated w, t_end)
    """
    stoch_grad = _get_stoch_grad_fn(multiclass)
    indices = np.random.permutation(n)
    t = t_start

    for i in indices:
        lr_t = lr0 / (1.0 + b * t / n)    # t-inverse schedule per spec
        g = stoch_grad(w, X[i], y[i], lam)
        w = w - lr_t * g
        t += 1

    return w, t


def sgd_decay(w, X, y, lr0, lam, n_epochs, b, multiclass=False, callback=None):
    """Run multiple epochs of SGD with t-inverse decaying learning rate.

    Args:
        w          : initial weights
        X          : feature matrix (n, d)
        y          : labels (n,)
        lr0        : initial learning rate eta_0
        lam        : regularization
        n_epochs   : number of epochs
        b          : decay parameter
        multiclass : multi-class flag
        callback   : optional function(w, epoch) called after each epoch

    Returns:
        final weights
    """
    n = len(y)
    t = 0
    for epoch in range(n_epochs):
        w, t = sgd_epoch_decay(w, X, y, lr0, lam, n, t, b, multiclass)
        if callback:
            callback(w, epoch)
    return w


# ---------------------------------------------------------------------------
# Warm-start Helper
# ---------------------------------------------------------------------------

def warm_start(X, y, lam, multiclass=False, n_epochs=1, lr=0.01):
    """Run SGD warm-start (1-10 epochs per paper setup).

    Args:
        X          : feature matrix (n, d)
        y          : labels (n,)
        lam        : regularization
        multiclass : multi-class flag
        n_epochs   : number of warm-start epochs (1 for convex, 10 for NN)
        lr         : learning rate for warm-start

    Returns:
        warmed-up weights
    """
    d = X.shape[1]
    if multiclass:
        K = len(np.unique(y))
        w = np.zeros((d, K))
    else:
        w = np.zeros(d)

    return sgd_constant(w, X, y, lr, lam, n_epochs, multiclass)


# ---------------------------------------------------------------------------
# Effective Passes Counting
# ---------------------------------------------------------------------------

def count_effective_passes_sgd(n_epochs):
    """For SGD, 1 epoch = 1 effective pass (n grad evals / n)."""
    return n_epochs
