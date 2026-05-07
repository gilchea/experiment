"""
sgd_nn.py — SGD Baselines for Neural Networks (Non-convex)

Implements two SGD variants for the 1-hidden-layer neural network:
1. SGD (Constant): Fixed learning rate.
2. SGD-best (Decaying): t-inverse schedule eta(t) = eta_0 / (1 + b * t/n)

Per PROCEDURE_SGD_.md:
    - Batch size = 10 for non-convex (neural network) problems.
    - X-axis in plots = #grad / n (effective passes).
"""

import numpy as np
from models.neural_net import stoch_grad, copy_params, add_params, scale_params


# ---------------------------------------------------------------------------
# SGD with Constant Learning Rate
# ---------------------------------------------------------------------------

def sgd_nn_epoch_constant(params, X, y, lr, lam, batch_size=10):
    """Run 1 epoch of SGD with constant learning rate for NN.

    Per spec: batch_size = 10 for non-convex problems.

    Args:
        params     : dict with 'W1', 'b1', 'W2', 'b2'
        X          : feature matrix (n, d)
        y          : label vector (n,)
        lr         : constant learning rate
        lam        : L2 regularization strength
        batch_size : mini-batch size (10 per spec for NN)

    Returns:
        updated params after 1 epoch
    """
    n = len(y)
    w = copy_params(params)
    indices = np.random.permutation(n)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        xi = X[batch_idx]
        yi = y[batch_idx]
        g = stoch_grad(w, xi, yi, lam)
        w = add_params(w, scale_params(g, -lr))

    return w


def sgd_nn_constant(params, X, y, lr, lam, n_epochs, batch_size=10, callback=None):
    """Run multiple epochs of SGD with constant learning rate for NN.

    Args:
        params     : initial parameters dict
        X          : feature matrix (n, d)
        y          : labels (n,)
        lr         : constant learning rate
        lam        : regularization
        n_epochs   : number of epochs
        batch_size : mini-batch size (10 per spec)
        callback   : optional function(params, epoch) called after each epoch

    Returns:
        final params
    """
    for epoch in range(n_epochs):
        params = sgd_nn_epoch_constant(params, X, y, lr, lam, batch_size)
        if callback:
            callback(params, epoch)
    return params


# ---------------------------------------------------------------------------
# SGD with Decaying Learning Rate  (SGD-best)
# ---------------------------------------------------------------------------

def sgd_nn_epoch_decay(params, X, y, lr0, lam, n, t_start, b, batch_size=10):
    """Run 1 epoch of SGD with t-inverse decaying learning rate for NN.

    Per PROCEDURE_SGD_.md, t-inverse schedule:
        eta(t) = eta_0 / (1 + b * t / n)
    where t = total mini-batch steps so far, n = dataset size.

    Args:
        params     : dict with 'W1', 'b1', 'W2', 'b2'
        X          : feature matrix (n, d)
        y          : label vector (n,)
        lr0        : initial learning rate eta_0
        lam        : L2 regularization strength
        n          : dataset size (for normalizing t in schedule)
        t_start    : total steps before this epoch
        b          : decay parameter
        batch_size : mini-batch size (10 per spec)

    Returns:
        (updated params, t_end)
    """
    w = copy_params(params)
    indices = np.random.permutation(n)
    t = t_start

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        xi = X[batch_idx]
        yi = y[batch_idx]

        lr_t = lr0 / (1.0 + b * t / n)    # t-inverse schedule per spec
        g = stoch_grad(w, xi, yi, lam)
        w = add_params(w, scale_params(g, -lr_t))
        t += 1

    return w, t


def sgd_nn_decay(params, X, y, lr0, lam, n_epochs, b, batch_size=10, callback=None):
    """Run multiple epochs of SGD with t-inverse decaying learning rate for NN.

    Args:
        params     : initial parameters dict
        X          : feature matrix (n, d)
        y          : labels (n,)
        lr0        : initial learning rate eta_0
        lam        : regularization
        n_epochs   : number of epochs
        b          : decay parameter
        batch_size : mini-batch size (10 per spec)
        callback   : optional function(params, epoch) called after each epoch

    Returns:
        final params
    """
    n = len(y)
    t = 0
    for epoch in range(n_epochs):
        params, t = sgd_nn_epoch_decay(params, X, y, lr0, lam, n, t, b, batch_size)
        if callback:
            callback(params, epoch)
    return params


# ---------------------------------------------------------------------------
# Warm-start Helper
# ---------------------------------------------------------------------------

def warm_start_nn(X, y, lam, d, n_hidden=100, n_classes=10,
                  n_epochs=10, lr=0.01, seed=42):
    """Run SGD warm-start for neural network (10 epochs per paper setup).

    Args:
        X         : feature matrix (n, d)
        y         : labels (n,)
        lam       : regularization
        d         : input dimension
        n_hidden  : number of hidden units
        n_classes : number of output classes
        n_epochs  : number of warm-start epochs (10 for NN per paper)
        lr        : learning rate for warm-start
        seed      : random seed

    Returns:
        warmed-up parameters
    """
    from models.neural_net import init_parameters
    params = init_parameters(d, n_hidden, n_classes, seed=seed)
    return sgd_nn_constant(params, X, y, lr, lam, n_epochs)
