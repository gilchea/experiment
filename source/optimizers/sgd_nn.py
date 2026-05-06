"""
sgd_nn.py — SGD Baselines for Neural Networks (Non-convex)

Implements two SGD variants for the 1-hidden-layer neural network:
1. SGD (Constant eta): Fixed learning rate throughout training
2. SGD (Best / Decaying): eta_t = eta_0 / (1 + a * eta_0 * t)
"""

import numpy as np
from models.neural_net import stoch_grad, copy_params, add_params, scale_params


# ---------------------------------------------------------------------------
# SGD with Constant Learning Rate
# ---------------------------------------------------------------------------

def sgd_nn_epoch_constant(params, X, y, lr, lam):
    """Run 1 epoch of SGD with constant learning rate for NN.

    Args:
        params: dict with 'W1', 'b1', 'W2', 'b2'
        X: feature matrix (n, d)
        y: label vector (n,)
        lr: constant learning rate
        lam: L2 regularization strength

    Returns:
        updated params after 1 epoch
    """
    n = len(y)
    indices = np.random.permutation(n)

    w = copy_params(params)

    for i in indices:
        xi = X[i]
        yi = y[i]
        g = stoch_grad(w, xi, yi, lam)
        w = add_params(w, scale_params(g, -lr))

    return w


# ---------------------------------------------------------------------------
# SGD with Decaying Learning Rate  (SGD-best)
# ---------------------------------------------------------------------------

def sgd_nn_epoch_decay(params, X, y, lr0, lam, t_start, a):
    """Run 1 epoch of SGD with decaying learning rate for NN.

    Learning rate schedule: eta(t) = eta_0 / (1 + a * eta_0 * t)
    where t = total gradient evaluations so far.

    Args:
        params: dict with 'W1', 'b1', 'W2', 'b2'
        X: feature matrix (n, d)
        y: label vector (n,)
        lr0: initial learning rate eta_0
        lam: L2 regularization strength
        t_start: total gradient evaluations before this epoch
        a: decay parameter

    Returns:
        (updated params, t_end) where t_end = t_start + n
    """
    n = len(y)
    indices = np.random.permutation(n)
    t = t_start

    w = copy_params(params)

    for i in indices:
        # Current learning rate
        lr_t = lr0 / (1.0 + a * lr0 * t)

        xi = X[i]
        yi = y[i]
        g = stoch_grad(w, xi, yi, lam)
        w = add_params(w, scale_params(g, -lr_t))

        t += 1

    return w, t


# ---------------------------------------------------------------------------
# Warm-start Helper
# ---------------------------------------------------------------------------

def warm_start_nn(X, y, lam, d, n_hidden=100, n_classes=10,
                  n_epochs=10, lr=0.01, seed=42):
    """Run SGD warm-start for neural network.

    Args:
        X: feature matrix
        y: labels
        lam: regularization
        d: input dimension
        n_hidden: number of hidden units
        n_classes: number of output classes
        n_epochs: number of warm-start epochs
        lr: learning rate for warm-start
        seed: random seed

    Returns:
        warmed-up parameters
    """
    from models.neural_net import init_parameters
    params = init_parameters(d, n_hidden, n_classes, seed=seed)

    for _ in range(n_epochs):
        params = sgd_nn_epoch_constant(params, X, y, lr, lam)

    return params
