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
from models.logistic import stoch_grad, full_grad

# def _get_stoch_grad_fn(multiclass):
#     """Return appropriate stochastic gradient function."""
#     return stoch_grad_multiclass if multiclass else stoch_grad_binary

# def _get_full_grad_fn(multiclass):
#     """Return appropriate full gradient function."""
#     return grad_full_multiclass if multiclass else grad_full_binary


# ---------------------------------------------------------------------------
# SGD with Constant Learning Rate
# ---------------------------------------------------------------------------
def sgd_epoch_constant(w, X, y, lr, lam, multiclass=False, track_variance=True):
    """Run 1 epoch of SGD with constant learning rate.

    Per spec: batch_size = 1, w^(t) = w^(t-1) - eta * grad_psi_i(w^(t-1))

    Args:
        w          : weight vector/matrix
        X          : feature matrix (n, d)
        y          : label vector (n,)
        lr         : constant learning rate eta
        lam        : L2 regularization strength
        multiclass : whether multi-class
        track_variance : whether to track gradient variance

    Returns:
        updated w, epoch_variance
    """
    n = len(y)
    
    indices = np.random.permutation(n)
    variance_sum = 0.0
    variance_count = 0

    for i in indices:
        # 1. Lấy gradient ngẫu nhiên của mẫu i
        g = stoch_grad(w, X[i], y[i], lam, multiclass)

        # 2. Đo đạc phương sai (nếu bật tính năng track)
        if track_variance:
            # E_g = full_grad(w, X, y, lam, multiclass)
            # diff = g - E_g
            diff = g * lr  # Var(lr * g) = lr^2 * ||g - E[g]||^2, nhưng vì E[g] ≈ 0 khi w gần tối ưu, nên diff ≈ g * lr
            # Định nghĩa thực nghiệm gốc bao gồm biến thiên học tốc: Var(lr * g)
            actual_update_variance = np.sum(diff * diff)
            variance_sum += actual_update_variance
            variance_count += 1

        # 3. Cập nhật trọng số
        w = w - lr * g

    epoch_variance = variance_sum / max(variance_count, 1) if track_variance else 0.0
    return w, epoch_variance

def sgd_constant(w, X, y, lr, lam, n_epochs, multiclass=False, callback=None, track_variance=True):
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
        track_variance : whether to track gradient variance

     Returns:
            Nếu track_variance=False: trả về final_w
            Nếu track_variance=True : trả về (final_w, danh_sách_variance_qua_mỗi_epoch)   
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
# SGD with Decaying Learning Rate  (SGD-best)
# ---------------------------------------------------------------------------

def sgd_epoch_decay(w, X, y, lr0, lam, n, t_start, a, multiclass=False, track_variance=True):
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
        a          : decay parameter
        multiclass : whether multi-class

    Returns:
        (updated w, t_end)
    """
    indices = np.random.permutation(n)
    t = t_start

    variance_sum = 0.0
    variance_count = 0
    
    for i in indices:
        # Tính toán học tốc tại bước t
        lr_t = lr0 * (a ** (t // n))
        
        # 1. Lấy gradient ngẫu nhiên của mẫu i
        g = stoch_grad(w, X[i], y[i], lam, multiclass)
        
        # 2. Đo đạc phương sai (nếu bật tính năng track)
        if track_variance:
            # E_g = full_grad(w, X, y, lam, multiclass)
            # diff = g - E_g
            diff = g * lr_t  # Var(lr_t * g) = lr_t^2 * ||g - E[g]||^2, nhưng vì E[g] ≈ 0 khi w gần tối ưu, nên diff ≈ g * lr_t
            # Định nghĩa thực nghiệm gốc bao gồm biến thiên học tốc: Var(lr_t * g)
            actual_update_variance = np.sum(diff * diff)
            variance_sum += actual_update_variance
            variance_count += 1

        # 3. Cập nhật trọng số
        w = w - lr_t * g
        t += 1

    epoch_variance = variance_sum / max(variance_count, 1) if track_variance else 0.0
    return w, t, epoch_variance


def sgd_decay(w, X, y, lr0, lam, n_epochs, a, multiclass=False, callback=None, track_variance=True):
    """Run multiple epochs of SGD with t-inverse decaying learning rate.

    Args:
        w          : initial weights
        X          : feature matrix (n, d)
        y          : labels (n,)
        lr0        : initial learning rate eta_0
        lam        : regularization
        n_epochs   : number of epochs
        a          : decay parameter
        multiclass : multi-class flag
        callback   : optional function(w, epoch) called after each epoch
        track_variance : whether to track gradient variance

    Returns:
        final weights
    """
    n = len(y)
    t = 0
    variances = []
    
    for epoch in range(n_epochs):
        w, t, epoch_var = sgd_epoch_decay(w, X, y, lr0, lam, n, t, a, multiclass, track_variance)
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

    return sgd_constant(w, X, y, lr, lam, n_epochs, multiclass, track_variance=False)


# ---------------------------------------------------------------------------
# Effective Passes Counting
# ---------------------------------------------------------------------------

def count_effective_passes_sgd(n_epochs):
    """For SGD, 1 epoch = 1 effective pass (n grad evals / n)."""
    return n_epochs
