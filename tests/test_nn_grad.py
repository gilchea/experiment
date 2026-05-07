"""
test_nn_grad.py — Numerical Gradient Check for Neural Network

Verifies that the analytical gradients from neural_net.py match
finite-difference approximations.
"""

import numpy as np
import sys
sys.path.insert(0, 'source')

from models.neural_net import (
    init_parameters, forward, loss, full_grad, stoch_grad,
    flatten_params, unflatten_params, copy_params
)


def numerical_grad_full(params, X, y, lam, eps=1e-6):
    """Compute numerical gradient via finite differences for full gradient."""
    theta0 = flatten_params(params)
    grad_flat = np.zeros_like(theta0)

    for i in range(len(theta0)):
        theta_plus = theta0.copy()
        theta_plus[i] += eps
        theta_minus = theta0.copy()
        theta_minus[i] -= eps

        d, n_hidden, n_classes = params['W1'].shape[0], params['W1'].shape[1], params['W2'].shape[1]
        params_plus = unflatten_params(theta_plus, d, n_hidden, n_classes)
        params_minus = unflatten_params(theta_minus, d, n_hidden, n_classes)

        loss_plus = loss(params_plus, X, y, lam)
        loss_minus = loss(params_minus, X, y, lam)

        grad_flat[i] = (loss_plus - loss_minus) / (2 * eps)

    return unflatten_params(grad_flat, params['W1'].shape[0],
                            params['W1'].shape[1], params['W2'].shape[1])


def numerical_grad_stoch(params, xi, yi, lam, eps=1e-6):
    """Compute numerical gradient for stochastic gradient."""
    theta0 = flatten_params(params)
    grad_flat = np.zeros_like(theta0)

    for i in range(len(theta0)):
        theta_plus = theta0.copy()
        theta_plus[i] += eps
        theta_minus = theta0.copy()
        theta_minus[i] -= eps

        d, n_hidden, n_classes = params['W1'].shape[0], params['W1'].shape[1], params['W2'].shape[1]
        params_plus = unflatten_params(theta_plus, d, n_hidden, n_classes)
        params_minus = unflatten_params(theta_minus, d, n_hidden, n_classes)

        # Loss for single sample
        cache_plus = forward(params_plus, xi.reshape(1, -1))
        probs_plus = cache_plus['probs'][0]
        loss_plus = -np.log(probs_plus[yi] + 1e-15)
        reg_plus = 0.5 * lam * (np.sum(params_plus['W1']**2) + np.sum(params_plus['W2']**2))

        cache_minus = forward(params_minus, xi.reshape(1, -1))
        probs_minus = cache_minus['probs'][0]
        loss_minus = -np.log(probs_minus[yi] + 1e-15)
        reg_minus = 0.5 * lam * (np.sum(params_minus['W1']**2) + np.sum(params_minus['W2']**2))

        grad_flat[i] = ((loss_plus + reg_plus) - (loss_minus + reg_minus)) / (2 * eps)

    return unflatten_params(grad_flat, params['W1'].shape[0],
                            params['W1'].shape[1], params['W2'].shape[1])


def check_grad(analytical, numerical, name="gradient"):
    """Compare analytical and numerical gradients."""
    max_diff = 0.0
    rel_diff = 0.0
    for key in analytical:
        a = analytical[key].ravel()
        n = numerical[key].ravel()
        diff = np.abs(a - n)
        max_diff = max(max_diff, diff.max())
        denom = np.maximum(np.abs(a), np.abs(n))
        denom = np.maximum(denom, 1e-8)
        rel_diff = max(rel_diff, (diff / denom).max())

    return max_diff, rel_diff


if __name__ == '__main__':
    print("=" * 60)
    print("Neural Network Gradient Check")
    print("=" * 60)

    # Small test case
    np.random.seed(42)
    d, n_hidden, n_classes = 10, 5, 3
    n_samples = 20
    lam = 1e-3

    X = np.random.randn(n_samples, d)
    y = np.random.randint(0, n_classes, n_samples)

    params = init_parameters(d, n_hidden, n_classes, seed=42)

    # --- Test 1: Full Gradient ---
    print("\n[Test 1] Full Gradient Check...")
    analytical = full_grad(params, X, y, lam)
    numerical = numerical_grad_full(params, X, y, lam)
    max_diff, rel_diff = check_grad(analytical, numerical)

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")

    if rel_diff < 1e-5:
        print("  >>> PASSED (relative diff < 1e-5)")
    else:
        print("  >>> FAILED (relative diff too large)")

    # --- Test 2: Stochastic Gradient ---
    print("\n[Test 2] Stochastic Gradient Check...")
    xi = X[0]
    yi = y[0]

    analytical_sg = stoch_grad(params, xi, yi, lam)
    numerical_sg = numerical_grad_stoch(params, xi, yi, lam)
    max_diff_sg, rel_diff_sg = check_grad(analytical_sg, numerical_sg)

    print(f"  Max absolute difference: {max_diff_sg:.2e}")
    print(f"  Max relative difference: {rel_diff_sg:.2e}")

    if rel_diff_sg < 1e-5:
        print("  >>> PASSED (relative diff < 1e-5)")
    else:
        print("  >>> FAILED (relative diff too large)")

    # --- Test 3: Loss decreases with gradient descent ---
    print("\n[Test 3] Loss Decrease with Gradient Descent...")
    params_test = copy_params(params)
    lr = 0.1
    losses = []
    for i in range(100):
        g = full_grad(params_test, X, y, lam)
        for k in params_test:
            params_test[k] -= lr * g[k]
        losses.append(loss(params_test, X, y, lam))

    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss:   {losses[-1]:.6f}")

    if losses[-1] < losses[0]:
        print("  >>> PASSED (loss decreased)")
    else:
        print("  >>> FAILED (loss did not decrease)")

    print("\n" + "=" * 60)
    print("Gradient Check Complete")
    print("=" * 60)
