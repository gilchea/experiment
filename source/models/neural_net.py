"""
neural_net.py — Neural Network with 1 Hidden Layer (Non-convex)

Architecture:
  Input (d) → Linear → Sigmoid → Linear → Softmax (K classes)

This is a NON-CONVEX problem due to the hidden layer, used to verify
SVRG's performance on complex loss surfaces.

Loss: Cross-entropy + L2 regularization on all weights.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.clip(x, -100, 100)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


def softmax(logits):
    """Numerically stable softmax.

    Args:
        logits: (n, K) or (K,) array

    Returns:
        probabilities of same shape
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Parameter Initialization
# ---------------------------------------------------------------------------

def init_parameters(d, n_hidden=100, n_classes=10, seed=42):
    """Initialize weights and biases for a 1-hidden-layer network.

    Uses Xavier initialization for weights, zero for biases.

    Args:
        d: input dimension
        n_hidden: number of hidden units
        n_classes: number of output classes
        seed: random seed for reproducibility

    Returns:
        dict with keys: 'W1', 'b1', 'W2', 'b2'
          W1: (d, n_hidden)
          b1: (n_hidden,)
          W2: (n_hidden, n_classes)
          b2: (n_classes,)
    """
    rng = np.random.RandomState(seed)

    params = {
        'W1': rng.randn(d, n_hidden) * np.sqrt(2.0 / d),
        'b1': np.zeros(n_hidden),
        'W2': rng.randn(n_hidden, n_classes) * np.sqrt(2.0 / n_hidden),
        'b2': np.zeros(n_classes),
    }
    return params


# ---------------------------------------------------------------------------
# Forward Pass
# ---------------------------------------------------------------------------

def forward(params, X):
    """Forward pass through the network.

    Args:
        params: dict with 'W1', 'b1', 'W2', 'b2'
        X: input matrix (n, d)

    Returns:
        cache: dict with intermediate values for backward pass
          'z1': (n, n_hidden)  — pre-activation of hidden layer
          'a1': (n, n_hidden)  — post-activation (sigmoid)
          'z2': (n, n_classes) — pre-activation of output layer (logits)
          'probs': (n, n_classes) — softmax probabilities
    """
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    z1 = X @ W1 + b1          # (n, n_hidden)
    a1 = sigmoid(z1)          # (n, n_hidden)
    z2 = a1 @ W2 + b2         # (n, n_classes)
    probs = softmax(z2)       # (n, n_classes)

    cache = {
        'z1': z1,
        'a1': a1,
        'z2': z2,
        'probs': probs,
    }
    return cache


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def loss(params, X, y, lam):
    """Cross-entropy loss with L2 regularization.

    Args:
        params: dict with 'W1', 'b1', 'W2', 'b2'
        X: input matrix (n, d)
        y: label vector (n,) with values in {0, ..., K-1}
        lam: L2 regularization strength

    Returns:
        scalar loss value
    """
    n = len(y)
    cache = forward(params, X)
    probs = cache['probs']

    # Cross-entropy
    correct_log_probs = -np.log(probs[np.arange(n), y] + 1e-15)
    loss_data = np.mean(correct_log_probs)

    # L2 regularization (on weights only, not biases)
    W1, W2 = params['W1'], params['W2']
    reg = 0.5 * lam * (np.sum(W1 * W1) + np.sum(W2 * W2))

    return loss_data + reg


# ---------------------------------------------------------------------------
# Full Gradient (Backpropagation)
# ---------------------------------------------------------------------------

def full_grad(params, X, y, lam):
    """Compute full gradient via backpropagation.

    Args:
        params: dict with 'W1', 'b1', 'W2', 'b2'
        X: input matrix (n, d)
        y: label vector (n,) with values in {0, ..., K-1}
        lam: L2 regularization strength

    Returns:
        grad: dict with same keys as params, containing gradients
    """
    n = len(y)
    K = params['W2'].shape[1]
    cache = forward(params, X)
    probs = cache['probs']
    a1 = cache['a1']
    z1 = cache['z1']

    # ── Output layer gradients ──
    # dL/dz2 = probs - one_hot  (softmax + cross-entropy derivative)
    one_hot = np.zeros((n, K))
    one_hot[np.arange(n), y] = 1.0
    dz2 = (probs - one_hot) / n           # (n, K), averaged over n

    # dL/dW2 = a1^T @ dz2 + lam * W2
    dW2 = a1.T @ dz2 + lam * params['W2']  # (n_hidden, K)
    # dL/db2 = sum(dz2, axis=0)
    db2 = np.sum(dz2, axis=0)              # (K,)

    # ── Hidden layer gradients ──
    # dL/da1 = dz2 @ W2^T
    da1 = dz2 @ params['W2'].T             # (n, n_hidden)
    # dL/dz1 = dL/da1 * sigmoid'(z1)
    dz1 = da1 * sigmoid_grad(z1)           # (n, n_hidden)

    # dL/dW1 = X^T @ dz1 + lam * W1
    dW1 = X.T @ dz1 + lam * params['W1']   # (d, n_hidden)
    # dL/db1 = sum(dz1, axis=0)
    db1 = np.sum(dz1, axis=0)              # (n_hidden,)

    grad = {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2,
    }
    return grad


def stoch_grad(params, xi, yi, lam):
    """Compute stochastic gradient for a mini-batch.
    
    This returns the average gradient over the mini-batch, which matches
    the scale of the full gradient (mu) used in SVRG.

    Args:
        params: dict with 'W1', 'b1', 'W2', 'b2'
        xi: mini-batch features (batch_size, d)
        yi: mini-batch labels (batch_size,)
        lam: L2 regularization strength

    Returns:
        grad: dict with same keys as params
    """
    return full_grad(params, xi, yi, lam)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(params, X):
    """Predict class labels.

    Args:
        params: dict with 'W1', 'b1', 'W2', 'b2'
        X: input matrix (n, d)

    Returns:
        predictions: (n,) integer class labels
    """
    cache = forward(params, X)
    return np.argmax(cache['probs'], axis=1)


def compute_error(params, X, y):
    """Compute classification error rate (percentage).

    Args:
        params: dict with 'W1', 'b1', 'W2', 'b2'
        X: input matrix (n, d)
        y: label vector (n,)

    Returns:
        error rate as percentage
    """
    preds = predict(params, X)
    return np.mean(preds != y) * 100


# ---------------------------------------------------------------------------
# Parameter Utilities
# ---------------------------------------------------------------------------

def flatten_params(params):
    """Flatten all parameters into a single 1D vector.

    Args:
        params: dict with 'W1', 'b1', 'W2', 'b2'

    Returns:
        1D numpy array
    """
    return np.concatenate([
        params['W1'].ravel(),
        params['b1'].ravel(),
        params['W2'].ravel(),
        params['b2'].ravel(),
    ])


def unflatten_params(theta, d, n_hidden, n_classes):
    """Unflatten a 1D vector back into parameter dict.

    Args:
        theta: 1D numpy array
        d: input dimension
        n_hidden: number of hidden units
        n_classes: number of output classes

    Returns:
        params dict
    """
    idx = 0
    size_W1 = d * n_hidden
    size_b1 = n_hidden
    size_W2 = n_hidden * n_classes
    size_b2 = n_classes

    W1 = theta[idx:idx + size_W1].reshape(d, n_hidden)
    idx += size_W1
    b1 = theta[idx:idx + size_b1]
    idx += size_b1
    W2 = theta[idx:idx + size_W2].reshape(n_hidden, n_classes)
    idx += size_W2
    b2 = theta[idx:idx + size_b2]

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def copy_params(params):
    """Deep copy parameter dict."""
    return {k: v.copy() for k, v in params.items()}


def add_params(params_a, params_b, scale=1.0):
    """Return params_a + scale * params_b (element-wise)."""
    result = {}
    for k in params_a:
        result[k] = params_a[k] + scale * params_b[k]
    return result


def scale_params(params, scale):
    """Return scale * params (element-wise)."""
    return {k: v * scale for k, v in params.items()}
