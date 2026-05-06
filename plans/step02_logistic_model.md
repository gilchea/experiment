# Step 02: `models/logistic.py` — L2-Regularized Logistic Regression

## Objective

Implement the convex objective function $P(w)$ and its gradients (full + stochastic) for L2-regularized multi-class logistic regression, as used in the NIPS 2013 paper.

## Mathematical Formulation

### Objective (Loss) Function

$$P(w) = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i w^\top x_i)) + \frac{\lambda}{2} \|w\|^2$$

Where:
- $y_i \in \{-1, +1\}$ for binary (RCV1, Covtype)
- $y_i \in \{0..9\}$ for multi-class (MNIST, CIFAR-10) — **one-vs-rest** or **multinomial**?

**Paper uses multi-class logistic regression** for MNIST and CIFAR-10. For multi-class with $K$ classes:
- $W \in \mathbb{R}^{d \times K}$ (weight matrix)
- $y_i \in \{0, 1, \dots, K-1\}$

$$P(W) = \frac{1}{n} \sum_{i=1}^n \left[ -w_{y_i}^\top x_i + \log\left( \sum_{j=0}^{K-1} \exp(w_j^\top x_i) \right) \right] + \frac{\lambda}{2} \|W\|_F^2$$

## File Structure

```
experiment/
└── models/
    └── logistic.py      ← THIS FILE
```

## Detailed Implementation

### 1. Binary Logistic Regression (for RCV1, Covtype)

```python
import numpy as np

def sigmoid(x):
    """Numerically stable sigmoid function."""
    # Clip to avoid overflow in np.exp
    x = np.clip(x, -100, 100)
    return 1.0 / (1.0 + np.exp(-x))

def loss_binary(w, X, y, lam):
    """P(w) for binary logistic regression.
    
    Args:
        w: weight vector (d,)
        X: feature matrix (n, d) — dense or sparse
        y: label vector (n,) with values in {-1, +1}
        lam: L2 regularization strength
    
    Returns:
        scalar loss value
    """
    n = len(y)
    margins = y * (X @ w)           # shape (n,)
    # Use log1p(exp(-margins)) for numerical stability
    log_part = np.log1p(np.exp(-margins))
    return np.mean(log_part) + 0.5 * lam * np.dot(w, w)

def full_grad_binary(w, X, y, lam):
    """Full gradient ∇P(w).
    
    ∇P(w) = (1/n) Σ ∇ψ_i(w) + λw
    ∇ψ_i(w) = -y_i * σ(-y_i w^T x_i) * x_i + λw
    
    Returns:
        gradient vector (d,)
    """
    n = len(y)
    margins = y * (X @ w)
    # sigmoid(-margin) = 1 / (1 + exp(margin))
    # For logistic loss: d/dw log(1+exp(-y*w^T x)) = -y * sigmoid(-y*w^T x) * x
    coefs = -y * sigmoid(-margins)  # shape (n,)
    grad_data = X.T @ coefs / n     # (d,)
    return grad_data + lam * w

def stoch_grad_binary(w, xi, yi, lam):
    """Stochastic gradient ∇ψ_i(w) for a single sample.
    
    Args:
        w: weight vector (d,)
        xi: single sample features (d,) — dense or sparse
        yi: single label in {-1, +1}
        lam: L2 regularization strength
    
    Returns:
        gradient vector (d,)
    """
    margin = yi * (xi @ w)
    coef = -yi * sigmoid(-margin)
    return coef * xi + lam * w
```

### 2. Multi-class Logistic Regression (for MNIST, CIFAR-10)

```python
def softmax(logits):
    """Numerically stable softmax.
    
    Args:
        logits: (n, K) or (K,) array of logits
    
    Returns:
        probabilities of same shape
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

def loss_multiclass(W, X, y, lam):
    """P(W) for multi-class logistic regression (softmax + cross-entropy).
    
    Args:
        W: weight matrix (d, K)
        X: feature matrix (n, d)
        y: label vector (n,) with values in {0, ..., K-1}
        lam: L2 regularization strength
    
    Returns:
        scalar loss value
    """
    n = len(y)
    K = W.shape[1]
    logits = X @ W                      # (n, K)
    probs = softmax(logits)             # (n, K)
    
    # Cross-entropy loss: -log(probs[i, y_i])
    # Use log of softmax directly for stability
    log_probs = logits - np.max(logits, axis=1, keepdims=True)
    log_probs -= np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))
    
    correct_log_probs = log_probs[np.arange(n), y]
    loss = -np.mean(correct_log_probs)
    reg = 0.5 * lam * np.sum(W * W)
    return loss + reg

def full_grad_multiclass(W, X, y, lam):
    """Full gradient for multi-class logistic regression.
    
    ∇P(W) = (1/n) Σ (softmax(W^T x_i) - e_{y_i}) ⊗ x_i + λW
    
    Returns:
        gradient matrix (d, K)
    """
    n = len(y)
    K = W.shape[1]
    logits = X @ W                      # (n, K)
    probs = softmax(logits)             # (n, K)
    
    # Gradient: (probs - one_hot) * x_i averaged
    # For each sample i: outer product x_i ⊗ (probs[i] - one_hot[i])
    # Efficient: X.T @ (probs - one_hot) / n
    one_hot = np.zeros((n, K))
    one_hot[np.arange(n), y] = 1.0
    
    grad_data = X.T @ (probs - one_hot) / n   # (d, K)
    return grad_data + lam * W

def stoch_grad_multiclass(W, xi, yi, lam):
    """Stochastic gradient for a single sample.
    
    Args:
        W: weight matrix (d, K)
        xi: single sample (d,)
        yi: label in {0, ..., K-1}
        lam: L2 regularization strength
    
    Returns:
        gradient matrix (d, K)
    """
    K = W.shape[1]
    logits = xi @ W                     # (K,)
    probs = softmax(logits)             # (K,)
    
    one_hot = np.zeros(K)
    one_hot[yi] = 1.0
    
    grad = np.outer(xi, probs - one_hot)  # (d, K)
    return grad + lam * W
```

### 3. Unified Interface

```python
def loss(w, X, y, lam, multiclass=False):
    """Unified loss function."""
    if multiclass:
        return loss_multiclass(w, X, y, lam)
    return loss_binary(w, X, y, lam)

def full_grad(w, X, y, lam, multiclass=False):
    """Unified full gradient."""
    if multiclass:
        return full_grad_multiclass(w, X, y, lam)
    return full_grad_binary(w, X, y, lam)

def stoch_grad(w, xi, yi, lam, multiclass=False):
    """Unified stochastic gradient."""
    if multiclass:
        return stoch_grad_multiclass(w, xi, yi, lam)
    return stoch_grad_binary(w, xi, yi, lam)
```

## Verification: Numerical Gradient Check

Run this to verify correctness:

```python
def numerical_grad_check():
    """Verify gradients via finite differences."""
    np.random.seed(42)
    n, d = 100, 10
    X = np.random.randn(n, d)
    w = np.random.randn(d)
    y = np.sign(np.random.randn(n))
    lam = 1e-4
    
    eps = 1e-6
    analytic = full_grad_binary(w, X, y, lam)
    
    # Numerical gradient
    numeric = np.zeros(d)
    for j in range(d):
        w_plus = w.copy(); w_plus[j] += eps
        w_minus = w.copy(); w_minus[j] -= eps
        numeric[j] = (loss_binary(w_plus, X, y, lam) - loss_binary(w_minus, X, y, lam)) / (2 * eps)
    
    diff = np.max(np.abs(analytic - numeric))
    print(f"Max gradient difference: {diff:.2e}")
    assert diff < 1e-7, f"Gradient check failed: {diff}"
```

## Constraints & Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| λ (MNIST) | $10^{-4}$ | For both binary and multi-class |
| λ (CIFAR-10) | $10^{-3}$ | For both binary and multi-class |
| λ (RCV1) | $10^{-5}$ | Binary only |
| λ (Covtype) | $10^{-5}$ | Binary only |
| Multi-class flag | `True` for MNIST, CIFAR-10; `False` for RCV1, Covtype |

## Verification Checklist

- [ ] `loss_binary()` returns scalar, decreases when w moves toward optimum
- [ ] `full_grad_binary()` passes numerical gradient check (diff < 1e-7)
- [ ] `stoch_grad_binary()` returns same shape as w
- [ ] `loss_multiclass()` returns scalar, correct for random weights
- [ ] `full_grad_multiclass()` passes numerical gradient check
- [ ] `stoch_grad_multiclass()` returns (d, K) matrix
- [ ] All functions handle both dense and sparse X correctly

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| `np.exp(-margin)` overflow for large negative margins | Use `np.log1p(np.exp(-margins))` for loss; clip for sigmoid |
| Softmax numerical overflow | Subtract max logit before exp |
| Multi-class gradient shape | Ensure outer product `np.outer(xi, probs - one_hot)` |
| Sparse X @ w returns dense | `X @ w` works with sparse X automatically |
| Forgetting regularization in gradient | Must add `lam * w` to data gradient |
