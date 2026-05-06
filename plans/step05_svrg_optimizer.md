# Step 05: `optimizers/svrg.py` — SVRG Algorithm (Core Contribution)

## Objective

Implement the **Stochastic Variance Reduced Gradient** algorithm exactly as described in Algorithm 1 of Johnson & Zhang (NIPS 2013). This is the most critical component of the entire experiment.

## Algorithm Overview

```
Algorithm 1: SVRG
─────────────────────────────────────────────
Input: η (step size), m (inner loop length), 
       initial weights w̃₀

for s = 1, 2, ... do
    μ̃ = (1/n) Σᵢ ∇ψᵢ(w̃ₛ₋₁)          ← Full gradient
    w₀ = w̃ₛ₋₁
    
    for t = 1 to m do
        Pick iₜ uniformly from {1,...,n}
        wₜ = wₜ₋₁ - η(∇ψᵢₜ(wₜ₋₁) - ∇ψᵢₜ(w̃ₛ₋₁) + μ̃)
    end for
    
    Option I:  w̃ₛ = wₘ
    Option II: w̃ₛ = wₜ for random t ∈ {0,...,m-1}
end for
```

## Memory Optimization (Scalar Storage)

For **linear prediction models** (logistic regression), we can optimize:

Instead of recomputing $\nabla \psi_i(\tilde{w}) = \phi'_i(\tilde{w}^\top x_i) \cdot x_i + \lambda \tilde{w}$ in the inner loop, we:
1. Precompute $z_i = \tilde{w}^\top x_i$ for all $i$ (scalars)
2. Precompute $\phi'_i(z_i)$ for all $i$ (scalars)
3. In the inner loop, use: $\nabla \psi_i(\tilde{w}) = \phi'_i(z_i) \cdot x_i + \lambda \tilde{w}$

This reduces inner-loop computation from $O(d)$ to $O(1)$ per sample (plus the $O(d)$ weight update).

## File Structure

```
experiment/
└── optimizers/
    └── svrg.py      ← THIS FILE
```

## Detailed Implementation

### 1. Imports

```python
import numpy as np
from models.logistic import sigmoid
```

### 2. SVRG for Binary Logistic Regression (with Scalar Optimization)

```python
def svrg_outer_loop_binary(w_tilde, X, y, lr, lam, m, option='I'):
    """One outer iteration of SVRG for binary logistic regression.
    
    Uses scalar storage optimization: precompute φ'(w̃^T x_i) for all i
    to avoid recomputing inner products in the inner loop.
    
    Args:
        w_tilde: snapshot weights (d,)
        X: feature matrix (n, d) — dense or sparse CSR
        y: label vector (n,) with values in {-1, +1}
        lr: step size η (constant)
        lam: L2 regularization strength
        m: number of inner loop iterations (typically 2n)
        option: 'I' (w̃ = w_m) or 'II' (random w_t from history)
    
    Returns:
        updated w_tilde for next outer iteration
    """
    n = len(y)
    
    # ── Step 1: Compute full gradient μ̃ ──
    # Precompute z_i = w_tilde^T x_i for all i
    z_tilde = X @ w_tilde                          # (n,)
    
    # φ'(z) for logistic loss: φ(z) = log(1 + exp(-y*z))
    # φ'(z) = -y * σ(-y*z) where σ is sigmoid
    phi_prime_tilde = -y * sigmoid(-y * z_tilde)   # (n,)
    
    # Full gradient: μ̃ = (1/n) Σ φ'_i(z_i) * x_i + λ * w_tilde
    mu = (X.T @ phi_prime_tilde) / n + lam * w_tilde  # (d,)
    
    # ── Step 2: Inner loop ──
    w = w_tilde.copy()
    
    if option == 'II':
        # Store history for random selection
        w_history = [w.copy()]
    
    for t in range(m):
        i = np.random.randint(n)
        xi = X[i]          # (d,) — dense 1D array
        yi = y[i]
        
        # ∇ψ_i(w) = -y_i * σ(-y_i * w^T x_i) * x_i + λ * w
        margin_w = yi * (xi @ w)
        g_current = (-yi * sigmoid(-margin_w)) * xi + lam * w
        
        # ∇ψ_i(w_tilde) — using precomputed scalar φ'_i(z_i)
        # = φ'_i(z_i) * x_i + λ * w_tilde
        g_snapshot = phi_prime_tilde[i] * xi + lam * w_tilde
        
        # SVRG update
        w = w - lr * (g_current - g_snapshot + mu)
        
        if option == 'II':
            w_history.append(w.copy())
    
    # ── Step 3: Update snapshot ──
    if option == 'I':
        return w
    else:  # Option II
        idx = np.random.randint(m + 1)
        return w_history[idx]
```

### 3. SVRG for Multi-class Logistic Regression

```python
def svrg_outer_loop_multiclass(W_tilde, X, y, lr, lam, m, option='I'):
    """One outer iteration of SVRG for multi-class logistic regression.
    
    For multi-class with K classes, W is (d, K).
    The scalar optimization still applies per class.
    
    Args:
        W_tilde: snapshot weight matrix (d, K)
        X: feature matrix (n, d)
        y: label vector (n,) with values in {0, ..., K-1}
        lr: step size η
        lam: L2 regularization
        m: inner loop length
        option: 'I' or 'II'
    
    Returns:
        updated W_tilde
    """
    n = len(y)
    K = W_tilde.shape[1]
    
    # ── Step 1: Full gradient ──
    logits_tilde = X @ W_tilde                      # (n, K)
    
    # Softmax probabilities
    logits_tilde -= np.max(logits_tilde, axis=1, keepdims=True)
    exp_logits = np.exp(logits_tilde)
    probs_tilde = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (n, K)
    
    # One-hot encoding
    one_hot = np.zeros((n, K))
    one_hot[np.arange(n), y] = 1.0
    
    # Gradient: (1/n) Σ (probs_i - e_{y_i}) ⊗ x_i + λW
    mu = (X.T @ (probs_tilde - one_hot)) / n + lam * W_tilde  # (d, K)
    
    # ── Step 2: Inner loop ──
    W = W_tilde.copy()
    
    if option == 'II':
        W_history = [W.copy()]
    
    for t in range(m):
        i = np.random.randint(n)
        xi = X[i]          # (d,)
        yi = y[i]
        
        # ∇ψ_i(W) for current w
        logits_i = xi @ W                              # (K,)
        logits_i -= np.max(logits_i)
        probs_i = np.exp(logits_i) / np.sum(np.exp(logits_i))  # (K,)
        
        e_yi = np.zeros(K)
        e_yi[yi] = 1.0
        
        g_current = np.outer(xi, probs_i - e_yi) + lam * W  # (d, K)
        
        # ∇ψ_i(W_tilde) — using precomputed probabilities
        g_snapshot = np.outer(xi, probs_tilde[i] - e_yi) + lam * W_tilde  # (d, K)
        
        # SVRG update
        W = W - lr * (g_current - g_snapshot + mu)
        
        if option == 'II':
            W_history.append(W.copy())
    
    if option == 'I':
        return W
    else:
        idx = np.random.randint(m + 1)
        return W_history[idx]
```

### 4. Unified Interface

```python
def svrg_outer_loop(w, X, y, lr, lam, m, multiclass=False, option='I'):
    """One outer iteration of SVRG.
    
    Args:
        w: snapshot weights (d,) for binary or (d, K) for multi-class
        X: feature matrix (n, d)
        y: label vector (n,)
        lr: step size
        lam: regularization
        m: inner loop length
        multiclass: multi-class flag
        option: 'I' or 'II'
    
    Returns:
        updated weights
    """
    if multiclass:
        return svrg_outer_loop_multiclass(w, X, y, lr, lam, m, option)
    return svrg_outer_loop_binary(w, X, y, lr, lam, m, option)


def effective_passes_svrg(n, m):
    """Compute effective passes for one SVRG outer iteration.
    
    Each outer iteration costs:
    - 1 pass for full gradient computation
    - m/n passes for inner loop
    
    Total = 1 + m/n
    
    Args:
        n: number of samples
        m: inner loop length
    
    Returns:
        effective passes for one outer iteration
    """
    return 1.0 + m / n
```

## Constraints & Configuration

### Inner Loop Length (m)

| Problem Type | m | Rationale |
|-------------|---|-----------|
| Convex (Logistic) | $2n$ | Paper default for convex |
| Non-convex (NN) | $5n$ | Paper default for neural networks |

### Learning Rates (η)

| Dataset | η (SVRG) | Notes |
|---------|----------|-------|
| MNIST | ~0.025 | From paper |
| CIFAR-10 | Tuned | Start with 0.01 |
| RCV1 | Tuned | Start with 0.01 |
| Covtype | Tuned | Start with 0.001 |

### Option Selection

| Option | Usage | Description |
|--------|-------|-------------|
| I | Default for experiments | w̃ = w_m (last iterate) |
| II | Theoretical analysis | Random pick from history |

## Effective Passes Calculation

For SVRG with $m = 2n$:
- Each outer iteration = $1 + 2n/n = 3$ effective passes
- After $s$ outer iterations: $3s$ effective passes (plus warm-start)

## Verification Checklist

- [ ] Loss decreases monotonically (on log scale) for all datasets
- [ ] SVRG converges faster than both SGD variants
- [ ] Loss residual forms a straight line on log scale (linear convergence)
- [ ] Option I and Option II produce similar results
- [ ] Scalar storage optimization works correctly (compare with naive version)
- [ ] Effective passes calculation is correct
- [ ] Works with both dense (MNIST, CIFAR-10) and sparse (RCV1, Covtype) data

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| **Wrong gradient sign in SVRG update** | Formula: $w - \eta(\nabla\psi_i(w) - \nabla\psi_i(\tilde{w}) + \tilde{\mu})$ — triple-check signs |
| **Forgetting λw in snapshot gradient** | $\nabla\psi_i(\tilde{w})$ must include $\lambda\tilde{w}$ |
| **Forgetting λw in current gradient** | $\nabla\psi_i(w)$ must include $\lambda w$ |
| **mu doesn't include λw_tilde** | Full gradient = data gradient + λw_tilde |
| **SVRG diverges** | Reduce learning rate η |
| **SVRG converges same as SGD** | Check that m is large enough (2n or 5n) |
| **Scalar optimization wrong for multi-class** | Multi-class needs full (d, K) gradient, can't use simple scalars |
| **Sparse X[i] returns 1D array** | This is correct for CSR matrices |
| **Option II picks from wrong range** | Random index from {0, ..., m} inclusive (m+1 items) |
