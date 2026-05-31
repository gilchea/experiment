# Step 04: `optimizers/sgd.py` — SGD Baselines (Constant & Decaying)

## Objective

Implement two SGD variants used as baselines in the paper:
1. **SGD (Constant η)**: Fixed learning rate throughout training
2. **SGD (Best / Decaying)**: Learning rate schedule $\eta_t = \eta_0 (1 + a \eta_0 t)^{-1}$

## Why Two SGD Variants?

The paper compares SVRG against:
- **SGD (constant)**: Shows the "variance floor" — loss plateaus because variance never goes to zero
- **SGD-best**: SGD with a carefully tuned decaying schedule — represents the best possible SGD performance

## File Structure

```
experiment/
└── optimizers/
    └── sgd.py      ← THIS FILE
```

## Detailed Implementation

### 1. Imports & Helper

```python
import numpy as np
from models.logistic import stoch_grad_binary, stoch_grad_multiclass

def _get_stoch_grad_fn(multiclass):
    """Return appropriate stochastic gradient function."""
    return stoch_grad_multiclass if multiclass else stoch_grad_binary
```

### 2. SGD with Constant Learning Rate

```python
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
```

### 3. SGD with Decaying Learning Rate (SGD-best)

The paper uses the schedule:
$$\eta_t = \frac{\eta_0}{1 + a \eta_0 t}$$

Where:
- $\eta_0$: initial learning rate
- $a$: decay factor (tuned per dataset)
- $t$: total number of gradient evaluations so far

```python
def sgd_epoch_decay(w, X, y, lr0, lam, t_start, a, multiclass=False, batch_size=1):
    """Run 1 epoch of SGD with decaying learning rate.
    
    Learning rate schedule: η(t) = η₀ / (1 + a * η₀ * t)
    where t = total gradient evaluations so far.
    
    Args:
        w: weight vector/matrix
        X: feature matrix (n, d)
        y: label vector (n,)
        lr0: initial learning rate η₀
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
```

### 4. Warm-start Helper

The paper requires warm-start before SVRG:
- **Convex (Logistic Regression)**: 1 epoch SGD
- **Non-convex (Neural Network)**: 10 epochs SGD

```python
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
```

## Constraints & Configuration

### Learning Rates from Paper

| Dataset | SGD-constant η | SGD-best η₀ | SGD-best a |
|---------|---------------|-------------|------------|
| MNIST | ~0.01 | ~0.1 | Tuned |
| CIFAR-10 | ~0.01 | ~0.1 | Tuned |
| RCV1 | ~0.001 | ~0.01 | Tuned |
| Covtype | ~0.001 | ~0.01 | Tuned |

**Note**: The paper says "SGD-best was carefully tuned." The exact values of `a` need to be found by experimentation. Start with `a=1.0` and adjust.

### Warm-start Configuration

| Problem Type | n_epochs | lr |
|-------------|----------|-----|
| Convex (Logistic) | 1 | 0.01 |
| Non-convex (NN) | 10 | 0.01 |

## Effective Passes Counting

For fair comparison, track effective passes:

```python
def count_effective_passes_sgd(n_epochs, n_samples):
    """For SGD, 1 epoch = 1 effective pass."""
    return n_epochs
```

## Verification Checklist

- [ ] `sgd_epoch_constant()`: loss decreases after 1 epoch
- [ ] `sgd_epoch_decay()`: learning rate decreases over time
- [ ] SGD-constant loss plateaus (doesn't reach very low values)
- [ ] SGD-decay loss goes lower than SGD-constant
- [ ] Warm-start produces reasonable initial weights
- [ ] Both variants handle sparse X (RCV1, Covtype) correctly
- [ ] Both variants handle multi-class (MNIST, CIFAR-10) correctly

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| SGD-constant loss oscillates | Reduce learning rate |
| SGD-decay learning rate drops too fast | Reduce `a` parameter |
| SGD-decay learning rate drops too slow | Increase `a` parameter |
| Warm-start with wrong n_epochs | Use 1 for convex, 10 for NN |
| Forgetting to shuffle indices | Use `np.random.permutation(n)` each epoch |
| Sparse X indexing returns 1D vector | `X[i]` on CSR returns dense 1D array — this is fine |
