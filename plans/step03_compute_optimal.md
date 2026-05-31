# Step 03: `compute_optimal.py` — Estimate $P(w^*)$ via Full Gradient Descent

## Objective

Compute a high-precision estimate of the optimal loss $P(w^*)$ for each (dataset, λ) combination. This value is used as the baseline for computing **loss residual** $P(w) - P(w^*)$ in all plots.

## Why This Is Critical

- The Y-axis of all figures in the paper is **loss residual** $P(w) - P(w^*)$ on a log scale
- If $P(w^*)$ is inaccurate, the residuals will be wrong (negative values break log scale)
- Must run GD long enough to reach very low loss (typically $< 10^{-12}$ residual)

## File Structure

```
experiment/
└── compute_optimal.py      ← THIS FILE
```

## Detailed Implementation

### 1. Imports

```python
import os
import json
import numpy as np
from utils.data_loader import load_dataset
from models.logistic import loss_binary, full_grad_binary, loss_multiclass, full_grad_multiclass
```

### 2. GD Solver

```python
def gd_solve(X, y, lam, multiclass=False, max_iter=2000, lr=0.1, tol=1e-14, verbose=True):
    """Run full-batch Gradient Descent to find P(w*).
    
    Args:
        X: feature matrix (n, d)
        y: label vector (n,)
        lam: L2 regularization strength
        multiclass: whether to use multi-class logistic regression
        max_iter: maximum GD iterations
        lr: learning rate (constant)
        tol: stop when relative loss change < tol
        verbose: print progress
    
    Returns:
        w_star: optimal weights
        P_star: optimal loss value
        loss_history: list of loss values per iteration
    """
    n, d = X.shape
    K = len(np.unique(y)) if multiclass else 1
    
    if multiclass:
        w = np.zeros((d, K))
        loss_fn = loss_multiclass
        grad_fn = full_grad_multiclass
    else:
        w = np.zeros(d)
        loss_fn = loss_binary
        grad_fn = full_grad_binary
    
    loss_history = []
    
    for t in range(max_iter):
        g = grad_fn(w, X, y, lam)
        w = w - lr * g
        
        if t % 100 == 0 or t == max_iter - 1:
            current_loss = loss_fn(w, X, y, lam)
            loss_history.append(current_loss)
            
            if verbose:
                print(f"  iter {t:4d}: loss = {current_loss:.12f}")
            
            # Check convergence
            if len(loss_history) >= 2:
                rel_change = abs(loss_history[-1] - loss_history[-2]) / max(1, abs(loss_history[-2]))
                if rel_change < tol:
                    print(f"  Converged at iter {t}")
                    break
    
    P_star = loss_fn(w, X, y, lam)
    return w, P_star, loss_history
```

### 3. Main Runner

```python
def compute_all_optimal():
    """Compute P(w*) for all 4 datasets with their respective λ values."""
    
    configs = [
        # (dataset_name, lam, multiclass)
        ('mnist',  1e-4, True),
        ('cifar10', 1e-3, True),
        ('rcv1',   1e-5, False),
        ('covtype', 1e-5, False),
    ]
    
    results = {}
    
    for name, lam, multiclass in configs:
        print(f"\n{'='*60}")
        print(f"Computing P(w*) for {name} (λ={lam}, multiclass={multiclass})")
        print(f"{'='*60}")
        
        X_train, y_train, _, _ = load_dataset(name)
        n, d = X_train.shape
        print(f"  Dataset: n={n}, d={d}")
        
        # Adjust learning rate based on dataset size
        # GD needs smaller lr for large datasets
        if name == 'covtype':
            lr = 0.01  # Large n, need smaller step
        elif name == 'rcv1':
            lr = 0.05
        else:
            lr = 0.1
        
        w_star, P_star, loss_hist = gd_solve(
            X_train, y_train, lam, 
            multiclass=multiclass,
            max_iter=2000,
            lr=lr,
            verbose=True
        )
        
        results[name] = {
            'P_star': float(P_star),
            'lam': lam,
            'multiclass': multiclass,
            'n': n,
            'd': d,
            'loss_history': [float(v) for v in loss_hist],
            'final_loss': float(loss_hist[-1]),
        }
        
        print(f"  ✓ P(w*) = {P_star:.12f}")
    
    return results
```

### 4. Save & Load Utilities

```python
def save_optimal(results, filepath='results/optimal_loss.json'):
    """Save P(w*) results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved optimal losses to {filepath}")

def load_optimal(filepath='results/optimal_loss.json'):
    """Load P(w*) results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    results = compute_all_optimal()
    save_optimal(results)
```

## Constraints & Configuration

| Dataset | λ | multiclass | GD lr | Max iter | Expected P(w*) |
|---------|---|-----------|-------|----------|-----------------|
| MNIST | 1e-4 | True | 0.1 | 2000 | ~0.3-0.5 (depends on init) |
| CIFAR-10 | 1e-3 | True | 0.1 | 2000 | ~1.5-2.0 |
| RCV1 | 1e-5 | False | 0.05 | 2000 | ~0.1-0.3 |
| Covtype | 1e-5 | False | 0.01 | 2000 | ~0.3-0.5 |

**Note**: The exact $P(w^*)$ values depend on the data split. What matters is that the loss **converges** (flattens out). The residual $P(w) - P(w^*)$ will then be computed relative to this converged value.

## Verification Checklist

- [ ] Loss decreases monotonically (or nearly so) for all datasets
- [ ] Loss flattens out (converges) within 2000 iterations
- [ ] Final loss is consistent across runs (deterministic GD)
- [ ] JSON file saved correctly to `results/optimal_loss.json`
- [ ] `load_optimal()` returns the same values
- [ ] For multi-class, weight shape is (d, K); for binary, (d,)

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| GD diverges (loss increases) | Reduce learning rate `lr` |
| GD converges too slowly | Increase `max_iter` or tune `lr` |
| Loss goes negative | This is fine — cross-entropy can be < 0 |
| Covtype takes too long (581k samples) | Full GD on 581k × 54 is fast; if slow, reduce max_iter |
| RCV1 is sparse but GD works fine | `X @ w` with sparse X returns dense vector |
| Multi-class GD on CIFAR-10 (50k × 3072 × 10) | May take a few minutes; be patient |
