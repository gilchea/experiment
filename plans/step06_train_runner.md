# Step 06: `train.py` — Main Training Runner

## Objective

Create the central orchestrator that runs all experiments:
1. Loads each dataset
2. Runs warm-start SGD
3. Runs SVRG (multiple outer iterations)
4. Runs SGD baselines (constant + decaying)
5. Logs loss residuals and effective passes
6. Saves results for plotting

## Experiment Matrix

| Dataset | Model Type | λ | m | Warm-start | η (SVRG) | η (SGD-const) | η₀ (SGD-best) |
|---------|-----------|----|----|-----------|----------|---------------|---------------|
| MNIST | Multi-class | 1e-4 | 2n | 1 epoch | 0.025 | 0.01 | 0.1 |
| CIFAR-10 | Multi-class | 1e-3 | 2n | 1 epoch | 0.01 | 0.01 | 0.1 |
| RCV1 | Binary | 1e-5 | 2n | 1 epoch | 0.01 | 0.001 | 0.01 |
| Covtype | Binary | 1e-5 | 2n | 1 epoch | 0.001 | 0.001 | 0.01 |

## File Structure

```
experiment/
├── train.py               ← THIS FILE
├── config.py              ← All hyperparameters
├── results/               ← Output directory (created automatically)
│   └── optimal_loss.json  ← From Step 03
└── figures/               ← Plot output (created automatically)
```

## Detailed Implementation

### 1. `config.py` — Centralized Hyperparameters

```python
# config.py
import numpy as np

DATASET_CONFIGS = {
    'mnist': {
        'lam': 1e-4,
        'multiclass': True,
        'svrg_lr': 0.025,
        'svrg_m_factor': 2,       # m = 2 * n
        'sgd_const_lr': 0.01,
        'sgd_best_lr0': 0.1,
        'sgd_best_a': 1.0,        # Tune this
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        'n_outer': 30,            # SVRG outer iterations
        'n_epochs_sgd': 90,       # SGD epochs (match effective passes)
    },
    'cifar10': {
        'lam': 1e-3,
        'multiclass': True,
        'svrg_lr': 0.01,
        'svrg_m_factor': 2,
        'sgd_const_lr': 0.01,
        'sgd_best_lr0': 0.1,
        'sgd_best_a': 1.0,
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        'n_outer': 30,
        'n_epochs_sgd': 90,
    },
    'rcv1': {
        'lam': 1e-5,
        'multiclass': False,
        'svrg_lr': 0.01,
        'svrg_m_factor': 2,
        'sgd_const_lr': 0.001,
        'sgd_best_lr0': 0.01,
        'sgd_best_a': 1.0,
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        'n_outer': 30,
        'n_epochs_sgd': 90,
    },
    'covtype': {
        'lam': 1e-5,
        'multiclass': False,
        'svrg_lr': 0.001,
        'svrg_m_factor': 2,
        'sgd_const_lr': 0.001,
        'sgd_best_lr0': 0.01,
        'sgd_best_a': 1.0,
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        'n_outer': 30,
        'n_epochs_sgd': 90,
    },
}
```

### 2. `train.py` — Main Runner

```python
import os
import json
import time
import numpy as np
from utils.data_loader import load_dataset
from models.logistic import loss
from optimizers.sgd import sgd_epoch_constant, sgd_epoch_decay, warm_start
from optimizers.svrg import svrg_outer_loop, effective_passes_svrg
from config import DATASET_CONFIGS


def run_experiment(dataset_name, config, P_star, results_dir='results'):
    """Run full experiment for one dataset.
    
    Args:
        dataset_name: name of dataset
        config: hyperparameter dict
        P_star: optimal loss value
        results_dir: output directory
    
    Returns:
        dict with all logged data
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {dataset_name}")
    print(f"{'='*60}")
    
    # ── Load data ──
    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    n, d = X_train.shape
    multiclass = config['multiclass']
    lam = config['lam']
    
    print(f"  Dataset: n={n}, d={d}, multiclass={multiclass}")
    
    # ── Initialize weights ──
    if multiclass:
        K = len(np.unique(y_train))
        w0 = np.zeros((d, K))
    else:
        w0 = np.zeros(d)
    
    # ── Warm-start ──
    print(f"\n  Warm-start: {config['warm_start_epochs']} epoch(s) SGD")
    w = warm_start(X_train, y_train, lam, multiclass,
                   n_epochs=config['warm_start_epochs'],
                   lr=config['warm_start_lr'])
    
    # ── Logging setup ──
    results = {
        'dataset': dataset_name,
        'config': config,
        'P_star': P_star,
        'svrg': {'passes': [], 'loss_residual': [], 'test_error': []},
        'sgd_const': {'passes': [], 'loss_residual': [], 'test_error': []},
        'sgd_best': {'passes': [], 'loss_residual': [], 'test_error': []},
    }
    
    # ── Run SVRG ──
    print(f"\n  Running SVRG (m={config['svrg_m_factor']}*n={config['svrg_m_factor']*n})...")
    w_svrg = w.copy()
    effective_pass = config['warm_start_epochs']  # Start after warm-start
    
    # Log initial state
    train_loss = loss(w_svrg, X_train, y_train, lam, multiclass)
    results['svrg']['passes'].append(effective_pass)
    results['svrg']['loss_residual'].append(float(train_loss - P_star))
    
    for s in range(config['n_outer']):
        m = config['svrg_m_factor'] * n
        w_svrg = svrg_outer_loop(
            w_svrg, X_train, y_train,
            lr=config['svrg_lr'],
            lam=lam,
            m=m,
            multiclass=multiclass,
            option='I'
        )
        
        effective_pass += effective_passes_svrg(n, m)
        train_loss = loss(w_svrg, X_train, y_train, lam, multiclass)
        
        results['svrg']['passes'].append(effective_pass)
        results['svrg']['loss_residual'].append(float(train_loss - P_star))
        
        if (s + 1) % 5 == 0:
            print(f"    Outer iter {s+1:2d}: loss residual = {train_loss - P_star:.2e}")
    
    # ── Run SGD (Constant η) ──
    print(f"\n  Running SGD (constant η={config['sgd_const_lr']})...")
    w_sgd_const = w.copy()
    effective_pass = config['warm_start_epochs']
    
    train_loss = loss(w_sgd_const, X_train, y_train, lam, multiclass)
    results['sgd_const']['passes'].append(effective_pass)
    results['sgd_const']['loss_residual'].append(float(train_loss - P_star))
    
    for epoch in range(config['n_epochs_sgd']):
        w_sgd_const = sgd_epoch_constant(
            w_sgd_const, X_train, y_train,
            lr=config['sgd_const_lr'],
            lam=lam,
            multiclass=multiclass
        )
        
        effective_pass += 1.0
        train_loss = loss(w_sgd_const, X_train, y_train, lam, multiclass)
        
        results['sgd_const']['passes'].append(effective_pass)
        results['sgd_const']['loss_residual'].append(float(train_loss - P_star))
    
    # ── Run SGD (Best / Decaying η) ──
    print(f"\n  Running SGD-best (η₀={config['sgd_best_lr0']}, a={config['sgd_best_a']})...")
    w_sgd_best = w.copy()
    effective_pass = config['warm_start_epochs']
    t = 0  # Total gradient evaluations
    
    train_loss = loss(w_sgd_best, X_train, y_train, lam, multiclass)
    results['sgd_best']['passes'].append(effective_pass)
    results['sgd_best']['loss_residual'].append(float(train_loss - P_star))
    
    for epoch in range(config['n_epochs_sgd']):
        w_sgd_best, t = sgd_epoch_decay(
            w_sgd_best, X_train, y_train,
            lr0=config['sgd_best_lr0'],
            lam=lam,
            t_start=t,
            a=config['sgd_best_a'],
            multiclass=multiclass
        )
        
        effective_pass += 1.0
        train_loss = loss(w_sgd_best, X_train, y_train, lam, multiclass)
        
        results['sgd_best']['passes'].append(effective_pass)
        results['sgd_best']['loss_residual'].append(float(train_loss - P_star))
    
    # ── Save results ──
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f'{dataset_name}_results.json')
    with open(filepath, 'w') as f:
        # Convert numpy values to Python floats
        json.dump(results, f, indent=2, default=lambda x: float(x))
    print(f"\n  ✓ Results saved to {filepath}")
    
    return results


def run_all_experiments():
    """Run experiments for all datasets."""
    # Load optimal losses
    optimal_path = 'results/optimal_loss.json'
    if not os.path.exists(optimal_path):
        print(f"ERROR: {optimal_path} not found. Run compute_optimal.py first.")
        return
    
    with open(optimal_path, 'r') as f:
        optimal_losses = json.load(f)
    
    all_results = {}
    for dataset_name, config in DATASET_CONFIGS.items():
        P_star = optimal_losses[dataset_name]['P_star']
        results = run_experiment(dataset_name, config, P_star)
        all_results[dataset_name] = results
    
    return all_results


if __name__ == '__main__':
    run_all_experiments()
```

## Execution Order

```
1. python compute_optimal.py     → results/optimal_loss.json
2. python train.py               → results/{dataset}_results.json
3. python plot_results.py        → figures/figure*.png
```

## Constraints & Configuration

### Effective Passes Alignment

For fair comparison, all methods should run for the **same number of effective passes**:

| Method | Passes per iteration | Total passes for 30 SVRG outer iters |
|--------|--------------------|--------------------------------------|
| SVRG (m=2n) | 3 passes/outer | 90 passes |
| SGD-constant | 1 pass/epoch | 90 epochs |
| SGD-best | 1 pass/epoch | 90 epochs |

### Logging Frequency

| Method | Logging point |
|--------|--------------|
| SVRG | After each outer iteration (every 3 effective passes) |
| SGD | After each epoch (every 1 effective pass) |

This means SVRG will have fewer data points but each point represents more computation.

## Verification Checklist

- [ ] All 4 datasets run without errors
- [ ] SVRG loss residual decreases faster than both SGD variants
- [ ] Effective passes are consistent across methods
- [ ] Results saved as JSON files in `results/` directory
- [ ] Warm-start correctly applied before all methods
- [ ] Multi-class (MNIST, CIFAR-10) and binary (RCV1, Covtype) both work
- [ ] Loss residual values are positive (or very close to zero)

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| SVRG takes too long on Covtype (581k × 2n = 1.16M inner steps) | This is expected; be patient or reduce n_outer |
| Loss residual becomes negative | P_star estimate is too high; run GD longer in compute_optimal.py |
| SGD-best doesn't outperform SGD-constant | Tune `a` parameter; try values from 0.1 to 10 |
| Out of memory for large datasets | RCV1 and Covtype are sparse; ensure sparse operations |
| JSON serialization fails with numpy types | Use `default=lambda x: float(x)` in json.dump |
