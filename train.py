"""
train.py — Main Training Runner

Orchestrates all experiments:
1. Loads each dataset
2. Runs warm-start SGD
3. Runs SVRG (multiple outer iterations)
4. Runs SGD baselines (constant + decaying)
5. Logs loss residuals and effective passes
6. Saves results for plotting
"""

import os
import json
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
        'config': {k: v for k, v in config.items()},
        'P_star': P_star,
        'svrg': {'passes': [], 'loss_residual': [], 'test_error': []},
        'sgd_const': {'passes': [], 'loss_residual': [], 'test_error': []},
        'sgd_best': {'passes': [], 'loss_residual': [], 'test_error': []},
    }

    # ── Run SVRG ──
    m = config['svrg_m_factor'] * n
    print(f"\n  Running SVRG (m={config['svrg_m_factor']}*n={m})...")
    w_svrg = w.copy()
    effective_pass = config['warm_start_epochs']  # Start after warm-start

    # Log initial state
    train_loss = loss(w_svrg, X_train, y_train, lam, multiclass)
    results['svrg']['passes'].append(effective_pass)
    results['svrg']['loss_residual'].append(float(train_loss - P_star))

    for s in range(config['n_outer']):
        w_svrg = svrg_outer_loop(
            w_svrg, X_train, y_train,
            lr=config['svrg_lr'],
            lam=lam,
            m=m,
            multiclass=multiclass,
            option='I',
        )

        effective_pass += effective_passes_svrg(n, m)
        train_loss = loss(w_svrg, X_train, y_train, lam, multiclass)

        results['svrg']['passes'].append(effective_pass)
        results['svrg']['loss_residual'].append(float(train_loss - P_star))

        if (s + 1) % 5 == 0:
            print(f"    Outer iter {s+1:2d}: loss residual = {train_loss - P_star:.2e}")

    # ── Run SGD (Constant eta) ──
    print(f"\n  Running SGD (constant eta={config['sgd_const_lr']})...")
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
            multiclass=multiclass,
        )

        effective_pass += 1.0
        train_loss = loss(w_sgd_const, X_train, y_train, lam, multiclass)

        results['sgd_const']['passes'].append(effective_pass)
        results['sgd_const']['loss_residual'].append(float(train_loss - P_star))

    # ── Run SGD (Best / Decaying eta) ──
    print(f"\n  Running SGD-best (eta_0={config['sgd_best_lr0']}, a={config['sgd_best_a']})...")
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
            multiclass=multiclass,
        )

        effective_pass += 1.0
        train_loss = loss(w_sgd_best, X_train, y_train, lam, multiclass)

        results['sgd_best']['passes'].append(effective_pass)
        results['sgd_best']['loss_residual'].append(float(train_loss - P_star))

    # ── Save results ──
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f'{dataset_name}_results.json')
    with open(filepath, 'w') as f:
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
