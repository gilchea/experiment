"""
train_nn.py — Neural Network Training Runner (Non-convex)

Orchestrates NN experiments for MNIST and CIFAR-10:
1. Loads dataset
2. Runs warm-start SGD
3. Runs SVRG (multiple outer iterations) with gradient variance tracking
4. Runs SGD baselines (constant + decaying)
5. Logs loss residuals, test errors, and effective passes
6. Saves results for plotting

Architecture: 1 hidden layer (100 nodes), Sigmoid, Softmax output (10 nodes)
"""

import os
import json
import numpy as np
from utils.data_loader import load_dataset
from models.neural_net import (
    loss, full_grad, compute_error, init_parameters, copy_params
)
from optimizers.svrg_nn import svrg_nn_outer_loop, effective_passes_svrg_nn
from optimizers.sgd_nn import (
    sgd_nn_epoch_constant, sgd_nn_epoch_decay, warm_start_nn
)
from config import NN_CONFIGS


# ---------------------------------------------------------------------------
# Main Experiment Runner
# ---------------------------------------------------------------------------

def run_nn_experiment(dataset_name, config, P_star=None, results_dir='results',
                      save_ckpt_every=5):
    """Run full NN experiment for one dataset.

    Args:
        dataset_name: name of dataset (e.g., 'mnist_nn')
        config: hyperparameter dict
        results_dir: output directory
        save_ckpt_every: save checkpoint every N epochs/outer iterations

    Returns:
        dict with all logged data
    """
    # Map to base dataset name
    base_name = dataset_name.replace('_nn', '')
    print(f"\n{'='*60}")
    print(f"Running NN experiment: {base_name}")
    print(f"{'='*60}")

    # ── Load data ──
    X_train, y_train, X_test, y_test = load_dataset(base_name)
    n, d = X_train.shape
    lam = config['lam']
    n_hidden = config['n_hidden']
    n_classes = config['n_classes']

    print(f"  Dataset: n={n}, d={d}")
    print(f"  Architecture: {d} -> {n_hidden} -> {n_classes}")

    # ── Initialize parameters ──
    params = init_parameters(d, n_hidden, n_classes, seed=42)

    # ── Warm-start ──
    print(f"\n  Warm-start: {config['warm_start_epochs']} epoch(s) SGD")
    params = warm_start_nn(
        X_train, y_train, lam, d, n_hidden, n_classes,
        n_epochs=config['warm_start_epochs'],
        lr=config['warm_start_lr'],
        seed=42,
    )

    # ── Logging setup ──
    results = {
        'dataset': dataset_name,
        'config': {k: v for k, v in config.items()},
        'P_star': float(P_star) if P_star is not None else None,
        'svrg': {'passes': [], 'loss_residual': [], 'test_error': [], 'grad_variance': []},
        'sgd_const': {'passes': [], 'loss_residual': [], 'test_error': []},
        'sgd_best': {'passes': [], 'loss_residual': [], 'test_error': []},
    }

    def _residual(loss_val):
        return float(loss_val - P_star) if P_star is not None else float(loss_val)

    # ── Run SVRG ──
    m = config['svrg_m_factor'] * n
    print(f"\n  Running SVRG (m={config['svrg_m_factor']}*n={m})...")
    w_svrg = copy_params(params)
    effective_pass = config['warm_start_epochs']

    # Log initial state
    train_loss = loss(w_svrg, X_train, y_train, lam)
    test_err = compute_error(w_svrg, X_test, y_test)
    results['svrg']['passes'].append(effective_pass)
    results['svrg']['loss_residual'].append(_residual(train_loss))
    results['svrg']['test_error'].append(float(test_err))
    results['svrg']['grad_variance'].append(None)

    for s in range(config['n_outer']):
        w_svrg, variance = svrg_nn_outer_loop(
            w_svrg, X_train, y_train,
            lr=config['svrg_lr'],
            lam=lam,
            m=m,
            option='I',
            track_variance=True,
        )

        effective_pass += effective_passes_svrg_nn(n, m, batch_size=config.get('batch_size', 10))
        train_loss = loss(w_svrg, X_train, y_train, lam)
        test_err = compute_error(w_svrg, X_test, y_test)

        results['svrg']['passes'].append(effective_pass)
        results['svrg']['loss_residual'].append(_residual(train_loss))
        results['svrg']['test_error'].append(float(test_err))
        results['svrg']['grad_variance'].append(float(variance))

        if (s + 1) % 5 == 0:
            print(f"    Outer iter {s+1:2d}: loss = {train_loss:.4f}, "
                  f"test error = {test_err:.2f}%, variance = {variance:.2e}")

    # ── Run SGD (Constant eta) ──
    print(f"\n  Running SGD (constant eta={config['sgd_const_lr']})...")
    w_sgd_const = copy_params(params)
    effective_pass = config['warm_start_epochs']

    train_loss = loss(w_sgd_const, X_train, y_train, lam)
    test_err = compute_error(w_sgd_const, X_test, y_test)
    results['sgd_const']['passes'].append(effective_pass)
    results['sgd_const']['loss_residual'].append(_residual(train_loss))
    results['sgd_const']['test_error'].append(float(test_err))

    for epoch in range(config['n_epochs_sgd']):
        w_sgd_const = sgd_nn_epoch_constant(
            w_sgd_const, X_train, y_train,
            lr=config['sgd_const_lr'],
            lam=lam,
        )

        effective_pass += 1.0
        train_loss = loss(w_sgd_const, X_train, y_train, lam)
        test_err = compute_error(w_sgd_const, X_test, y_test)

        results['sgd_const']['passes'].append(effective_pass)
        results['sgd_const']['loss_residual'].append(_residual(train_loss))
        results['sgd_const']['test_error'].append(float(test_err))

        if (epoch + 1) % 10 == 0:
            print(f"    SGD-const epoch {epoch+1:3d}: loss = {train_loss:.4f}, "
                  f"test error = {test_err:.2f}%")

    # ── Run SGD (Best / Decaying eta) ──
    print(f"\n  Running SGD-best (eta_0={config['sgd_best_lr0']}, "
          f"b={config['sgd_best_b']})...")
    w_sgd_best = copy_params(params)
    effective_pass = config['warm_start_epochs']
    t = 0

    train_loss = loss(w_sgd_best, X_train, y_train, lam)
    test_err = compute_error(w_sgd_best, X_test, y_test)
    results['sgd_best']['passes'].append(effective_pass)
    results['sgd_best']['loss_residual'].append(_residual(train_loss))
    results['sgd_best']['test_error'].append(float(test_err))

    for epoch in range(config['n_epochs_sgd']):
        w_sgd_best, t = sgd_nn_epoch_decay(
            w_sgd_best, X_train, y_train,
            lr0=config['sgd_best_lr0'],
            lam=lam,
            n=n,
            t_start=t,
            b=config['sgd_best_b'],
        )

        effective_pass += 1.0
        train_loss = loss(w_sgd_best, X_train, y_train, lam)
        test_err = compute_error(w_sgd_best, X_test, y_test)

        results['sgd_best']['passes'].append(effective_pass)
        results['sgd_best']['loss_residual'].append(_residual(train_loss))
        results['sgd_best']['test_error'].append(float(test_err))

        if (epoch + 1) % 10 == 0:
            print(f"    SGD-best epoch {epoch+1:3d}: loss = {train_loss:.4f}, "
                  f"test error = {test_err:.2f}%")

    # ── Save results ──
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f'{dataset_name}_results.json')
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if x is not None else None)
    print(f"\n  [V] Results saved to {filepath}")

    return results


def run_all_nn_experiments(optimal_path='results/optimal_loss.json'):
    """Run NN experiments for all configured datasets."""
    import json, os
    optimal_losses = {}
    if os.path.exists(optimal_path):
        with open(optimal_path) as f:
            optimal_losses = json.load(f)
    else:
        print(f"  [!] {optimal_path} not found - logging raw loss instead of residual.")

    all_results = {}
    for dataset_name, config in NN_CONFIGS.items():
        base_name = dataset_name.replace('_nn', '')
        P_star = optimal_losses.get(base_name, {}).get('P_star', None)
        results = run_nn_experiment(dataset_name, config, P_star=P_star)
        all_results[dataset_name] = results
    return all_results


if __name__ == '__main__':
    run_all_nn_experiments()
