"""
train.py — Main Training Runner

Orchestrates all experiments:
1. Loads each dataset
2. Runs warm-start SGD
3. Runs SVRG (multiple outer iterations) with gradient variance tracking
4. Runs SGD baselines (constant + decaying)
5. Runs SDCA baseline
6. Runs SAG baseline
7. Logs loss residuals, test errors, and effective passes
8. Saves checkpoints periodically
9. Saves results for plotting
"""

import os
import json
import pickle
import numpy as np
from utils.data_loader import load_dataset
from models.logistic import loss
from optimizers.sgd import sgd_epoch_constant, sgd_epoch_decay, warm_start
from optimizers.svrg import svrg_outer_loop, effective_passes_svrg
from optimizers.sdca import sdca_train
from optimizers.sag import sag_train
from config import DATASET_CONFIGS


# ---------------------------------------------------------------------------
# Test Error Computation
# ---------------------------------------------------------------------------

def compute_test_error(w, X_test, y_test, multiclass=False):
    """Compute test error rate (percentage).

    Args:
        w: weight vector (d,) for binary or (d, K) for multi-class
        X_test: test features (n_test, d)
        y_test: test labels (n_test,)
        multiclass: multi-class flag

    Returns:
        error rate as percentage
    """
    if multiclass:
        logits = X_test @ w
        preds = np.argmax(logits, axis=1)
    else:
        scores = X_test @ w
        preds = np.sign(scores)

    errors = np.mean(preds != y_test)
    return errors * 100  # Convert to percentage


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = 'checkpoints'


def save_checkpoint(algorithm, dataset_name, w, passes, loss_val, test_err,
                    variance, extra_state=None, epoch=0):
    """Save a training checkpoint.

    Args:
        algorithm: algorithm name (e.g., 'svrg', 'sgd_const')
        dataset_name: dataset name
        w: current weights
        passes: effective passes so far
        loss_val: current loss value
        test_err: current test error
        variance: current gradient variance (or None)
        extra_state: additional algorithm-specific state
        epoch: current epoch/iteration number
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filename = f'{algorithm}_{dataset_name}_epoch{epoch}.pkl'
    filepath = os.path.join(CHECKPOINT_DIR, filename)

    checkpoint = {
        'algorithm': algorithm,
        'dataset': dataset_name,
        'epoch': epoch,
        'passes': passes,
        'loss': loss_val,
        'test_error': test_err,
        'variance': variance,
        'weights': w,
        'extra_state': extra_state,
    }

    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(algorithm, dataset_name, epoch):
    """Load a training checkpoint.

    Args:
        algorithm: algorithm name
        dataset_name: dataset name
        epoch: epoch number

    Returns:
        checkpoint dict or None if not found
    """
    filename = f'{algorithm}_{dataset_name}_epoch{epoch}.pkl'
    filepath = os.path.join(CHECKPOINT_DIR, filename)

    if not os.path.exists(filepath):
        return None

    with open(filepath, 'rb') as f:
        return pickle.load(f)


def clean_checkpoints(algorithm, dataset_name, keep_last=1):
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        algorithm: algorithm name
        dataset_name: dataset name
        keep_last: number of most recent checkpoints to keep
    """
    if not os.path.exists(CHECKPOINT_DIR):
        return

    prefix = f'{algorithm}_{dataset_name}_epoch'
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR)
                   if f.startswith(prefix) and f.endswith('.pkl')]

    # Sort by epoch number
    def extract_epoch(filename):
        try:
            return int(filename.replace(prefix, '').replace('.pkl', ''))
        except ValueError:
            return -1

    checkpoints.sort(key=extract_epoch)

    # Remove old ones
    for f in checkpoints[:-keep_last]:
        os.remove(os.path.join(CHECKPOINT_DIR, f))


# ---------------------------------------------------------------------------
# Main Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(dataset_name, config, P_star, results_dir='results',
                   save_ckpt_every=5):
    """Run full experiment for one dataset.

    Args:
        dataset_name: name of dataset
        config: hyperparameter dict
        P_star: optimal loss value
        results_dir: output directory
        save_ckpt_every: save checkpoint every N epochs/outer iterations

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
        'svrg': {'passes': [], 'loss_residual': [], 'test_error': [],
                 'grad_variance': []},
        'sgd_const': {'passes': [], 'loss_residual': [], 'test_error': []},
        'sgd_best': {'passes': [], 'loss_residual': [], 'test_error': []},
        'sdca': {'passes': [], 'loss_residual': [], 'test_error': []},
        'sag': {'passes': [], 'loss_residual': [], 'test_error': []},
    }

    # ── Run SVRG ──
    m = config['svrg_m_factor'] * n
    print(f"\n  Running SVRG (m={config['svrg_m_factor']}*n={m})...")
    w_svrg = w.copy()
    effective_pass = config['warm_start_epochs']  # Start after warm-start

    # Log initial state
    train_loss = loss(w_svrg, X_train, y_train, lam, multiclass)
    test_err = compute_test_error(w_svrg, X_test, y_test, multiclass)
    results['svrg']['passes'].append(effective_pass)
    results['svrg']['loss_residual'].append(float(train_loss - P_star))
    results['svrg']['test_error'].append(float(test_err))
    results['svrg']['grad_variance'].append(None)

    for s in range(config['n_outer']):
        w_svrg, variance = svrg_outer_loop(
            w_svrg, X_train, y_train,
            lr=config['svrg_lr'],
            lam=lam,
            m=m,
            multiclass=multiclass,
            option='I',
            track_variance=True,
        )

        effective_pass += effective_passes_svrg(n, m)
        train_loss = loss(w_svrg, X_train, y_train, lam, multiclass)
        test_err = compute_test_error(w_svrg, X_test, y_test, multiclass)

        results['svrg']['passes'].append(effective_pass)
        results['svrg']['loss_residual'].append(float(train_loss - P_star))
        results['svrg']['test_error'].append(float(test_err))
        results['svrg']['grad_variance'].append(float(variance))

        # Save checkpoint
        if (s + 1) % save_ckpt_every == 0:
            save_checkpoint('svrg', dataset_name, w_svrg, effective_pass,
                            train_loss, test_err, variance, epoch=s + 1)
            clean_checkpoints('svrg', dataset_name, keep_last=2)

        if (s + 1) % 5 == 0:
            print(f"    Outer iter {s+1:2d}: loss residual = {train_loss - P_star:.2e}, "
                  f"test error = {test_err:.2f}%, variance = {variance:.2e}")

    # ── Run SGD (Constant eta) ──
    print(f"\n  Running SGD (constant eta={config['sgd_const_lr']})...")
    w_sgd_const = w.copy()
    effective_pass = config['warm_start_epochs']

    train_loss = loss(w_sgd_const, X_train, y_train, lam, multiclass)
    test_err = compute_test_error(w_sgd_const, X_test, y_test, multiclass)
    results['sgd_const']['passes'].append(effective_pass)
    results['sgd_const']['loss_residual'].append(float(train_loss - P_star))
    results['sgd_const']['test_error'].append(float(test_err))

    for epoch in range(config['n_epochs_sgd']):
        w_sgd_const = sgd_epoch_constant(
            w_sgd_const, X_train, y_train,
            lr=config['sgd_const_lr'],
            lam=lam,
            multiclass=multiclass,
        )

        effective_pass += 1.0
        train_loss = loss(w_sgd_const, X_train, y_train, lam, multiclass)
        test_err = compute_test_error(w_sgd_const, X_test, y_test, multiclass)

        results['sgd_const']['passes'].append(effective_pass)
        results['sgd_const']['loss_residual'].append(float(train_loss - P_star))
        results['sgd_const']['test_error'].append(float(test_err))

        # Save checkpoint
        if (epoch + 1) % save_ckpt_every == 0:
            save_checkpoint('sgd_const', dataset_name, w_sgd_const,
                            effective_pass, train_loss, test_err, None,
                            epoch=epoch + 1)
            clean_checkpoints('sgd_const', dataset_name, keep_last=2)

    # ── Run SGD (Best / Decaying eta) ──
    print(f"\n  Running SGD-best (eta_0={config['sgd_best_lr0']}, a={config['sgd_best_a']})...")
    w_sgd_best = w.copy()
    effective_pass = config['warm_start_epochs']
    t = 0  # Total gradient evaluations

    train_loss = loss(w_sgd_best, X_train, y_train, lam, multiclass)
    test_err = compute_test_error(w_sgd_best, X_test, y_test, multiclass)
    results['sgd_best']['passes'].append(effective_pass)
    results['sgd_best']['loss_residual'].append(float(train_loss - P_star))
    results['sgd_best']['test_error'].append(float(test_err))

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
        test_err = compute_test_error(w_sgd_best, X_test, y_test, multiclass)

        results['sgd_best']['passes'].append(effective_pass)
        results['sgd_best']['loss_residual'].append(float(train_loss - P_star))
        results['sgd_best']['test_error'].append(float(test_err))

        if (epoch + 1) % save_ckpt_every == 0:
            save_checkpoint('sgd_best', dataset_name, w_sgd_best,
                            effective_pass, train_loss, test_err, None,
                            epoch=epoch + 1)
            clean_checkpoints('sgd_best', dataset_name, keep_last=2)

    # ── Run SDCA ──
    print(f"\n  Running SDCA...")
    w_sdca = w.copy()
    effective_pass = config['warm_start_epochs']

    train_loss = loss(w_sdca, X_train, y_train, lam, multiclass)
    test_err = compute_test_error(w_sdca, X_test, y_test, multiclass)
    results['sdca']['passes'].append(effective_pass)
    results['sdca']['loss_residual'].append(float(train_loss - P_star))
    results['sdca']['test_error'].append(float(test_err))

    # SDCA uses callbacks to log per-epoch progress
    n_epochs_sdca = config['n_epochs_sdca']
    log_interval = max(1, n_epochs_sdca // 30)

    def sdca_callback(w_curr, epoch):
        nonlocal effective_pass
        effective_pass = config['warm_start_epochs'] + epoch + 1
        loss_curr = loss(w_curr, X_train, y_train, lam, multiclass)
        err_curr = compute_test_error(w_curr, X_test, y_test, multiclass)

        results['sdca']['passes'].append(effective_pass)
        results['sdca']['loss_residual'].append(float(loss_curr - P_star))
        results['sdca']['test_error'].append(float(err_curr))

        if (epoch + 1) % log_interval == 0 or epoch + 1 == n_epochs_sdca:
            print(f"    SDCA epoch {epoch+1:3d}: loss residual = {loss_curr - P_star:.2e}, "
                  f"test error = {err_curr:.2f}%")

    w_sdca = sdca_train(X_train, y_train, lam, n_epochs_sdca,
                        multiclass=multiclass, callback=sdca_callback)

    # ── Run SAG ──
    print(f"\n  Running SAG (lr={config['sag_lr']})...")
    w_sag = w.copy()
    effective_pass = config['warm_start_epochs']

    train_loss = loss(w_sag, X_train, y_train, lam, multiclass)
    test_err = compute_test_error(w_sag, X_test, y_test, multiclass)
    results['sag']['passes'].append(effective_pass)
    results['sag']['loss_residual'].append(float(train_loss - P_star))
    results['sag']['test_error'].append(float(test_err))

    # SAG uses callbacks to log per-epoch progress
    n_epochs_sag = config['n_epochs_sag']
    log_interval = max(1, n_epochs_sag // 30)

    def sag_callback(w_curr, epoch):
        nonlocal effective_pass
        effective_pass = config['warm_start_epochs'] + epoch + 1
        loss_curr = loss(w_curr, X_train, y_train, lam, multiclass)
        err_curr = compute_test_error(w_curr, X_test, y_test, multiclass)

        results['sag']['passes'].append(effective_pass)
        results['sag']['loss_residual'].append(float(loss_curr - P_star))
        results['sag']['test_error'].append(float(err_curr))

        if (epoch + 1) % log_interval == 0 or epoch + 1 == n_epochs_sag:
            print(f"    SAG epoch {epoch+1:3d}: loss residual = {loss_curr - P_star:.2e}, "
                  f"test error = {err_curr:.2f}%")

    w_sag = sag_train(X_train, y_train, lam, n_epochs_sag,
                      multiclass=multiclass, lr=config['sag_lr'],
                      callback=sag_callback)

    # ── Save results ──
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f'{dataset_name}_results.json')
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if x is not None else None)
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
