"""
train.py — Flexible Training Runner

Run experiments for any combination of datasets and optimizers via CLI.

Usage examples:
    # Run all datasets, all models (default)
    python train.py

    # Single dataset
    python train.py --dataset mnist

    # Single optimizer
    python train.py --model svrg

    # Dataset + model + override learning rate
    python train.py --dataset mnist --model svrg --lr 0.01

    # Override multiple hyperparameters
    python train.py --dataset mnist --model sgd_const --lr 0.05 --epochs 30

    # List valid datasets and models
    python train.py --list
"""

import os
import json
import pickle
import argparse

import numpy as np

from utils.data_loader import load_dataset
from models.logistic import loss
from optimizers.sgd import sgd_epoch_constant, sgd_epoch_decay, warm_start
from optimizers.svrg import svrg_outer_loop, effective_passes_svrg
from config import DATASET_CONFIGS

VALID_MODELS   = ['svrg', 'sgd_const', 'sgd_best']
CHECKPOINT_DIR = 'checkpoints'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Flexible training runner — dataset / model / lr',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--dataset', '-d', type=str, default=None,
        help=f'Dataset to run. Default: all.\nValid: {list(DATASET_CONFIGS.keys())}',
    )
    parser.add_argument(
        '--model', '-m', type=str, default=None, choices=VALID_MODELS + [None],
        help=f'Optimizer to run. Default: all.\nValid: {VALID_MODELS}',
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help=(
            'Override learning rate for the selected model.\n'
            'SVRG → svrg_lr | sgd_const → sgd_const_lr | sgd_best → sgd_best_lr0'
        ),
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of epochs / outer iterations.',
    )
    parser.add_argument(
        '--lam', type=float, default=None,
        help='Override regularization lambda.',
    )
    parser.add_argument(
        '--no-warmstart', action='store_true',
        help='Skip SVRG warm-start.',
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Directory for saving results (default: results/).',
    )
    parser.add_argument(
        '--save-ckpt-every', type=int, default=5,
        help='Save a checkpoint every N epochs (default: 5).',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='Print valid datasets and models, then exit.',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_test_error(w, X_test, y_test, multiclass=False):
    """Return test error rate (%) for the current weights."""
    if multiclass:
        preds = np.argmax(X_test @ w, axis=1)
    else:
        preds = np.sign(X_test @ w)
    return np.mean(preds != y_test) * 100.0


def save_checkpoint(tag, dataset_name, w, passes, loss_val, test_err,
                    variance=None, epoch=0):
    """Persist a training checkpoint to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'{tag}_{dataset_name}_epoch{epoch}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({
            'tag': tag, 'dataset': dataset_name, 'epoch': epoch,
            'passes': passes, 'loss': loss_val, 'test_error': test_err,
            'variance': variance, 'weights': w,
        }, f)


def clean_checkpoints(tag, dataset_name, keep_last=2):
    """Remove old checkpoints, keeping only the most recent `keep_last` files."""
    if not os.path.exists(CHECKPOINT_DIR):
        return
    prefix = f'{tag}_{dataset_name}_epoch'
    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR)
         if f.startswith(prefix) and f.endswith('.pkl')],
        key=lambda f: int(f.replace(prefix, '').replace('.pkl', '') or -1),
    )
    for f in files[:-keep_last]:
        os.remove(os.path.join(CHECKPOINT_DIR, f))


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def _make_result_dict():
    return {'passes': [], 'loss': [], 'loss_residual': [], 'test_error': [], 'grad_variance': []}


def _log_state(result, w, passes, X_train, y_train, X_test, y_test, lam, multiclass, P_star, variance=None):
    """Evaluate current weights and append metrics to result dict."""
    train_loss = loss(w, X_train, y_train, lam, multiclass)
    test_err   = compute_test_error(w, X_test, y_test, multiclass)

    result['passes'].append(passes)
    result['loss'].append(float(train_loss))
    result['loss_residual'].append(float(train_loss - P_star))
    result['test_error'].append(float(test_err))
    result['grad_variance'].append(float(variance) if variance is not None else None)

    return train_loss, test_err


def run_svrg(w_init, X_train, y_train, X_test, y_test,
             config, P_star, dataset_name,
             save_ckpt_every=5, skip_warmstart=False):
    """Run SVRG and return a results dict."""
    lam        = config['lam']
    multiclass = config['multiclass']
    n          = X_train.shape[0]
    m          = config['svrg_m_factor'] * n

    print(f"\n  [SVRG] lr={config['svrg_lr']}, m={config['svrg_m_factor']}*n={m}, "
          f"outer_iters={config['n_outer']}")

    if skip_warmstart:
        w, ep_offset = w_init.copy(), 0.0
    else:
        print(f"  Warm-start: {config['warm_start_epochs']} epoch(s)")
        w = warm_start(X_train, y_train, lam, multiclass,
                       n_epochs=config['warm_start_epochs'],
                       lr=config['warm_start_lr'])
        ep_offset = float(config['warm_start_epochs'])

    result = _make_result_dict()
    _log_state(result, w, ep_offset, X_train, y_train, X_test, y_test, lam, multiclass, P_star)

    for s in range(config['n_outer']):
        w, variance = svrg_outer_loop(
            w, X_train, y_train,
            lr=config['svrg_lr'], lam=lam, m=m,
            multiclass=multiclass, option='I', track_variance=True,
        )
        ep_offset += effective_passes_svrg(n, m)
        tl, ter = _log_state(result, w, ep_offset, X_train, y_train, X_test, y_test,
                             lam, multiclass, P_star, variance)

        if (s + 1) % save_ckpt_every == 0:
            save_checkpoint('svrg', dataset_name, w, ep_offset, tl, ter, variance, epoch=s + 1)
            clean_checkpoints('svrg', dataset_name)

        if (s + 1) % 5 == 0:
            print(f"    iter {s+1:3d} | residual={tl-P_star:.2e} | err={ter:.2f}% | var={variance:.2e}")

    return result


def run_sgd_const(w_init, X_train, y_train, X_test, y_test,
                  config, P_star, dataset_name, save_ckpt_every=5):
    """Run SGD with a constant learning rate and return a results dict."""
    lam        = config['lam']
    multiclass = config['multiclass']

    print(f"\n  [SGD-const] lr={config['sgd_const_lr']}, epochs={config['n_epochs_sgd']}")

    w      = w_init.copy()
    ep     = 0.0
    result = _make_result_dict()
    _log_state(result, w, ep, X_train, y_train, X_test, y_test, lam, multiclass, P_star)

    for epoch in range(config['n_epochs_sgd']):
        w, variance = sgd_epoch_constant(
            w, X_train, y_train,
            lr=config['sgd_const_lr'], lam=lam,
            multiclass=multiclass, track_variance=True,
        )
        ep += 1.0
        tl, ter = _log_state(result, w, ep, X_train, y_train, X_test, y_test,
                             lam, multiclass, P_star, variance)

        if (epoch + 1) % save_ckpt_every == 0:
            save_checkpoint('sgd_const', dataset_name, w, ep, tl, ter, variance, epoch=epoch + 1)
            clean_checkpoints('sgd_const', dataset_name)

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d} | residual={tl-P_star:.2e} | err={ter:.2f}% | var={variance:.2e}")

    return result


def run_sgd_best(w_init, X_train, y_train, X_test, y_test,
                 config, P_star, dataset_name, save_ckpt_every=5):
    """Run SGD with a decaying learning rate and return a results dict."""
    lam        = config['lam']
    multiclass = config['multiclass']
    n          = X_train.shape[0]

    print(f"\n  [SGD-best] lr0={config['sgd_best_lr0']}, b={config['sgd_best_b']}, "
          f"epochs={config['n_epochs_sgd']}")

    w      = w_init.copy()
    ep     = 0.0
    t      = 0
    result = _make_result_dict()
    _log_state(result, w, ep, X_train, y_train, X_test, y_test, lam, multiclass, P_star)

    for epoch in range(config['n_epochs_sgd']):
        w, t, variance = sgd_epoch_decay(
            w, X_train, y_train,
            lr0=config['sgd_best_lr0'], lam=lam, n=n, t_start=t,
            b=config['sgd_best_b'], multiclass=multiclass, track_variance=True,
        )
        ep += 1.0
        tl, ter = _log_state(result, w, ep, X_train, y_train, X_test, y_test,
                             lam, multiclass, P_star, variance)

        if (epoch + 1) % save_ckpt_every == 0:
            save_checkpoint('sgd_best', dataset_name, w, ep, tl, ter, variance, epoch=epoch + 1)
            clean_checkpoints('sgd_best', dataset_name)

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d} | residual={tl-P_star:.2e} | err={ter:.2f}% | var={variance:.2e}")

    return result


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------

def run_experiment(dataset_name, config, P_star, models=None,
                   results_dir='results', save_ckpt_every=5, skip_warmstart=False):
    """Run an experiment for one dataset with the specified optimizers.

    Args:
        dataset_name:   name of the dataset to load
        config:         hyperparameter dict for this dataset
        P_star:         optimal loss P(w*) for loss-residual computation
        models:         list of model keys to run (None = all)
        results_dir:    directory to write JSON results
        save_ckpt_every: checkpoint frequency in epochs
        skip_warmstart: skip SVRG warm-start if True
    """
    if models is None:
        models = VALID_MODELS

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")
    print(f"Models  : {models}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    n, d       = X_train.shape
    multiclass = config['multiclass']
    print(f"  n={n}, d={d}, multiclass={multiclass}, λ={config['lam']}")

    # Zero-initialize weights
    if multiclass:
        K  = len(np.unique(y_train))
        w0 = np.zeros((d, K))
    else:
        w0 = np.zeros(d)

    os.makedirs(results_dir, exist_ok=True)
    all_results = {'dataset': dataset_name, 'config': dict(config), 'P_star': P_star}

    runner_map = {
        'svrg':      lambda: run_svrg(w0, X_train, y_train, X_test, y_test,
                                      config, P_star, dataset_name,
                                      save_ckpt_every, skip_warmstart),
        'sgd_const': lambda: run_sgd_const(w0, X_train, y_train, X_test, y_test,
                                           config, P_star, dataset_name, save_ckpt_every),
        'sgd_best':  lambda: run_sgd_best(w0, X_train, y_train, X_test, y_test,
                                          config, P_star, dataset_name, save_ckpt_every),
    }

    for model in models:
        all_results[model] = runner_map[model]()

    out_path = os.path.join(results_dir, f'{dataset_name}_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ Results saved → {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# Config override helpers
# ---------------------------------------------------------------------------

# Maps model key → config key for learning rate and epoch count
LR_KEY_MAP = {
    'svrg':      'svrg_lr',
    'sgd_const': 'sgd_const_lr',
    'sgd_best':  'sgd_best_lr0',
}

EPOCH_KEY_MAP = {
    'svrg':      'n_outer',
    'sgd_const': 'n_epochs_sgd',
    'sgd_best':  'n_epochs_sgd',
}


def apply_overrides(config, model, lr=None, epochs=None, lam=None):
    """Return a copy of config with CLI overrides applied."""
    cfg = dict(config)

    if lr is not None and (key := LR_KEY_MAP.get(model)):
        print(f"  [override] {key}: {cfg.get(key)} → {lr}")
        cfg[key] = lr

    if epochs is not None and (key := EPOCH_KEY_MAP.get(model)):
        print(f"  [override] {key}: {cfg.get(key)} → {epochs}")
        cfg[key] = epochs

    if lam is not None:
        print(f"  [override] lam: {cfg.get('lam')} → {lam}")
        cfg['lam'] = lam

    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.list:
        print("Valid datasets:")
        for ds in DATASET_CONFIGS:
            print(f"  {ds}")
        print("\nValid models:")
        for m in VALID_MODELS:
            print(f"  {m}")
        return

    # Resolve datasets and models to run
    if args.dataset is not None:
        if args.dataset not in DATASET_CONFIGS:
            print(f"ERROR: Unknown dataset '{args.dataset}'. "
                  f"Valid options: {list(DATASET_CONFIGS.keys())}")
            return
        datasets = [args.dataset]
    else:
        datasets = list(DATASET_CONFIGS.keys())

    models = [args.model] if args.model else VALID_MODELS

    # Load precomputed optimal losses
    optimal_path = os.path.join(args.results_dir, 'optimal_loss.json')
    if not os.path.exists(optimal_path):
        print(f"ERROR: {optimal_path} not found. Run compute_optimal.py first.")
        return

    with open(optimal_path) as f:
        optimal_losses = json.load(f)

    # Main loop
    all_results = {}
    for ds in datasets:
        base_config = dict(DATASET_CONFIGS[ds])
        P_star      = optimal_losses[ds]['P_star']

        if args.lr is not None or args.epochs is not None or args.lam is not None:
            config = base_config.copy()
            for m in models:
                config = apply_overrides(config, m, lr=args.lr,
                                         epochs=args.epochs, lam=args.lam)
        else:
            config = base_config

        all_results[ds] = run_experiment(
            dataset_name    = ds,
            config          = config,
            P_star          = P_star,
            models          = models,
            results_dir     = args.results_dir,
            save_ckpt_every = args.save_ckpt_every,
            skip_warmstart  = args.no_warmstart,
        )

    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()