"""
train.py — Flexible Training Runner

Cho phép chạy từng dataset, từng model, learning rate tuỳ chỉnh qua CLI.

Cách dùng:
  # Chạy tất cả dataset, tất cả model (mặc định)
  python train.py

  # Chỉ chạy 1 dataset
  python train.py --dataset mnist

  # Chỉ chạy 1 model
  python train.py --model svrg

  # Chỉ chạy dataset + model cụ thể, override learning rate
  python train.py --dataset mnist --model svrg --lr 0.01

  # Override nhiều hyper-param cùng lúc
  python train.py --dataset mnist --model sgd_const --lr 0.05 --epochs 30

  # Liệt kê dataset/model hợp lệ
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

VALID_MODELS = ['svrg', 'sgd_const', 'sgd_best']
CHECKPOINT_DIR = 'checkpoints'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Flexible training runner — chạy từng dataset / model / lr',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str, default=None,
        help='Tên dataset cần chạy. Mặc định: chạy tất cả.\n'
             f'Hợp lệ: {list(DATASET_CONFIGS.keys())}',
    )
    parser.add_argument(
        '--model', '-m',
        type=str, default=None,
        choices=VALID_MODELS + [None],
        help='Model cần chạy. Mặc định: chạy tất cả.\n'
             f'Hợp lệ: {VALID_MODELS}',
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Override learning rate cho model được chọn.\n'
             'SVRG → svrg_lr | sgd_const → sgd_const_lr | sgd_best → sgd_best_lr0',
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override số epoch / outer iteration.',
    )
    parser.add_argument(
        '--lam', type=float, default=None,
        help='Override regularisation lambda.',
    )
    parser.add_argument(
        '--no-warmstart', action='store_true',
        help='Bỏ qua warm-start (chỉ ảnh hưởng SVRG).',
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Thư mục lưu kết quả (mặc định: results/).',
    )
    parser.add_argument(
        '--save-ckpt-every', type=int, default=5,
        help='Lưu checkpoint mỗi N epoch (mặc định: 5).',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='In danh sách dataset và model hợp lệ rồi thoát.',
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers — Test Error
# ---------------------------------------------------------------------------

def compute_test_error(w, X_test, y_test, multiclass=False):
    """Tính tỉ lệ lỗi trên tập test (%)."""
    if multiclass:
        preds = np.argmax(X_test @ w, axis=1)
    else:
        preds = np.sign(X_test @ w)
    return np.mean(preds != y_test) * 100


# ---------------------------------------------------------------------------
# Helpers — Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(tag, dataset_name, w, passes, loss_val, test_err,
                    variance=None, epoch=0):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'{tag}_{dataset_name}_epoch{epoch}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({
            'tag': tag, 'dataset': dataset_name, 'epoch': epoch,
            'passes': passes, 'loss': loss_val, 'test_error': test_err,
            'variance': variance, 'weights': w,
        }, f)


def clean_checkpoints(tag, dataset_name, keep_last=2):
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
# Individual model runners
# ---------------------------------------------------------------------------

def run_svrg(w_init, X_train, y_train, X_test, y_test,
             config, P_star, dataset_name,
             save_ckpt_every=5, skip_warmstart=False):
    """Chạy SVRG, trả về dict kết quả."""
    lam        = config['lam']
    multiclass = config['multiclass']
    n          = X_train.shape[0]
    m          = config['svrg_m_factor'] * n

    print(f"\n  [SVRG] lr={config['svrg_lr']}, m={config['svrg_m_factor']}*n={m}, "
          f"outer_iters={config['n_outer']}")

    # Warm-start
    if skip_warmstart:
        w = w_init.copy()
        ep_offset = 0.0
    else:
        print(f"  Warm-start: {config['warm_start_epochs']} epoch(s)")
        w = warm_start(X_train, y_train, lam, multiclass,
                       n_epochs=config['warm_start_epochs'],
                       lr=config['warm_start_lr'])
        ep_offset = float(config['warm_start_epochs'])

    result = {'passes': [], 'loss': [], 'loss_residual': [], 'test_error': [], 'grad_variance': []}

    # Log trạng thái ban đầu
    def _log(ep, variance=None):
        tl  = loss(w, X_train, y_train, lam, multiclass)
        ter = compute_test_error(w, X_test, y_test, multiclass)
        result['passes'].append(ep)
        result['loss'].append(float(tl))
        result['loss_residual'].append(float(tl - P_star))
        result['test_error'].append(float(ter))
        result['grad_variance'].append(float(variance) if variance is not None else None)
        return tl, ter

    _log(ep_offset)

    for s in range(config['n_outer']):
        w, variance = svrg_outer_loop(
            w, X_train, y_train,
            lr=config['svrg_lr'], lam=lam, m=m,
            multiclass=multiclass, option='I', track_variance=True,
        )
        ep_offset += effective_passes_svrg(n, m)
        tl, ter = _log(ep_offset, variance)

        if (s + 1) % save_ckpt_every == 0:
            save_checkpoint('svrg', dataset_name, w, ep_offset,
                            tl, ter, variance, epoch=s + 1)
            clean_checkpoints('svrg', dataset_name)

        if (s + 1) % 5 == 0:
            print(f"    iter {s+1:3d} | residual={tl-P_star:.2e} "
                  f"| err={ter:.2f}% | var={variance:.2e}")

    return result


def run_sgd_const(w_init, X_train, y_train, X_test, y_test,
                  config, P_star, dataset_name, save_ckpt_every=5):
    """Chạy SGD với learning rate hằng số."""
    lam        = config['lam']
    multiclass = config['multiclass']

    print(f"\n  [SGD-const] lr={config['sgd_const_lr']}, "
          f"epochs={config['n_epochs_sgd']}")

    w      = w_init.copy()
    ep     = 0.0
    result = {'passes': [], 'loss': [], 'loss_residual': [], 'test_error': [], 'grad_variance': []}

    # Log trạng thái ban đầu
    def _log(ep, variance=None):
        tl  = loss(w, X_train, y_train, lam, multiclass)
        ter = compute_test_error(w, X_test, y_test, multiclass)
        result['passes'].append(ep)
        result['loss'].append(float(tl))
        result['loss_residual'].append(float(tl - P_star))
        result['test_error'].append(float(ter))
        result['grad_variance'].append(float(variance) if variance is not None else None)
        return tl, ter

    _log(ep)

    for epoch in range(config['n_epochs_sgd']):
        w, variance = sgd_epoch_constant(w, X_train, y_train,
                                 lr=config['sgd_const_lr'],
                                 lam=lam, multiclass=multiclass, track_variance=True)
        ep += 1.0
        tl, ter = _log(ep, variance)

        if (epoch + 1) % save_ckpt_every == 0:
            save_checkpoint('sgd_const', dataset_name, w, ep,
                            tl, ter, variance, epoch=epoch + 1)
            clean_checkpoints('sgd_const', dataset_name)

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d} | residual={tl-P_star:.2e} | err={ter:.2f}% | var={variance:.2e}" )

    return result


def run_sgd_best(w_init, X_train, y_train, X_test, y_test,
                 config, P_star, dataset_name, save_ckpt_every=5):
    """Chạy SGD với learning rate giảm dần."""
    lam        = config['lam']
    multiclass = config['multiclass']
    n          = X_train.shape[0]

    print(f"\n  [SGD-best] lr0={config['sgd_best_lr0']}, b={config['sgd_best_b']}, "
          f"epochs={config['n_epochs_sgd']}")

    w      = w_init.copy()
    ep     = 0.0
    t      = 0
    result = {'passes': [], 'loss': [], 'loss_residual': [], 'test_error': [], 'grad_variance': []}

    def _log(ep, variance=None):
        tl  = loss(w, X_train, y_train, lam, multiclass)
        ter = compute_test_error(w, X_test, y_test, multiclass)
        result['passes'].append(ep)
        result['loss'].append(float(tl))
        result['loss_residual'].append(float(tl - P_star))
        result['test_error'].append(float(ter))
        result['grad_variance'].append(float(variance) if variance is not None else None)  # Placeholder, không track variance cho SGD-best
        return tl, ter

    _log(ep)

    for epoch in range(config['n_epochs_sgd']):
        w, t, variance = sgd_epoch_decay(w, X_train, y_train,
                                          lr0=config['sgd_best_lr0'],
                                          lam=lam, n=n, t_start=t,
                                          b=config['sgd_best_b'],
                               multiclass=multiclass, track_variance=True)
        ep += 1.0
        tl, ter = _log(ep, variance)

        if (epoch + 1) % save_ckpt_every == 0:
            save_checkpoint('sgd_best', dataset_name, w, ep,
                            tl, ter, variance, epoch=epoch + 1)
            clean_checkpoints('sgd_best', dataset_name)

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d} | residual={tl-P_star:.2e} | err={ter:.2f}% | var={variance:.2e}")

    return result


# ---------------------------------------------------------------------------
# Per-dataset experiment
# ---------------------------------------------------------------------------

def run_experiment(dataset_name, config, P_star,
                   models=None, results_dir='results',
                   save_ckpt_every=5, skip_warmstart=False):
    """Chạy thí nghiệm cho 1 dataset với các model được chọn.

    Args:
        models: list subset của VALID_MODELS, None = tất cả
    """
    if models is None:
        models = VALID_MODELS

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")
    print(f"Models  : {models}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    n, d = X_train.shape
    multiclass = config['multiclass']

    print(f"  n={n}, d={d}, multiclass={multiclass}, λ={config['lam']}")

    # Weight khởi tạo
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
                                           config, P_star, dataset_name,
                                           save_ckpt_every),
        'sgd_best':  lambda: run_sgd_best(w0, X_train, y_train, X_test, y_test,
                                          config, P_star, dataset_name,
                                          save_ckpt_every),
    }

    for model in models:
        all_results[model] = runner_map[model]()

    # Lưu kết quả JSON
    out_path = os.path.join(results_dir, f'{dataset_name}_results.json')
    with open(out_path, 'w') as f:
        # grad_variance có thể chứa None → dùng default=str
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ Kết quả đã lưu → {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# Config override helpers
# ---------------------------------------------------------------------------

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
    """Trả về bản copy của config với các giá trị được override."""
    cfg = dict(config)

    if lr is not None:
        key = LR_KEY_MAP.get(model)
        if key:
            old = cfg.get(key)
            cfg[key] = lr
            print(f"  [override] {key}: {old} → {lr}")

    if epochs is not None:
        key = EPOCH_KEY_MAP.get(model)
        if key:
            old = cfg.get(key)
            cfg[key] = epochs
            print(f"  [override] {key}: {old} → {epochs}")

    if lam is not None:
        old = cfg.get('lam')
        cfg['lam'] = lam
        print(f"  [override] lam: {old} → {lam}")

    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --list
    if args.list:
        print("Datasets hợp lệ:")
        for ds in DATASET_CONFIGS:
            print(f"  {ds}")
        print("\nModels hợp lệ:")
        for m in VALID_MODELS:
            print(f"  {m}")
        return

    # Xác định datasets cần chạy
    if args.dataset is not None:
        if args.dataset not in DATASET_CONFIGS:
            print(f"ERROR: Dataset '{args.dataset}' không hợp lệ.")
            print(f"       Hợp lệ: {list(DATASET_CONFIGS.keys())}")
            return
        datasets = [args.dataset]
    else:
        datasets = list(DATASET_CONFIGS.keys())

    # Xác định models cần chạy
    models = [args.model] if args.model else VALID_MODELS

    # Load optimal losses
    optimal_path = os.path.join(args.results_dir, 'optimal_loss.json')
    if not os.path.exists(optimal_path):
        print(f"ERROR: Không tìm thấy {optimal_path}. Chạy compute_optimal.py trước.")
        return

    with open(optimal_path) as f:
        optimal_losses = json.load(f)

    # Vòng lặp chính
    all_results = {}
    for ds in datasets:
        base_config = dict(DATASET_CONFIGS[ds])
        P_star      = optimal_losses[ds]['P_star']

        # Override config nếu có (override áp dụng cho từng model riêng)
        if args.lr is not None or args.epochs is not None or args.lam is not None:
            # Nếu chỉ chạy 1 model → override rõ ràng
            # Nếu chạy nhiều model → override lần lượt từng model
            merged_config = base_config.copy()
            for m in models:
                merged_config = apply_overrides(
                    merged_config, m,
                    lr=args.lr, epochs=args.epochs, lam=args.lam
                )
            config = merged_config
        else:
            config = base_config

        result = run_experiment(
            dataset_name   = ds,
            config         = config,
            P_star         = P_star,
            models         = models,
            results_dir    = args.results_dir,
            save_ckpt_every= args.save_ckpt_every,
            skip_warmstart = args.no_warmstart,
        )
        all_results[ds] = result

    print(f"\n{'='*60}")
    print("Hoàn thành tất cả thí nghiệm.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()