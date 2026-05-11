"""
run_parallel.py — Parallel Experiment Launcher

Chạy các thí nghiệm song song theo dataset:
  - compute_optimal : 4 dataset chạy đồng thời trên 4 cores
  - train (convex)  : 4 dataset chạy đồng thời
  - train_nn        : 2 dataset chạy đồng thời

Cách chạy (từ thư mục source/):
  python run_parallel.py                     # chạy tất cả
  python run_parallel.py --step optimal      # chỉ tính P(w*)
  python run_parallel.py --step convex       # chỉ convex experiments
  python run_parallel.py --step nn           # chỉ NN experiments
  python run_parallel.py --workers 2         # giới hạn số cores

Lưu ý: numpy dùng multi-threaded BLAS. Khi chạy nhiều process, mỗi process
sẽ tự dùng thêm BLAS threads → có thể bị contention. Script này set
OMP_NUM_THREADS=1 để tránh nested parallelism.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Worker functions (phải ở top-level để picklable trên Windows)
# ---------------------------------------------------------------------------

def _worker_init():
    """Limit numpy BLAS threads inside each worker to avoid nested parallelism."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _run_optimal_one(args):
    """Worker: compute P(w*) for one dataset."""
    _worker_init()
    name, lam, multiclass = args
    from compute_optimal import compute_one
    print(f"  [optimal] START  {name}")
    t0 = time.time()
    result = compute_one(name, lam, multiclass)
    elapsed = time.time() - t0
    print(f"  [optimal] DONE   {name}  P*={result['P_star']:.8f}  ({elapsed:.0f}s)")
    return name, result


def _run_convex_one(args):
    """Worker: run full convex experiment for one dataset."""
    _worker_init()
    dataset_name, config, P_star = args
    from train import run_experiment
    print(f"  [convex]  START  {dataset_name}")
    t0 = time.time()
    result = run_experiment(dataset_name, config, P_star)
    elapsed = time.time() - t0
    print(f"  [convex]  DONE   {dataset_name}  ({elapsed:.0f}s)")
    return dataset_name, result


def _run_nn_one(args):
    """Worker: run full NN experiment for one dataset."""
    _worker_init()
    dataset_name, config, P_star = args
    from train_nn import run_nn_experiment
    print(f"  [nn]      START  {dataset_name}")
    t0 = time.time()
    result = run_nn_experiment(dataset_name, config, P_star=P_star)
    elapsed = time.time() - t0
    print(f"  [nn]      DONE   {dataset_name}  ({elapsed:.0f}s)")
    return dataset_name, result


# ---------------------------------------------------------------------------
# Step 1: Compute P(w*) in parallel
# ---------------------------------------------------------------------------

def step_optimal(workers, optimal_path='results/optimal_loss.json'):
    print(f"\n{'='*60}")
    print(f"STEP 1: Computing P(w*) — parallel across 4 datasets")
    print(f"{'='*60}")

    tasks = [
        ('mnist',   1e-4, True),
        ('cifar10', 1e-3, True),
        ('rcv1',    1e-5, False),
        ('covtype', 1e-5, False),
    ]

    # Load existing results to allow partial reruns
    os.makedirs(os.path.dirname(optimal_path), exist_ok=True)
    existing = {}
    if os.path.exists(optimal_path):
        with open(optimal_path) as f:
            existing = json.load(f)

    pending = [t for t in tasks if t[0] not in existing]
    if not pending:
        print("  All P(w*) already computed. Skipping.")
        return existing

    n_workers = min(workers, len(pending))
    print(f"  Running {len(pending)} datasets with {n_workers} workers...")
    t0 = time.time()

    results = dict(existing)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_optimal_one, t): t[0] for t in pending}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            # Save incrementally after each dataset finishes
            with open(optimal_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  [optimal] Saved {name} → {optimal_path}")

    print(f"\n  Total time: {time.time() - t0:.0f}s")
    return results


# ---------------------------------------------------------------------------
# Step 2: Convex experiments in parallel
# ---------------------------------------------------------------------------

def step_convex(workers, optimal_path='results/optimal_loss.json',
                results_dir='results'):
    print(f"\n{'='*60}")
    print(f"STEP 2: Convex experiments — parallel across 4 datasets")
    print(f"{'='*60}")

    from config import DATASET_CONFIGS

    if not os.path.exists(optimal_path):
        print(f"  ERROR: {optimal_path} not found. Run --step optimal first.")
        return {}

    with open(optimal_path) as f:
        optimal_losses = json.load(f)

    tasks = []
    for name, config in DATASET_CONFIGS.items():
        P_star = optimal_losses.get(name, {}).get('P_star', None)
        tasks.append((name, config, P_star))

    n_workers = min(workers, len(tasks))
    print(f"  Running {len(tasks)} datasets with {n_workers} workers...")
    t0 = time.time()

    all_results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_convex_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            name, result = future.result()
            all_results[name] = result

    print(f"\n  Total time: {time.time() - t0:.0f}s")
    return all_results


# ---------------------------------------------------------------------------
# Step 3: NN experiments in parallel
# ---------------------------------------------------------------------------

def step_nn(workers, optimal_path='results/optimal_loss.json',
            results_dir='results'):
    print(f"\n{'='*60}")
    print(f"STEP 3: NN experiments — parallel across 2 datasets")
    print(f"{'='*60}")

    from config import NN_CONFIGS

    optimal_losses = {}
    if os.path.exists(optimal_path):
        with open(optimal_path) as f:
            optimal_losses = json.load(f)
    else:
        print(f"  [!] {optimal_path} not found — logging raw loss.")

    tasks = []
    for dataset_name, config in NN_CONFIGS.items():
        base_name = dataset_name.replace('_nn', '')
        P_star = optimal_losses.get(base_name, {}).get('P_star', None)
        tasks.append((dataset_name, config, P_star))

    n_workers = min(workers, len(tasks))
    print(f"  Running {len(tasks)} datasets with {n_workers} workers...")
    t0 = time.time()

    all_results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_nn_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            name, result = future.result()
            all_results[name] = result

    print(f"\n  Total time: {time.time() - t0:.0f}s")
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Parallel experiment launcher')
    parser.add_argument('--step', choices=['optimal', 'convex', 'nn', 'all'],
                        default='all',
                        help='Which step to run (default: all)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Max parallel workers (default: 4 = number of datasets)')
    args = parser.parse_args()

    print(f"Workers: {args.workers}  |  Step: {args.step}")

    if args.step in ('optimal', 'all'):
        step_optimal(args.workers)

    if args.step in ('convex', 'all'):
        step_convex(args.workers)

    if args.step in ('nn', 'all'):
        step_nn(args.workers)

    print(f"\n{'='*60}")
    print("All requested steps complete.")
    print(f"{'='*60}")


if __name__ == '__main__':
    # Windows requires this guard for multiprocessing
    main()
