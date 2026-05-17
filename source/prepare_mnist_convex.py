"""
prepare_mnist_convex.py — Tạo w_init dùng cho SVRG (warm-start)

Saves: results/w_init_svrg_mnist.npy
"""

import os, sys
import numpy as np

np.random.seed(42)
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_dataset
from optimizers.sgd import warm_start

DATASET       = 'mnist'
LAM           = 1e-4
MULTICLASS    = True
WARM_EPOCHS   = 1          # convex case: 1 epoch
WARM_LR       = 0.01
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'results')
W_INIT_SVRG_PATH = os.path.join(RESULTS_DIR, 'w_init_svrg_mnist.npy')
W_RANDOM_PATH    = os.path.join(RESULTS_DIR, 'w_random_init_mnist.npy')


def main():
    print("=" * 50)
    print("Preparing initial weights for MNIST convex")
    print("=" * 50)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, y_train, _, _ = load_dataset(DATASET)
    n, d = X_train.shape
    print(f"Dataset: n={n}, d={d}")

    # 1. Warm-start initialization for SVRG (as per paper)
    print(f"Warm-start for SVRG: {WARM_EPOCHS} epoch SGD (lr={WARM_LR}) ...")
    w_svrg_init = warm_start(X_train, y_train, LAM, MULTICLASS,
                              n_epochs=WARM_EPOCHS, lr=WARM_LR)
    np.save(W_INIT_SVRG_PATH, w_svrg_init)
    print(f"[OK] SVRG init saved -> {os.path.basename(W_INIT_SVRG_PATH)} shape={w_svrg_init.shape}")

    # 2. Random initialization for SGD and SDCA (starting from x=0)
    if MULTICLASS:
        K = len(np.unique(y_train))
        w_random = np.zeros((d, K))  # zero initialization for multi-class
    else:
        w_random = np.zeros(d)
    np.save(W_RANDOM_PATH, w_random)
    print(f"[OK] Random init (zero) saved -> {os.path.basename(W_RANDOM_PATH)} shape={w_random.shape}")


if __name__ == '__main__':
    main()