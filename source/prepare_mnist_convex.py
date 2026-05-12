"""
prepare_mnist_convex.py — Tạo w_init dùng chung (chạy 1 lần trước tất cả)

Saves: results/w_init_mnist.npy
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
WARM_EPOCHS   = 1
WARM_LR       = 0.01
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'results')
W_INIT_PATH   = os.path.join(RESULTS_DIR, 'w_init_mnist.npy')


def main():
    print("=" * 50)
    print("Preparing shared w_init for MNIST convex")
    print("=" * 50)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, y_train, _, _ = load_dataset(DATASET)
    n, d = X_train.shape
    print(f"Dataset: n={n}, d={d}")

    print(f"Warm-start: {WARM_EPOCHS} epoch SGD (lr={WARM_LR}) ...")
    w_init = warm_start(X_train, y_train, LAM, MULTICLASS,
                        n_epochs=WARM_EPOCHS, lr=WARM_LR)

    np.save(W_INIT_PATH, w_init)
    print(f"[OK] w_init saved -> {os.path.basename(W_INIT_PATH)}  shape={w_init.shape}")


if __name__ == '__main__':
    main()
