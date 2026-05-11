"""
data_loader.py — Load & Preprocess 4 Datasets (MNIST, CIFAR-10, RCV1, Covtype)

Preprocessing follows data_exploration_preprocessing.ipynb (Section 9 summary):
  - MNIST    : divide by 255 → [0, 1]  
  - CIFAR-10 : divide by 255 → [0, 1] 
  - RCV1     : MaxAbsScaler
  - Covtype  : dùng file data/covtype/covtype.libsvm.binary.scale/covtype.libsvm.binary.scale
               và chia 50/50 train/test

All loaders return (X_train, y_train, X_test, y_test).
"""

import os
import numpy as np
import pickle
import gzip
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

# Base data directory (relative to this file: source/utils/ → ../../data)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')


# ---------------------------------------------------------------------------
# 1. MNIST
# ---------------------------------------------------------------------------

def _read_idx_images(filepath):
    """Read IDX image file, return (n, rows*cols) uint8 array."""
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        assert magic == 2051, f"Bad image magic: {magic}"
        n    = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data


def _read_idx_labels(filepath):
    """Read IDX label file, return (n,) int64 array."""
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        assert magic == 2049, f"Bad label magic: {magic}"
        n      = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_mnist():
    """Load MNIST, normalize by /255 → [0, 1].

    Returns:
        X_train : (60000, 784) float64 in [0, 1]
        y_train : (60000,) int64  {0..9}
        X_test  : (10000, 784) float64 in [0, 1]
        y_test  : (10000,) int64  {0..9}
    """
    base = os.path.join(DATA_DIR, 'mnist')
    X_train = _read_idx_images(
        os.path.join(base, 'train-images-idx3-ubyte', 'train-images.idx3-ubyte')
    ).astype(np.float64) / 255.0

    y_train = _read_idx_labels(
        os.path.join(base, 'train-labels-idx1-ubyte', 'train-labels.idx1-ubyte')
    )
    X_test = _read_idx_images(
        os.path.join(base, 't10k-images-idx3-ubyte', 't10k-images.idx3-ubyte')
    ).astype(np.float64) / 255.0

    y_test = _read_idx_labels(
        os.path.join(base, 't10k-labels-idx1-ubyte', 't10k-labels.idx1-ubyte')
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 2. CIFAR-10
# ---------------------------------------------------------------------------

def _unpickle(filepath):
    """Load a CIFAR-10 pickle batch. Returns (X uint8 (n,3072), y int64 (n,))."""
    with open(filepath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d[b'data'], np.array(d[b'labels'], dtype=np.int64)


def _cifar_normalize(X, ch_stats):
    """Apply per-channel Z-score normalization using precomputed stats.

    Args:
        X        : (n, 3072) float64
        ch_stats : list of (mean, std) per channel, fitted on train

    Returns:
        X_norm : (n, 3072) float64
    """
    X_out = X.copy()
    for ch, (mu, sig) in enumerate(ch_stats):
        s, e = ch * 1024, (ch + 1) * 1024
        X_out[:, s:e] = (X[:, s:e] - mu) / (sig + 1e-8)
    return X_out


def load_cifar10():
    """Load CIFAR-10, normalize by /255 → [0, 1].

    Returns:
        X_train : (50000, 3072) float64
        y_train : (50000,) int64  {0..9}
        X_test  : (10000, 3072) float64
        y_test  : (10000,) int64  {0..9}
    """
    base = os.path.join(DATA_DIR, 'cifar-10-python', 'cifar-10-batches-py')

    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        X, y = _unpickle(os.path.join(base, f'data_batch_{i}'))
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.vstack(X_train_list).astype(np.float64) / 255.0
    y_train = np.concatenate(y_train_list)
    X_test, y_test = _unpickle(os.path.join(base, 'test_batch'))
    X_test = X_test.astype(np.float64) / 255.0

    # Per-channel Z-score: fit stats on train only, apply to both
    # ch_stats = []
    # for ch in range(3):
    #     s, e = ch * 1024, (ch + 1) * 1024
    #     mu  = X_train[:, s:e].mean()
    #     sig = X_train[:, s:e].std()
    #     ch_stats.append((mu, sig))

    # X_train = _cifar_normalize(X_train, ch_stats)
    # X_test  = _cifar_normalize(X_test,  ch_stats)

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 3. RCV1
# ---------------------------------------------------------------------------

def load_rcv1():
    """Load RCV1, MaxAbsScaler.

    Uses rcv1_train.binary and rcv1_test.binary (official split).

    Returns:
        X_train : sparse CSR (20242, 47236) float64  
        y_train : (20242,) float64  {-1, +1}
        X_test  : sparse CSR (677399, 47236) float64 
        y_test  : (677399,) float64  {-1, +1}
    """
    train_path = os.path.join(DATA_DIR, 'rcv1', 'rcv1_train.binary', 'rcv1_train.binary')
    test_path  = os.path.join(DATA_DIR, 'rcv1', 'rcv1_test.binary',  'rcv1_test.binary')

    X_train, y_train = load_svmlight_file(train_path)
    X_test,  y_test  = load_svmlight_file(test_path, n_features=X_train.shape[1])

    y_train = y_train.astype(np.float64)
    y_test  = y_test.astype(np.float64)

    # MaxAbsScaler: fit on train only, apply to both
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 4. Covtype
# ---------------------------------------------------------------------------

def load_covtype():
    """Load Covtype, binarize labels, 50/50 split.

    Binarization: class 2 → +1.0, all others → -1.0 (as per paper).

    Returns:
        X_train : sparse CSR (≈290000, 54) float64  
        y_train : (≈290000,) float64  {-1, +1}
        X_test  : sparse CSR (≈290000, 54)
        y_test  : (≈290000,) float64
    """
    filepath = os.path.join(DATA_DIR, 'covtype', 'covtype.libsvm.binary.scale',
                            'covtype.libsvm.binary.scale')
    X, y = load_svmlight_file(filepath)

    # Binarize: class 2 → +1, rest → -1
    y_bin = np.where(y == 2, 1.0, -1.0)

    # 50/50 split per paper
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.5, random_state=42
    )

    # # MaxAbsScaler preserves sparsity
    # scaler = MaxAbsScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

DATASETS = {
    'mnist':   load_mnist,
    'cifar10': load_cifar10,
    'rcv1':    load_rcv1,
    'covtype': load_covtype,
}


def load_dataset(name):
    """Load and preprocess dataset by name.

    Args:
        name : one of {'mnist', 'cifar10', 'rcv1', 'covtype'}

    Returns:
        X_train : np.ndarray or scipy.sparse.csr_matrix
        y_train : np.ndarray
        X_test  : np.ndarray or scipy.sparse.csr_matrix
        y_test  : np.ndarray
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: '{name}'. Options: {list(DATASETS.keys())}")
    return DATASETS[name]()
