"""
data_loader.py — Load & Preprocess 4 Datasets (MNIST, CIFAR-10, RCV1, Covtype)

Preprocessing:
  - MNIST    : divide by 255 → [0, 1]
  - CIFAR-10 : divide by 255 → [0, 1]
  - RCV1     : MaxAbsScaler (fit on train, apply to both splits)
  - Covtype  : binary labels (class 2 → +1, others → -1), 50/50 train/test split

All loaders return (X_train, y_train, X_test, y_test).
"""

import os
import gzip
import pickle

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

# Base data directory: source/utils/ → ../../data
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data'
)


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

def _read_idx_images(filepath):
    """Read IDX image file. Returns (n, rows*cols) uint8 array."""
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
    """Read IDX label file. Returns (n,) int64 array."""
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        assert magic == 2049, f"Bad label magic: {magic}"
        n      = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_mnist():
    """Load MNIST and normalize pixel values to [0, 1].

    Returns:
        X_train : (60000, 784) float64
        y_train : (60000,)    int64  {0..9}
        X_test  : (10000, 784) float64
        y_test  : (10000,)    int64  {0..9}
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
# CIFAR-10
# ---------------------------------------------------------------------------

def _unpickle(filepath):
    """Load a CIFAR-10 pickle batch. Returns (X uint8 (n,3072), y int64 (n,))."""
    with open(filepath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d[b'data'], np.array(d[b'labels'], dtype=np.int64)


def load_cifar10():
    """Load CIFAR-10 and normalize pixel values to [0, 1].

    Returns:
        X_train : (50000, 3072) float64
        y_train : (50000,)     int64  {0..9}
        X_test  : (10000, 3072) float64
        y_test  : (10000,)     int64  {0..9}
    """
    base = os.path.join(DATA_DIR, 'cifar-10-python', 'cifar-10-batches-py')

    X_batches, y_batches = [], []
    for i in range(1, 6):
        X, y = _unpickle(os.path.join(base, f'data_batch_{i}'))
        X_batches.append(X)
        y_batches.append(y)

    X_train = np.vstack(X_batches).astype(np.float64) / 255.0
    y_train = np.concatenate(y_batches)
    X_test, y_test = _unpickle(os.path.join(base, 'test_batch'))
    X_test = X_test.astype(np.float64) / 255.0

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# RCV1
# ---------------------------------------------------------------------------

def load_rcv1():
    """Load RCV1 (official split) and apply MaxAbsScaler.

    Returns:
        X_train : sparse CSR (20242, 47236) float64
        y_train : (20242,)  float64  {-1, +1}
        X_test  : sparse CSR (677399, 47236) float64
        y_test  : (677399,) float64  {-1, +1}
    """
    train_path = os.path.join(DATA_DIR, 'rcv1', 'rcv1_train.binary', 'rcv1_train.binary')
    test_path  = os.path.join(DATA_DIR, 'rcv1', 'rcv1_test.binary',  'rcv1_test.binary')

    X_train, y_train = load_svmlight_file(train_path)
    X_test,  y_test  = load_svmlight_file(test_path, n_features=X_train.shape[1])

    y_train = y_train.astype(np.float64)
    y_test  = y_test.astype(np.float64)

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Covtype
# ---------------------------------------------------------------------------

def load_covtype():
    """Load Covtype with binarized labels and a 50/50 train/test split.

    Label binarization: class 2 → +1, all others → -1.

    Returns:
        X_train : sparse CSR (~290000, 54) float64
        y_train : (~290000,) float64  {-1, +1}
        X_test  : sparse CSR (~290000, 54) float64
        y_test  : (~290000,) float64  {-1, +1}
    """
    filepath = os.path.join(
        DATA_DIR, 'covtype', 'covtype.libsvm.binary.scale',
        'covtype.libsvm.binary.scale'
    )
    X, y = load_svmlight_file(filepath)
    y_bin = np.where(y == 2, 1.0, -1.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.5, random_state=42
    )
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
    """Load and preprocess a dataset by name.

    Args:
        name: one of {'mnist', 'cifar10', 'rcv1', 'covtype'}

    Returns:
        X_train, y_train, X_test, y_test
        (np.ndarray or scipy.sparse.csr_matrix depending on dataset)
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Valid options: {list(DATASETS.keys())}")
    return DATASETS[name]()