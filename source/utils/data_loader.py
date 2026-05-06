"""
data_loader.py — Load & Preprocess 4 Datasets (MNIST, CIFAR-10, RCV1, Covtype)

All loaders return (X_train, y_train, X_test, y_test) with consistent NumPy/SciPy formats.
"""

import os
import numpy as np
import pickle
import gzip
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


# ---------------------------------------------------------------------------
# 1. MNIST
# ---------------------------------------------------------------------------

def _read_idx_images(filepath):
    """Read IDX image file, return numpy array shape (n, rows*cols) in float64 [0,1]."""
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        assert magic == 2051, f"Bad image magic: {magic}"
        n = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(np.float64) / 255.0


def _read_idx_labels(filepath):
    """Read IDX label file, return numpy array shape (n,) of int64."""
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        assert magic == 2049, f"Bad label magic: {magic}"
        n = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_mnist():
    """Returns (X_train, y_train, X_test, y_test).

    X_train: (60000, 784) float64 in [0, 1]
    y_train: (60000,) int64  {0..9}
    """
    base = os.path.join(DATA_DIR, 'mnist')
    X_train = _read_idx_images(os.path.join(base, 'train-images-idx3-ubyte.gz'))
    y_train = _read_idx_labels(os.path.join(base, 'train-labels-idx1-ubyte.gz'))
    X_test = _read_idx_images(os.path.join(base, 't10k-images-idx3-ubyte.gz'))
    y_test = _read_idx_labels(os.path.join(base, 't10k-labels-idx1-ubyte.gz'))
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 2. CIFAR-10
# ---------------------------------------------------------------------------

def _unpickle(filepath):
    """Load a CIFAR-10 pickle batch file. Returns (X, y)."""
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    X = data_dict[b'data'].astype(np.float64) / 255.0
    y = np.array(data_dict[b'labels'], dtype=np.int64)
    return X, y


def load_cifar10():
    """Returns (X_train, y_train, X_test, y_test).

    X_train: (50000, 3072) float64 in [0, 1]
    y_train: (50000,) int64  {0..9}
    """
    base = os.path.join(DATA_DIR, 'cifar-10-python', 'cifar-10-batches-py')
    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        X, y = _unpickle(os.path.join(base, f'data_batch_{i}'))
        X_train_list.append(X)
        y_train_list.append(y)
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test, y_test = _unpickle(os.path.join(base, 'test_batch'))
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 3. RCV1
# ---------------------------------------------------------------------------

def load_rcv1():
    """Returns (X_train, y_train, X_test, y_test) — sparse CSR X.

    X_train shape ≈ (16193, 47236), labels in {-1.0, +1.0}.
    No official test set; uses 20% holdout.
    """
    filepath = os.path.join(DATA_DIR, 'rcv1', 'rcv1_train.binary', 'rcv1_train.binary')
    X, y = load_svmlight_file(filepath)
    y = y.astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 4. Covtype
# ---------------------------------------------------------------------------

def load_covtype():
    """Returns (X_train, y_train, X_test, y_test) — sparse CSR X.

    Binarization: class 2 → +1.0, all others → -1.0 (as per paper).
    Split 50/50.
    X_train shape ≈ (290506, 54).
    """
    filepath = os.path.join(DATA_DIR, 'covtype', 'covtype.libsvm.binary', 'covtype.libsvm.binary')
    X, y = load_svmlight_file(filepath)
    # Binarize: class 2 -> +1, rest -> -1
    y_bin = np.where(y == 2, 1.0, -1.0)
    # Split 50/50 as per paper
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.5, random_state=42
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Unified Interface
# ---------------------------------------------------------------------------

DATASETS = {
    'mnist': load_mnist,
    'cifar10': load_cifar10,
    'rcv1': load_rcv1,
    'covtype': load_covtype,
}


def load_dataset(name):
    """Load dataset by name. Returns (X_train, y_train, X_test, y_test).

    Parameters
    ----------
    name : str
        One of {'mnist', 'cifar10', 'rcv1', 'covtype'}.

    Returns
    -------
    X_train : np.ndarray or scipy.sparse.csr_matrix
    y_train : np.ndarray
    X_test  : np.ndarray or scipy.sparse.csr_matrix
    y_test  : np.ndarray
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Options: {list(DATASETS.keys())}")
    return DATASETS[name]()
