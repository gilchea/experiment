# Step 01: `utils/data_loader.py` — Load & Preprocess 4 Datasets

## Objective

Implement a unified data loader that loads all 4 datasets (MNIST, CIFAR-10, RCV1, Covtype) into memory with consistent NumPy/SciPy formats, ready for training.

## Constraints & Configuration

| Dataset | Type | n (samples) | d (features) | Labels | Preprocessing |
|---------|------|-------------|--------------|--------|---------------|
| MNIST | Dense (uint8) | 60,000 | 784 | {0..9} | `/255` → float64 in [0,1] |
| CIFAR-10 | Dense (uint8) | 50,000 | 3072 | {0..9} | `/255` → float64 in [0,1] |
| RCV1 | Sparse (LIBSVM) | 20,242 | 47,236 | {-1, +1} | Keep sparse, no scaling |
| Covtype | Sparse (LIBSVM) | 581,012 | 54 | {-1, +1} | Binarize (class 2→+1, rest→-1), split 50/50 |

## File Structure

```
experiment/
└── utils/
    └── data_loader.py      ← THIS FILE
```

## Detailed Implementation

### 1. Imports & Constants

```python
import os
import numpy as np
import pickle
import gzip
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
```

### 2. Function: `load_mnist()`

**Source files** (from `data/mnist/`):
- `train-images-idx3-ubyte.gz` (or unzipped)
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

**Binary format** (IDX):
- First 4 bytes: magic number
- Next 4 bytes: number of items
- For images: next 8 bytes = rows, cols
- For labels: next 4 bytes = number of items
- Remaining: raw pixel/label data

```python
def _read_idx_images(filepath):
    """Read IDX image file, return numpy array shape (n, rows*cols)"""
    with gzip.open(filepath, 'rb') if filepath.endswith('.gz') else open(filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        assert magic == 2051, f"Bad image magic: {magic}"
        n = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(np.float64) / 255.0

def _read_idx_labels(filepath):
    """Read IDX label file, return numpy array shape (n,)"""
    with gzip.open(filepath, 'rb') if filepath.endswith('.gz') else open(filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        assert magic == 2049, f"Bad label magic: {magic}"
        n = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)

def load_mnist():
    """Returns (X_train, y_train, X_test, y_test)"""
    base = os.path.join(DATA_DIR, 'mnist')
    X_train = _read_idx_images(os.path.join(base, 'train-images-idx3-ubyte.gz'))
    y_train = _read_idx_labels(os.path.join(base, 'train-labels-idx1-ubyte.gz'))
    X_test = _read_idx_images(os.path.join(base, 't10k-images-idx3-ubyte.gz'))
    y_test = _read_idx_labels(os.path.join(base, 't10k-labels-idx1-ubyte.gz'))
    return X_train, y_train, X_test, y_test
```

### 3. Function: `load_cifar10()`

**Source**: `data/cifar-10-python/cifar-10-batches-py/`
- `data_batch_1` through `data_batch_5` (training)
- `test_batch` (testing)

**Pickle format** (Python 2 pickle, `encoding='bytes'`):
- `b'data'`: numpy array (10000, 3072) — uint8, row-major (R, G, B)
- `b'labels'`: list of 10000 ints

```python
def _unpickle(filepath):
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    X = data_dict[b'data'].astype(np.float64) / 255.0
    y = np.array(data_dict[b'labels'], dtype=np.int64)
    return X, y

def load_cifar10():
    """Returns (X_train, y_train, X_test, y_test)"""
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
```

### 4. Function: `load_rcv1()`

**Source**: `data/rcv1/rcv1_train.binary/rcv1_train.binary`

**Format**: LIBSVM sparse (each line: `label idx:val idx:val ...`)

```python
def load_rcv1():
    """Returns (X_train, y_train, X_test, y_test) — sparse X"""
    filepath = os.path.join(DATA_DIR, 'rcv1', 'rcv1_train.binary', 'rcv1_train.binary')
    X, y = load_svmlight_file(filepath)
    y = y.astype(np.float64)
    # RCV1 labels are already {-1, +1}
    # No test set provided — use 20% holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, y_train, X_test, y_test
```

### 5. Function: `load_covtype()`

**Source**: `data/covtype/covtype.libsvm.binary/covtype.libsvm.binary`

**Binarization**: Class 2 → +1, all others → -1 (as per paper)

```python
def load_covtype():
    """Returns (X_train, y_train, X_test, y_test) — sparse X"""
    filepath = os.path.join(DATA_DIR, 'covtype', 'covtype.libsvm.binary', 'covtype.libsvm.binary')
    X, y = load_svmlight_file(filepath)
    # Binarize: class 2 -> +1, rest -> -1
    y_bin = np.where(y == 2, 1.0, -1.0)
    # Split 50/50 as per paper
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.5, random_state=42
    )
    return X_train, y_train, X_test, y_test
```

### 6. Unified Interface

```python
DATASETS = {
    'mnist': load_mnist,
    'cifar10': load_cifar10,
    'rcv1': load_rcv1,
    'covtype': load_covtype,
}

def load_dataset(name):
    """Load dataset by name. Returns (X_train, y_train, X_test, y_test)."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Options: {list(DATASETS.keys())}")
    return DATASETS[name]()
```

## Verification Checklist

- [ ] `load_mnist()`: X_train shape (60000, 784), values in [0, 1], y_train shape (60000,)
- [ ] `load_cifar10()`: X_train shape (50000, 3072), values in [0, 1], y_train shape (50000,)
- [ ] `load_rcv1()`: X_train is sparse (CSR), shape ≈ (16193, 47236), labels in {-1, +1}
- [ ] `load_covtype()`: X_train shape ≈ (290506, 54), labels in {-1, +1}, binary class 2→+1
- [ ] All return tuples of 4 elements: (X_train, y_train, X_test, y_test)

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| CIFAR-10 pickle is Python 2 format | Use `encoding='bytes'` in `pickle.load()` |
| MNIST IDX byte order is big-endian | Use `int.from_bytes(..., 'big')` |
| RCV1/Covtype file not found | Verify `DATA_DIR` path; check `.gitignore` excludes `data/` |
| Covtype has 7 classes, need binary | Explicitly binarize: `np.where(y == 2, 1.0, -1.0)` |
| Memory for Covtype (581k samples) | Use sparse CSR format from `load_svmlight_file` |
