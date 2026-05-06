# SVRG Experiment — Reproducing NIPS 2013 Paper

> **Paper**: *Accelerating Stochastic Gradient Descent using Predictive Variance Reduction*
> — Rie Johnson & Tong Zhang, NIPS 2013

## Overview

This project reproduces the key experiments from the SVRG paper, demonstrating how **Stochastic Variance Reduced Gradient** (SVRG) achieves linear convergence for both convex (L2-regularized Logistic Regression) and non-convex (Neural Network) problems.

## Project Structure

```
experiment/
├── utils/
│   └── data_loader.py       # Load & preprocess 4 datasets
├── models/
│   └── logistic.py           # L2-logistic regression (binary + multi-class)
├── optimizers/
│   ├── sgd.py                # SGD baselines (constant + decaying lr)
│   └── svrg.py               # SVRG Algorithm 1 (core contribution)
├── config.py                 # Centralized hyperparameters
├── compute_optimal.py        # Estimate P(w*) via full GD
├── train.py                  # Main training runner
├── plot_results.py           # Reproduce paper figures
├── results/                  # Output: JSON result files
├── figures/                  # Output: PNG figures
├── data/                     # Dataset files (gitignored)
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Type | Samples | Features | Task |
|---------|------|---------|----------|------|
| MNIST | Dense | 60,000 | 784 | Multi-class (10 digits) |
| CIFAR-10 | Dense | 50,000 | 3,072 | Multi-class (10 objects) |
| RCV1 | Sparse | 20,242 | 47,236 | Binary classification |
| Covtype | Sparse | 581,012 | 54 | Binary classification |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

Place datasets in the `data/` directory:

```
data/
├── mnist/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── cifar-10-python/
│   └── cifar-10-batches-py/
│       ├── data_batch_1 .. 5
│       └── test_batch
├── rcv1/
│   └── rcv1_train.binary/
│       └── rcv1_train.binary
└── covtype/
    └── covtype.libsvm.binary/
        └── covtype.libsvm.binary
```

## Usage

Run the experiments in order:

```bash
# Step 1: Estimate optimal loss P(w*) via full GD
python compute_optimal.py

# Step 2: Run all experiments (SVRG + SGD baselines)
python train.py

# Step 3: Generate figures
python plot_results.py
```

## Algorithms

### SVRG (Stochastic Variance Reduced Gradient)

SVRG uses a double-loop structure to reduce gradient variance:

1. **Outer loop**: Compute full gradient $\tilde{\mu} = \nabla P(\tilde{w})$ at snapshot $\tilde{w}$
2. **Inner loop**: Perform $m$ updates with variance-reduced gradient:
   $$w_t = w_{t-1} - \eta(\nabla\psi_{i_t}(w_{t-1}) - \nabla\psi_{i_t}(\tilde{w}) + \tilde{\mu})$$

### Baselines

- **SGD (constant η)**: Fixed learning rate — shows variance floor
- **SGD-best**: Decaying learning rate $\eta_t = \eta_0(1 + a\eta_0 t)^{-1}$ — carefully tuned

## Hyperparameters

| Dataset | λ | SVRG η | m | Warm-start |
|---------|---|--------|---|------------|
| MNIST | 10⁻⁴ | 0.025 | 2n | 1 epoch SGD |
| CIFAR-10 | 10⁻³ | 0.01 | 2n | 1 epoch SGD |
| RCV1 | 10⁻⁵ | 0.01 | 2n | 1 epoch SGD |
| Covtype | 10⁻⁵ | 0.001 | 2n | 1 epoch SGD |

## Key Metrics

- **Loss residual**: $P(w) - P(w^*)$ on log scale (primary metric)
- **Effective passes**: Total gradient evaluations / n (fair comparison)
- **Test error rate**: Classification error on held-out data

## Expected Results

- SVRG converges linearly (straight line on log scale)
- SGD-constant plateaus due to non-vanishing variance
- SGD-best converges but slower than SVRG
- SVRG matches or exceeds SDCA performance

## References

- Johnson, R., & Zhang, T. (2013). *Accelerating Stochastic Gradient Descent using Predictive Variance Reduction*. NIPS 2013.
- Johnson, R., & Zhang, T. (2020). *Accelerating Stochastic Gradient Descent using Predictive Variance Reduction*. Mathematical Programming.
