"""
plot_results.py — Reproduce Paper Figures (Figure 1 & 2)

Generates publication-quality plots from experiment results:
- Figure 1: Training loss residual P(w) - P(w*) vs. effective passes (2x2 grid)
- Figure 2: Test error rate vs. effective passes (2x2 grid)
- Individual high-resolution plots per dataset
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Plot styling to match paper
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 4,
})

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

# Color scheme matching paper convention
COLORS = {
    'svrg': '#1f77b4',       # Blue
    'sgd_const': '#ff7f0e',  # Orange
    'sgd_best': '#2ca02c',   # Green
    'sdca': '#d62728',       # Red (if available)
}

DATASETS = ['rcv1', 'covtype', 'mnist', 'cifar10']
TITLES = ['rcv1.binary', 'covtype.binary', 'MNIST', 'CIFAR-10']


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_results(dataset_name):
    """Load experiment results from JSON."""
    filepath = os.path.join(RESULTS_DIR, f'{dataset_name}_results.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_results():
    """Load results for all datasets."""
    return {name: load_results(name) for name in DATASETS}


# ---------------------------------------------------------------------------
# Figure 1: Training Loss Residual  (log scale y-axis)
# ---------------------------------------------------------------------------

def _safe_residuals(residuals):
    """Clip residuals to avoid log(0) or negative values on log scale."""
    return [max(r, 1e-16) for r in residuals]


def plot_figure1(results_dict):
    """Figure 1: Training loss residual P(w) - P(w*) for convex problems.

    2x2 grid, one subplot per dataset.
    Y-axis: log scale.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        ax = axes[idx]
        data = results_dict.get(dataset)

        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        # SVRG
        svrg = data['svrg']
        ax.semilogy(svrg['passes'], _safe_residuals(svrg['loss_residual']),
                    color=COLORS['svrg'], label='SVRG',
                    marker='o', markevery=5)

        # SGD Constant
        sgd_const = data['sgd_const']
        ax.semilogy(sgd_const['passes'], _safe_residuals(sgd_const['loss_residual']),
                    color=COLORS['sgd_const'], label='SGD (const $\\eta$)')

        # SGD Best
        sgd_best = data['sgd_best']
        ax.semilogy(sgd_best['passes'], _safe_residuals(sgd_best['loss_residual']),
                    color=COLORS['sgd_best'], label='SGD-best',
                    linestyle='--')

        # Formatting
        ax.set_xlabel('Number of effective passes')
        ax.set_ylabel('Training loss residual $P(w) - P(w^*)$')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set reasonable y-axis range
        all_residuals = (
            _safe_residuals(svrg['loss_residual']) +
            _safe_residuals(sgd_const['loss_residual']) +
            _safe_residuals(sgd_best['loss_residual'])
        )
        positive_res = [r for r in all_residuals if r > 0]
        if positive_res:
            min_res = min(positive_res)
            max_res = max(all_residuals)
            ax.set_ylim([max(1e-10, min_res * 0.5), max_res * 2])

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'figure1_training_loss.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filepath}")


# ---------------------------------------------------------------------------
# Figure 2: Test Error Rate  (linear scale y-axis)
# ---------------------------------------------------------------------------

def compute_test_error(w, X_test, y_test, multiclass=False):
    """Compute test error rate (percentage).

    Args:
        w: weight vector (d,) for binary or (d, K) for multi-class
        X_test: test features (n_test, d)
        y_test: test labels (n_test,)
        multiclass: multi-class flag

    Returns:
        error rate as percentage
    """
    if multiclass:
        logits = X_test @ w
        preds = np.argmax(logits, axis=1)
    else:
        scores = X_test @ w
        preds = np.sign(scores)

    errors = np.mean(preds != y_test)
    return errors * 100  # Convert to percentage


def plot_figure2(results_dict, X_test_dict, y_test_dict, multiclass_dict):
    """Figure 2: Test error rate vs effective passes.

    2x2 grid, one subplot per dataset.
    Y-axis: linear scale (percentage).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        ax = axes[idx]
        data = results_dict.get(dataset)

        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        X_test = X_test_dict[dataset]
        y_test = y_test_dict[dataset]
        multiclass = multiclass_dict[dataset]

        # SVRG
        svrg = data['svrg']
        if svrg.get('test_error'):
            ax.plot(svrg['passes'], svrg['test_error'],
                    color=COLORS['svrg'], label='SVRG',
                    marker='o', markevery=5)

        # SGD Constant
        sgd_const = data['sgd_const']
        if sgd_const.get('test_error'):
            ax.plot(sgd_const['passes'], sgd_const['test_error'],
                    color=COLORS['sgd_const'], label='SGD (const $\\eta$)')

        # SGD Best
        sgd_best = data['sgd_best']
        if sgd_best.get('test_error'):
            ax.plot(sgd_best['passes'], sgd_best['test_error'],
                    color=COLORS['sgd_best'], label='SGD-best',
                    linestyle='--')

        # Formatting
        ax.set_xlabel('Number of effective passes')
        ax.set_ylabel('Test error rate (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'figure2_test_error.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filepath}")


# ---------------------------------------------------------------------------
# Individual Dataset Plots (Detailed View)
# ---------------------------------------------------------------------------

def plot_individual_figures(results_dict):
    """Create individual high-resolution plots for each dataset."""
    for dataset, title in zip(DATASETS, TITLES):
        fig, ax = plt.subplots(figsize=(8, 6))
        data = results_dict.get(dataset)

        if data is None:
            continue

        # SVRG
        svrg = data['svrg']
        ax.semilogy(svrg['passes'], _safe_residuals(svrg['loss_residual']),
                    color=COLORS['svrg'], label='SVRG',
                    marker='o', markersize=6, markevery=3, linewidth=2.5)

        # SGD Constant
        sgd_const = data['sgd_const']
        ax.semilogy(sgd_const['passes'], _safe_residuals(sgd_const['loss_residual']),
                    color=COLORS['sgd_const'], label='SGD (const $\\eta$)',
                    linewidth=1.5, alpha=0.8)

        # SGD Best
        sgd_best = data['sgd_best']
        ax.semilogy(sgd_best['passes'], _safe_residuals(sgd_best['loss_residual']),
                    color=COLORS['sgd_best'], label='SGD-best',
                    linestyle='--', linewidth=2)

        ax.set_xlabel('Number of effective passes', fontsize=14)
        ax.set_ylabel('Training loss residual $P(w) - P(w^*)$', fontsize=14)
        ax.set_title(f'{title} — Training Loss Residual', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(FIGURES_DIR, f'figure_{dataset}_loss.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate all figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading results...")
    results = load_all_results()

    print("\nGenerating Figure 1: Training Loss Residual...")
    plot_figure1(results)

    print("\nGenerating Figure 2: Test Error Rate...")
    print("  (Requires intermediate weights — run enhanced train.py first)")
    # Uncomment when test_error data is available:
    # X_test_dict, y_test_dict, multiclass_dict = load_test_data()
    # plot_figure2(results, X_test_dict, y_test_dict, multiclass_dict)

    print("\nGenerating individual figures...")
    plot_individual_figures(results)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
