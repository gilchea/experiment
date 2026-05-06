"""
plot_results.py — Reproduce Paper Figures (Figure 1 & 2)

Generates publication-quality plots from experiment results:
- Figure 1: Training loss residual P(w) - P(w*) vs. effective passes (2x2 grid)
- Figure 2: Test error rate vs. effective passes (2x2 grid)
- Figure 3: Gradient variance vs. effective passes (2x2 grid)
- Individual high-resolution plots per dataset
- NN (Non-convex) Figures: Training loss, test error, gradient variance
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
    'sdca': '#d62728',       # Red
    'sag': '#9467bd',        # Purple
}

DATASETS = ['rcv1', 'covtype', 'mnist', 'cifar10']
TITLES = ['rcv1.binary', 'covtype.binary', 'MNIST', 'CIFAR-10']

# Algorithms to plot (in order)
ALGORITHMS = ['svrg', 'sgd_const', 'sgd_best', 'sdca', 'sag']
ALGORITHM_LABELS = {
    'svrg': 'SVRG',
    'sgd_const': 'SGD (const $\\eta$)',
    'sgd_best': 'SGD-best',
    'sdca': 'SDCA',
    'sag': 'SAG',
}
ALGORITHM_STYLES = {
    'svrg': {'marker': 'o', 'markevery': 5, 'linestyle': '-'},
    'sgd_const': {'marker': None, 'linestyle': '-'},
    'sgd_best': {'marker': None, 'linestyle': '--'},
    'sdca': {'marker': 's', 'markevery': 5, 'linestyle': '-'},
    'sag': {'marker': '^', 'markevery': 5, 'linestyle': ':'},
}


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
# Helpers
# ---------------------------------------------------------------------------

def _safe_residuals(residuals):
    """Clip residuals to avoid log(0) or negative values on log scale."""
    return [max(r, 1e-16) for r in residuals]


def _plot_algorithm(ax, data, algo_key, passes_key='passes',
                    loss_key='loss_residual', log_scale=True):
    """Plot a single algorithm's data on an axis.

    Args:
        ax: matplotlib axis
        data: results dict for one dataset
        algo_key: algorithm key (e.g., 'svrg')
        passes_key: key for x-axis data
        loss_key: key for y-axis data
        log_scale: use semilogy if True
    """
    algo_data = data.get(algo_key)
    if algo_data is None:
        return

    x = algo_data.get(passes_key, [])
    y = algo_data.get(loss_key, [])

    if not x or not y:
        return

    # Filter out None values
    valid = [(xx, yy) for xx, yy in zip(x, y) if yy is not None]
    if not valid:
        return
    xv, yv = zip(*valid)

    style = ALGORITHM_STYLES.get(algo_key, {})
    color = COLORS.get(algo_key, '#000000')
    label = ALGORITHM_LABELS.get(algo_key, algo_key)

    if log_scale:
        yv_safe = _safe_residuals(list(yv))
        ax.semilogy(list(xv), yv_safe, color=color, label=label, **style)
    else:
        ax.plot(list(xv), list(yv), color=color, label=label, **style)


# ---------------------------------------------------------------------------
# Figure 1: Training Loss Residual  (log scale y-axis)
# ---------------------------------------------------------------------------

def plot_figure1(results_dict):
    """Figure 1: Training loss residual P(w) - P(w*) for convex problems.

    2x2 grid, one subplot per dataset.
    Y-axis: log scale.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        ax = axes[idx]
        data = results_dict.get(dataset)

        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Plot all algorithms
        for algo in ALGORITHMS:
            _plot_algorithm(ax, data, algo, log_scale=True)

        # Formatting
        ax.set_xlabel('Number of effective passes')
        ax.set_ylabel('Training loss residual $P(w) - P(w^*)$')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set reasonable y-axis range
        all_residuals = []
        for algo in ALGORITHMS:
            algo_data = data.get(algo)
            if algo_data:
                residuals = algo_data.get('loss_residual', [])
                all_residuals.extend(_safe_residuals(
                    [r for r in residuals if r is not None]))
        positive_res = [r for r in all_residuals if r > 0]
        if positive_res:
            min_res = min(positive_res)
            max_res = max(all_residuals)
            ax.set_ylim([max(1e-12, min_res * 0.5), max_res * 2])

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'figure1_training_loss.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filepath}")


# ---------------------------------------------------------------------------
# Figure 2: Test Error Rate  (linear scale y-axis)
# ---------------------------------------------------------------------------

def plot_figure2(results_dict):
    """Figure 2: Test error rate vs effective passes.

    2x2 grid, one subplot per dataset.
    Y-axis: linear scale (percentage).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        ax = axes[idx]
        data = results_dict.get(dataset)

        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Plot all algorithms with test_error
        for algo in ALGORITHMS:
            _plot_algorithm(ax, data, algo, loss_key='test_error',
                            log_scale=False)

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
# Figure 3: Gradient Variance  (log scale y-axis)
# ---------------------------------------------------------------------------

def plot_figure3(results_dict):
    """Figure 3: Gradient variance vs effective passes for SVRG.

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

        # SVRG variance
        svrg = data.get('svrg', {})
        passes = svrg.get('passes', [])
        variances = svrg.get('grad_variance', [])

        # Filter out None values
        valid = [(p, v) for p, v in zip(passes, variances) if v is not None]
        if valid:
            pv, vv = zip(*valid)
            ax.semilogy(list(pv), _safe_residuals(list(vv)),
                        color=COLORS['svrg'], label='SVRG gradient variance',
                        marker='o', markevery=5, linewidth=2)

        # Formatting
        ax.set_xlabel('Number of effective passes')
        ax.set_ylabel('Gradient variance $\\mathbb{E}\\|v - \\mu\\|^2$')
        ax.set_title(f'{title} — Gradient Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'figure3_gradient_variance.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filepath}")


# ---------------------------------------------------------------------------
# Individual Dataset Plots (Detailed View)
# ---------------------------------------------------------------------------

def plot_individual_figures(results_dict):
    """Create individual high-resolution plots for each dataset."""
    for dataset, title in zip(DATASETS, TITLES):
        # Loss residual plot
        fig, ax = plt.subplots(figsize=(8, 6))
        data = results_dict.get(dataset)

        if data is None:
            plt.close()
            continue

        for algo in ALGORITHMS:
            _plot_algorithm(ax, data, algo, log_scale=True)

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

        # Test error plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for algo in ALGORITHMS:
            _plot_algorithm(ax, data, algo, loss_key='test_error',
                            log_scale=False)

        ax.set_xlabel('Number of effective passes', fontsize=14)
        ax.set_ylabel('Test error rate (%)', fontsize=14)
        ax.set_title(f'{title} — Test Error Rate', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(FIGURES_DIR, f'figure_{dataset}_test_error.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {filepath}")


# ---------------------------------------------------------------------------
# Neural Network (Non-convex) Plotting
# ---------------------------------------------------------------------------

NN_DATASETS = ['mnist_nn', 'cifar10_nn']
NN_TITLES = ['MNIST (NN)', 'CIFAR-10 (NN)']

# NN algorithms (only SVRG and SGD — no SDCA/SAG for NN)
NN_ALGORITHMS = ['svrg', 'sgd_const', 'sgd_best']
NN_ALGORITHM_LABELS = {
    'svrg': 'SVRG',
    'sgd_const': 'SGD (const $\\eta$)',
    'sgd_best': 'SGD-best',
}
NN_ALGORITHM_STYLES = {
    'svrg': {'marker': 'o', 'markevery': 5, 'linestyle': '-'},
    'sgd_const': {'marker': None, 'linestyle': '-'},
    'sgd_best': {'marker': None, 'linestyle': '--'},
}


def _plot_nn_algorithm(ax, data, algo_key, passes_key='passes',
                       loss_key='loss', log_scale=True):
    """Plot a single NN algorithm's data on an axis.

    NN results use 'loss' key instead of 'loss_residual' (no P*).
    """
    algo_data = data.get(algo_key)
    if algo_data is None:
        return

    x = algo_data.get(passes_key, [])
    y = algo_data.get(loss_key, [])

    if not x or not y:
        return

    # Filter out None values
    valid = [(xx, yy) for xx, yy in zip(x, y) if yy is not None]
    if not valid:
        return
    xv, yv = zip(*valid)

    style = NN_ALGORITHM_STYLES.get(algo_key, {})
    color = COLORS.get(algo_key, '#000000')
    label = NN_ALGORITHM_LABELS.get(algo_key, algo_key)

    if log_scale:
        yv_safe = _safe_residuals(list(yv))
        ax.semilogy(list(xv), yv_safe, color=color, label=label, **style)
    else:
        ax.plot(list(xv), list(yv), color=color, label=label, **style)


def plot_nn_figure1(results_dict):
    """Figure NN-1: Training loss vs effective passes for NN (non-convex).

    1x2 grid, one subplot per dataset.
    Y-axis: log scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (dataset, title) in enumerate(zip(NN_DATASETS, NN_TITLES)):
        ax = axes[idx]
        data = results_dict.get(dataset)

        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        for algo in NN_ALGORITHMS:
            _plot_nn_algorithm(ax, data, algo, log_scale=True)

        ax.set_xlabel('Number of effective passes')
        ax.set_ylabel('Training loss')
        ax.set_title(f'{title} — Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'figure_nn1_training_loss.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filepath}")


def plot_nn_figure2(results_dict):
    """Figure NN-2: Test error rate vs effective passes for NN.

    1x2 grid, one subplot per dataset.
    Y-axis: linear scale (percentage).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (dataset, title) in enumerate(zip(NN_DATASETS, NN_TITLES)):
        ax = axes[idx]
        data = results_dict.get(dataset)

        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        for algo in NN_ALGORITHMS:
            _plot_nn_algorithm(ax, data, algo, loss_key='test_error',
                               log_scale=False)

        ax.set_xlabel('Number of effective passes')
        ax.set_ylabel('Test error rate (%)')
        ax.set_title(f'{title} — Test Error Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'figure_nn2_test_error.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filepath}")


def plot_nn_figure3(results_dict):
    """Figure NN-3: Gradient variance vs effective passes for SVRG (NN).

    1x2 grid, one subplot per dataset.
    Y-axis: log scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (dataset, title) in enumerate(zip(NN_DATASETS, NN_TITLES)):
        ax = axes[idx]
        data = results_dict.get(dataset)

        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        svrg = data.get('svrg', {})
        passes = svrg.get('passes', [])
        variances = svrg.get('grad_variance', [])

        valid = [(p, v) for p, v in zip(passes, variances) if v is not None]
        if valid:
            pv, vv = zip(*valid)
            ax.semilogy(list(pv), _safe_residuals(list(vv)),
                        color=COLORS['svrg'], label='SVRG gradient variance',
                        marker='o', markevery=5, linewidth=2)

        ax.set_xlabel('Number of effective passes')
        ax.set_ylabel('Gradient variance $\\mathbb{E}\\|v - \\mu\\|^2$')
        ax.set_title(f'{title} — Gradient Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'figure_nn3_gradient_variance.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filepath}")


def load_nn_results():
    """Load NN experiment results."""
    nn_results = {}
    for name in NN_DATASETS:
        try:
            nn_results[name] = load_results(name)
        except FileNotFoundError:
            print(f"  Warning: NN results for {name} not found, skipping")
    return nn_results


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
    plot_figure2(results)

    print("\nGenerating Figure 3: Gradient Variance...")
    plot_figure3(results)

    print("\nGenerating individual figures...")
    plot_individual_figures(results)

    # ── NN (Non-convex) Figures ──
    print("\n" + "="*60)
    print("Neural Network (Non-convex) Figures")
    print("="*60)

    print("\nLoading NN results...")
    nn_results = load_nn_results()

    if nn_results:
        print("\nGenerating NN Figure 1: Training Loss...")
        plot_nn_figure1(nn_results)

        print("\nGenerating NN Figure 2: Test Error Rate...")
        plot_nn_figure2(nn_results)

        print("\nGenerating NN Figure 3: Gradient Variance...")
        plot_nn_figure3(nn_results)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
