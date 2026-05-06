# Step 07: `plot_results.py` — Reproduce Paper Figures

## Objective

Generate publication-quality plots that reproduce Figures 1, 2, and 3 from the NIPS 2013 paper:
- **Figure 1**: Training loss residual $P(w) - P(w^*)$ vs. effective passes for **convex** problems (Logistic Regression on all 4 datasets)
- **Figure 2**: Test error rate vs. effective passes for convex problems
- **Figure 3**: Training loss residual for **non-convex** problems (Neural Network on MNIST, CIFAR-10) — *future work, not in current scope*

## Figure Specifications (from Paper)

### Figure 1: Training Loss Residual (Convex)

```
Format: 2×2 grid (4 subplots)
Each subplot:
  - X-axis: "Number of effective passes" (linear scale, 0 to ~90)
  - Y-axis: "Training loss residual P(w) - P(w*)" (log scale, 10^-6 to 10^0)
  - Lines: SVRG, SGD (const η), SGD-best, SDCA
  - Title: Dataset name
```

### Figure 2: Test Error Rate (Convex)

```
Format: 2×2 grid (4 subplots)
Each subplot:
  - X-axis: "Number of effective passes" (linear scale)
  - Y-axis: "Test error rate" (linear scale, percentage)
  - Lines: SVRG, SGD (const η), SGD-best, SDCA
  - Title: Dataset name
```

## File Structure

```
experiment/
├── plot_results.py      ← THIS FILE
├── results/             ← Input: JSON results from train.py
└── figures/             ← Output: PNG figures
```

## Detailed Implementation

### 1. Imports & Setup

```python
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
```

### 2. Data Loading

```python
def load_results(dataset_name):
    """Load experiment results from JSON."""
    filepath = os.path.join(RESULTS_DIR, f'{dataset_name}_results.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_results():
    """Load results for all datasets."""
    datasets = ['mnist', 'cifar10', 'rcv1', 'covtype']
    return {name: load_results(name) for name in datasets}
```

### 3. Figure 1: Training Loss Residual

```python
def plot_figure1(results_dict):
    """Figure 1: Training loss residual P(w) - P(w*) for convex problems.
    
    2×2 grid, one subplot per dataset.
    Y-axis: log scale.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    datasets = ['rcv1', 'covtype', 'mnist', 'cifar10']
    titles = ['rcv1.binary', 'covtype.binary', 'MNIST', 'CIFAR-10']
    
    colors = {
        'svrg': '#1f77b4',       # Blue
        'sgd_const': '#ff7f0e',  # Orange
        'sgd_best': '#2ca02c',   # Green
        'sdca': '#d62728',       # Red (if available)
    }
    
    for idx, (dataset, title) in enumerate(zip(datasets, titles)):
        ax = axes[idx]
        data = results_dict.get(dataset)
        
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # SVRG
        svrg = data['svrg']
        ax.semilogy(svrg['passes'], svrg['loss_residual'],
                    color=colors['svrg'], label='SVRG', marker='o', markevery=5)
        
        # SGD Constant
        sgd_const = data['sgd_const']
        ax.semilogy(sgd_const['passes'], sgd_const['loss_residual'],
                    color=colors['sgd_const'], label='SGD (const $\eta$)')
        
        # SGD Best
        sgd_best = data['sgd_best']
        ax.semilogy(sgd_best['passes'], sgd_best['loss_residual'],
                    color=colors['sgd_best'], label='SGD-best', linestyle='--')
        
        # Formatting
        ax.set_xlabel('Number of effective passes')
        ax.set_ylabel('Training loss residual $P(w) - P(w^*)$')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ensure y-axis shows reasonable range
        all_residuals = (
            svrg['loss_residual'] + 
            sgd_const['loss_residual'] + 
            sgd_best['loss_residual']
        )
        min_res = min([r for r in all_residuals if r > 0])
        max_res = max(all_residuals)
        ax.set_ylim([max(1e-8, min_res * 0.5), max_res * 2])
    
    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, 'figure1_training_loss.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {filepath}")
```

### 4. Figure 2: Test Error Rate

```python
def compute_test_error(w, X_test, y_test, multiclass=False):
    """Compute test error rate (percentage)."""
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
    
    2×2 grid, one subplot per dataset.
    Y-axis: linear scale (percentage).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    datasets = ['rcv1', 'covtype', 'mnist', 'cifar10']
    titles = ['rcv1.binary', 'covtype.binary', 'MNIST', 'CIFAR-10']
    
    colors = {
        'svrg': '#1f77b4',
        'sgd_const': '#ff7f0e',
        'sgd_best': '#2ca02c',
    }
    
    for idx, (dataset, title) in enumerate(zip(datasets, titles)):
        ax = axes[idx]
        data = results_dict.get(dataset)
        
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        X_test = X_test_dict[dataset]
        y_test = y_test_dict[dataset]
        multiclass = multiclass_dict[dataset]
        
        # Note: We need to re-run training with weight logging to get test errors.
        # For now, this is a placeholder showing the structure.
        # The actual implementation requires saving intermediate weights during training.
        
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
```

### 5. Figure 3: Individual Dataset Plots (Detailed View)

```python
def plot_individual_figures(results_dict):
    """Create individual high-resolution plots for each dataset.
    
    These are more detailed versions for analysis.
    """
    datasets = ['rcv1', 'covtype', 'mnist', 'cifar10']
    titles = ['rcv1.binary', 'covtype.binary', 'MNIST', 'CIFAR-10']
    
    for dataset, title in zip(datasets, titles):
        fig, ax = plt.subplots(figsize=(8, 6))
        data = results_dict.get(dataset)
        
        if data is None:
            continue
        
        # SVRG
        svrg = data['svrg']
        ax.semilogy(svrg['passes'], svrg['loss_residual'],
                    color='#1f77b4', label='SVRG', 
                    marker='o', markersize=6, markevery=3, linewidth=2.5)
        
        # SGD Constant
        sgd_const = data['sgd_const']
        ax.semilogy(sgd_const['passes'], sgd_const['loss_residual'],
                    color='#ff7f0e', label='SGD (const $\eta$)', 
                    linewidth=1.5, alpha=0.8)
        
        # SGD Best
        sgd_best = data['sgd_best']
        ax.semilogy(sgd_best['passes'], sgd_best['loss_residual'],
                    color='#2ca02c', label='SGD-best', 
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
```

### 6. Main Function

```python
def main():
    """Generate all figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading results...")
    results = load_all_results()
    
    print("\nGenerating Figure 1: Training Loss Residual...")
    plot_figure1(results)
    
    print("\nGenerating Figure 2: Test Error Rate...")
    # Note: Requires intermediate weights — see enhancement section
    # plot_figure2(results, ...)
    
    print("\nGenerating individual figures...")
    plot_individual_figures(results)
    
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
```

## Enhancement: Logging Intermediate Weights

To plot test error rates (Figure 2), modify `train.py` to save intermediate weights:

```python
# In train.py, add to the logging section:
results['svrg']['weights'].append(w_svrg.copy())  # Save weight snapshot
results['svrg']['test_error'].append(compute_test_error(w_svrg, X_test, y_test, multiclass))
```

Then in `plot_results.py`, compute test errors from saved weights.

## Constraints & Configuration

### Plot Styling

| Element | Specification |
|---------|--------------|
| Figure size | 12×10 for 2×2 grid, 8×6 for individual |
| DPI | 150 for grid, 200 for individual |
| Y-axis scale | Log (loss residual), Linear (test error) |
| X-axis | Linear, "Number of effective passes" |
| Colors | SVRG=blue, SGD-const=orange, SGD-best=green, SDCA=red |
| Grid | Light grid lines (alpha=0.3) |
| Legend | Include all methods |

### File Output

```
figures/
├── figure1_training_loss.png    ← 2×2 grid, all datasets
├── figure2_test_error.png       ← 2×2 grid, all datasets
├── figure_rcv1_loss.png         ← Individual
├── figure_covtype_loss.png      ← Individual
├── figure_mnist_loss.png        ← Individual
└── figure_cifar10_loss.png      ← Individual
```

## Verification Checklist

- [ ] Figure 1 shows SVRG converging fastest (steepest line on log scale)
- [ ] SGD-constant plateaus (variance floor)
- [ ] SGD-best is between SVRG and SGD-constant
- [ ] All 4 datasets plotted correctly
- [ ] Y-axis is log scale, X-axis is effective passes
- [ ] Loss residuals are positive (no negative values on log scale)
- [ ] Figure titles match paper convention
- [ ] Legend clearly identifies each method
- [ ] Figures saved as high-resolution PNG

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Log scale with zero/negative values | Clip residuals: `max(residual, 1e-16)` |
| Lines look jagged/noisy | SVRG should be smooth; if SGD is noisy, that's expected |
| Missing SDCA baseline | SDCA is optional; plot without it if not implemented |
| Test error requires intermediate weights | Modify train.py to save weight checkpoints |
| Y-axis range too wide | Set `ax.set_ylim()` based on data range |
| Figure looks different from paper | Check: log scale? effective passes? same datasets? |
