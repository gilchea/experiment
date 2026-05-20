"""
plot_mnist_convex.py — Trực quan hóa kết quả cho MNIST Convex
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'mnist_convex_results.json')

plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# Cấu hình hiển thị
STYLES = {
    'svrg':       {'label': 'SVRG', 'color': '#d62728', 'marker': 'o', 'markevery': 5, 'linestyle': '-'},
    'sgd_0.001':  {'label': 'SGD (lr=0.001)', 'color': '#1f77b4', 'marker': None, 'linestyle': '-'},
    'sgd_0.0025': {'label': 'SGD (lr=0.0025)', 'color': '#ff7f0e', 'marker': None, 'linestyle': '-'},
    'sgd_0.005':  {'label': 'SGD (lr=0.005)', 'color': '#2ca02c', 'marker': None, 'linestyle': '-'},
    'sgd_best':   {'label': 'SGD-best', 'color': '#9467bd', 'marker': None, 'linestyle': '--'},
    # 'sdca':       {'label': 'SDCA', 'color': '#8c564b', 'marker': 's', 'markevery': 5, 'linestyle': '-'},
}

KEYS = ['svrg', 'sgd_0.001', 'sgd_0.0025', 'sgd_0.005', 'sgd_best']#, 'sdca']

def _safe_residuals(residuals):
    return [max(r, 1e-16) for r in residuals]

def plot_loss_residual(data):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for key in KEYS:
        if key not in data:
            continue
        algo_data = data[key]
        passes = algo_data.get('passes', [])
        residual = algo_data.get('loss_residual', [])
        
        valid = [(p, r) for p, r in zip(passes, residual) if r is not None]
        if not valid:
            continue
        pv, rv = zip(*valid)
        
        style = STYLES[key]
        ax.semilogy(list(pv), _safe_residuals(list(rv)), **style)
        
    ax.set_xlabel('Number of effective passes')
    ax.set_ylabel('Training loss residual $P(w) - P(w^*)$')
    ax.set_title('MNIST Convex: Training Loss Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, 'mnist_convex_loss.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved loss plot to: {os.path.basename(out_path)}")

def plot_variance(data):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for key in KEYS:
        if key not in data:
            continue
        algo_data = data[key]
        passes = algo_data.get('passes', [])
        var = algo_data.get('grad_variance', [])
        
        # Chỉ vẽ các điểm có variance
        valid = [(p, v) for p, v in zip(passes, var) if v is not None and v > 0]
        if not valid:
            continue
        pv, vv = zip(*valid)
        
        style = STYLES[key]
        ax.semilogy(list(pv), list(vv), **style)
        
    ax.set_xlabel('Number of effective passes')
    ax.set_ylabel('Gradient variance $\\mathbb{E}\\|\\nabla\\psi_i(w) - \\nabla P(w)\\|^2$')
    ax.set_title('MNIST Convex: Gradient Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_path = os.path.join(FIGURES_DIR, 'mnist_convex_variance.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved variance plot to: {os.path.basename(out_path)}")

def plot_training_loss(data):
    """Plot raw training loss P(w) instead of residual."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for key in KEYS:
        if key not in data:
            continue
        algo_data = data[key]
        passes = algo_data.get('passes', [])
        # Dùng training_loss thay vì loss_residual
        loss = algo_data.get('training_loss', [])
        
        valid = [(p, l) for p, l in zip(passes, loss) if l is not None]
        if not valid:
            continue
        pv, lv = zip(*valid)
        
        style = STYLES[key]
        ax.plot(list(pv), list(lv), **style)
        
    ax.set_xlabel('Number of effective passes')
    ax.set_ylabel('Training loss $P(w)$')
    ax.set_title('MNIST Convex: Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, 'mnist_convex_training_loss.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved training loss plot to: {os.path.basename(out_path)}")

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Cannot find {RESULTS_FILE}. Please run merge_mnist_convex.py first.")
        return
        
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        
    print("Starting plotting...")
    plot_training_loss(data)
    plot_loss_residual(data)
    plot_variance(data)
    print("Done!")

if __name__ == '__main__':
    main()
