"""
merge_mnist_convex.py — Merge separate results into one JSON
"""

import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MERGED_FILE = os.path.join(RESULTS_DIR, 'mnist_convex_results.json')

# The separate result files
FILES = [
    'svrg_mnist_convex.json',
    'sgd_const_mnist_convex.json',
    'sgd_best_mnist_convex.json',
    'sdca_mnist_convex.json'
]

# Shared configuration that we know was used
DATASET = 'mnist'
LAM = 1e-4

def main():
    print("=" * 60)
    print("Merging MNIST Convex Results")
    print("=" * 60)

    # Load P_star from optimal_loss.json
    optimal_path = os.path.join(RESULTS_DIR, 'optimal_loss.json')
    if not os.path.exists(optimal_path):
        print(f"ERROR: {optimal_path} not found.")
        return
    with open(optimal_path, 'r') as f:
        optimal = json.load(f)
    P_star = float(optimal[DATASET]['P_star'])

    # Initialize merged structure
    merged_results = {
        'dataset': DATASET,
        'P_star': P_star,
        'config': {
            'lam': LAM,
            'svrg_lr': 0.025,
            'svrg_m_factor': 2,
            'sgd_const_lrs': [0.001, 0.0025, 0.005],
            'sgd_best_lr0': 0.1,
            'sgd_best_b': 1.0,
            'warm_start_epochs': 1,
        }
    }

    # Merge individual results
    all_success = True
    for file in FILES:
        filepath = os.path.join(RESULTS_DIR, file)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            all_success = False
            continue
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for key, value in data.items():
            merged_results[key] = value
            print(f"  Merged key '{key}' from {file}")

    # Save merged
    with open(MERGED_FILE, 'w') as f:
        json.dump(merged_results, f, indent=2)
    
    print(f"\n[OK] Merged results saved to {MERGED_FILE}")
    if not all_success:
        print("Note: Some files were missing. Run all individual scripts first.")

if __name__ == '__main__':
    main()
