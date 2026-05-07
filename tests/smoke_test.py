import sys
import os
import numpy as np
import json
import shutil

# Add project root and source to path
current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'source'))

# Project imports
try:
    import source.utils.data_loader as data_loader
    import source.train as train
    import source.train_nn as train_nn
    from source.config import DATASET_CONFIGS, NN_CONFIGS
except ImportError:
    import utils.data_loader as data_loader
    import train as train
    import train_nn as train_nn
    from config import DATASET_CONFIGS, NN_CONFIGS

def clean_str(s):
    """Remove non-ASCII characters for Windows terminal compatibility."""
    return "".join(i for i in str(s) if ord(i) < 128)

# Monkey-patch load_dataset
original_load_dataset = data_loader.load_dataset

def smoke_load_dataset(name):
    print(f"  [SmokeTest] Loading sample data for: {name}...")
    try:
        X_train, y_train, X_test, y_test = original_load_dataset(name)
        n_samples = 50
        return X_train[:n_samples], y_train[:n_samples], X_test[:n_samples], y_test[:n_samples]
    except Exception as e:
        safe_msg = clean_str(e)
        print(f"  [SmokeTest] Cannot load real data. Generating synthetic data... (Error: {safe_msg[:50]}...)")
        d = 100
        if name == 'mnist': d = 784
        elif name == 'cifar10': d = 3072
        elif name == 'rcv1': d = 47236
        elif name == 'covtype': d = 54
        
        X = np.random.randn(100, d)
        y = np.random.randint(0, 2 if name in ['rcv1', 'covtype'] else 10, 100)
        if name in ['rcv1', 'covtype']: y = np.where(y == 0, -1.0, 1.0)
        return X[:50], y[:50], X[50:70], y[50:70]

# Monkey-patch load_dataset in all relevant modules
data_loader.load_dataset = smoke_load_dataset
if hasattr(train, 'load_dataset'):
    train.load_dataset = smoke_load_dataset
if hasattr(train_nn, 'load_dataset'):
    train_nn.load_dataset = smoke_load_dataset

def get_smoke_config(original_config):
    smoke_cfg = original_config.copy()
    smoke_cfg['n_outer'] = 2
    smoke_cfg['n_epochs_sgd'] = 2
    smoke_cfg['n_epochs_sdca'] = 2
    smoke_cfg['n_epochs_sag'] = 2
    smoke_cfg['warm_start_epochs'] = 1
    return smoke_cfg

def run_smoke_test():
    smoke_results_dir = 'smoke_results'
    smoke_figures_dir = 'smoke_figures'
    
    for d in [smoke_results_dir, smoke_figures_dir, 'checkpoints']:
        if os.path.exists(d):
            try: shutil.rmtree(d)
            except: pass
    
    os.makedirs(smoke_results_dir, exist_ok=True)
    os.makedirs(smoke_figures_dir, exist_ok=True)

    print("\n" + "="*60)
    print("      SYSTEM SMOKE TEST - PROJECT SVRG")
    print("="*60)
    
    fake_p_star = 0.1 

    print("\n>>> TESTING CONVEX MODELS")
    for name in list(DATASET_CONFIGS.keys()):
        config = DATASET_CONFIGS[name]
        print(f"\n[*] Checking: {name}")
        smoke_cfg = get_smoke_config(config)
        try:
            train.run_experiment(name, smoke_cfg, fake_p_star, results_dir=smoke_results_dir, save_ckpt_every=1)
            print(f"  [V] Success: {name}")
        except Exception as e:
            safe_e = clean_str(e)
            print(f"  [X] ERROR at {name}: {safe_e}")

    print("\n>>> TESTING NEURAL NETWORKS")
    for name in list(NN_CONFIGS.keys()):
        config = NN_CONFIGS[name]
        print(f"\n[*] Checking: {name}")
        smoke_cfg = get_smoke_config(config)
        try:
            train_nn.run_nn_experiment(name, smoke_cfg, results_dir=smoke_results_dir, save_ckpt_every=1)
            print(f"  [V] Success: {name}")
        except Exception as e:
            safe_e = clean_str(e)
            print(f"  [X] ERROR at {name}: {safe_e}")

    print("\n" + "="*60)
    print("      SMOKE TEST COMPLETED")
    print("="*60)

if __name__ == '__main__':
    run_smoke_test()
