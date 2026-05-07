"""Smoke test for config.py and train.py imports"""
import sys
from source.config import DATASET_CONFIGS
print('config.py: OK -', len(DATASET_CONFIGS), 'datasets configured')

from source.train import run_experiment, run_all_experiments
print('train.py: OK - functions imported')

# Verify config structure
for name, cfg in DATASET_CONFIGS.items():
    required = ['lam', 'multiclass', 'svrg_lr', 'svrg_m_factor',
                'sgd_const_lr', 'sgd_best_lr0', 'sgd_best_a',
                'warm_start_epochs', 'warm_start_lr', 'n_outer', 'n_epochs_sgd']
    missing = [k for k in required if k not in cfg]
    assert not missing, f'{name} missing: {missing}'
    print(f'  {name}: lam={cfg["lam"]}, multiclass={cfg["multiclass"]}, '
          f'svrg_lr={cfg["svrg_lr"]}, sgd_const_lr={cfg["sgd_const_lr"]}')

print()
print('All imports and config validation passed!')
sys.stdout.flush()
