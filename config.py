"""
config.py — Centralized Hyperparameters for All Experiments

All dataset-specific settings (learning rates, regularization, etc.) are
defined here to keep train.py clean and make tuning easier.
"""

DATASET_CONFIGS = {
    'mnist': {
        'lam': 1e-4,
        'multiclass': True,
        'svrg_lr': 0.025,
        'svrg_m_factor': 2,           # m = 2 * n
        'sgd_const_lr': 0.01,
        'sgd_best_lr0': 0.1,
        'sgd_best_a': 1.0,            # Tune this parameter
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        'n_outer': 30,                # SVRG outer iterations
        'n_epochs_sgd': 90,           # SGD epochs (match effective passes)
    },
    'cifar10': {
        'lam': 1e-3,
        'multiclass': True,
        'svrg_lr': 0.01,
        'svrg_m_factor': 2,
        'sgd_const_lr': 0.01,
        'sgd_best_lr0': 0.1,
        'sgd_best_a': 1.0,
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        'n_outer': 30,
        'n_epochs_sgd': 90,
    },
    'rcv1': {
        'lam': 1e-5,
        'multiclass': False,
        'svrg_lr': 0.01,
        'svrg_m_factor': 2,
        'sgd_const_lr': 0.001,
        'sgd_best_lr0': 0.01,
        'sgd_best_a': 1.0,
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        'n_outer': 30,
        'n_epochs_sgd': 90,
    },
    'covtype': {
        'lam': 1e-5,
        'multiclass': False,
        'svrg_lr': 0.001,
        'svrg_m_factor': 2,
        'sgd_const_lr': 0.001,
        'sgd_best_lr0': 0.01,
        'sgd_best_a': 1.0,
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        'n_outer': 30,
        'n_epochs_sgd': 90,
    },
}
