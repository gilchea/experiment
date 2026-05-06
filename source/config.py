"""
config.py — Centralized Hyperparameters for All Experiments

All dataset-specific settings (learning rates, regularization, etc.) are
defined here to keep train.py clean and make tuning easier.
"""

DATASET_CONFIGS = {
    'mnist': {
        'lam': 1e-4,
        'multiclass': True,
        # SVRG
        'svrg_lr': 0.025,
        'svrg_m_factor': 2,           # m = 2 * n
        # SGD constant
        'sgd_const_lr': 0.01,
        # SGD best (decaying)
        'sgd_best_lr0': 0.1,
        'sgd_best_a': 1.0,            # Tune this parameter
        # SDCA
        'sdca_lr': 1.0,
        # SAG
        'sag_lr': 0.01,
        # Warm-start
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        # Epochs
        'n_outer': 30,                # SVRG outer iterations
        'n_epochs_sgd': 90,           # SGD epochs (match effective passes)
        'n_epochs_sdca': 90,          # SDCA epochs
        'n_epochs_sag': 90,           # SAG epochs
    },
    'cifar10': {
        'lam': 1e-3,
        'multiclass': True,
        # SVRG
        'svrg_lr': 0.01,
        'svrg_m_factor': 2,
        # SGD constant
        'sgd_const_lr': 0.01,
        # SGD best (decaying)
        'sgd_best_lr0': 0.1,
        'sgd_best_a': 1.0,
        # SDCA
        'sdca_lr': 1.0,
        # SAG
        'sag_lr': 0.01,
        # Warm-start
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        # Epochs
        'n_outer': 30,
        'n_epochs_sgd': 90,
        'n_epochs_sdca': 90,
        'n_epochs_sag': 90,
    },
    'rcv1': {
        'lam': 1e-5,
        'multiclass': False,
        # SVRG
        'svrg_lr': 0.01,
        'svrg_m_factor': 2,
        # SGD constant
        'sgd_const_lr': 0.001,
        # SGD best (decaying)
        'sgd_best_lr0': 0.01,
        'sgd_best_a': 1.0,
        # SDCA
        'sdca_lr': 1.0,
        # SAG
        'sag_lr': 0.001,
        # Warm-start
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        # Epochs
        'n_outer': 30,
        'n_epochs_sgd': 90,
        'n_epochs_sdca': 90,
        'n_epochs_sag': 90,
    },
    'covtype': {
        'lam': 1e-5,
        'multiclass': False,
        # SVRG
        'svrg_lr': 0.001,
        'svrg_m_factor': 2,
        # SGD constant
        'sgd_const_lr': 0.001,
        # SGD best (decaying)
        'sgd_best_lr0': 0.01,
        'sgd_best_a': 1.0,
        # SDCA
        'sdca_lr': 1.0,
        # SAG
        'sag_lr': 0.001,
        # Warm-start
        'warm_start_epochs': 1,
        'warm_start_lr': 0.01,
        # Epochs
        'n_outer': 30,
        'n_epochs_sgd': 90,
        'n_epochs_sdca': 90,
        'n_epochs_sag': 90,
    },
}
