from .sgd import (
    sgd_epoch_constant,
    sgd_constant,
    sgd_epoch_decay,
    sgd_decay,
    warm_start,
    count_effective_passes_sgd,
)
from .svrg import (
    svrg_outer_loop,
    effective_passes_svrg,
)
from .sdca import (
    sdca_train,
)

from .svrg_nn import (
    svrg_nn_outer_loop,
    effective_passes_svrg_nn,
)
from .sgd_nn import (
    sgd_nn_epoch_constant,
    sgd_nn_epoch_decay,
    warm_start_nn,
)

__all__ = [
    'sgd_epoch_constant', 'sgd_constant', 'sgd_epoch_decay', 'sgd_decay',
    'warm_start', 'count_effective_passes_sgd',
    'svrg_outer_loop', 'effective_passes_svrg',
    'sdca_train',

    'svrg_nn_outer_loop', 'effective_passes_svrg_nn',
    'sgd_nn_epoch_constant', 'sgd_nn_epoch_decay', 'warm_start_nn',
]
