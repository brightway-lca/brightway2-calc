# flake8: noqa
__all__ = [
    "DenseLCA",
    "LCA",
    "LeastSquaresLCA",
    "IterativeLCA",
    "MultiLCA",
]

__version__ = "2.0.DEV16"


import json_logging

json_logging.init_non_web(enable_json=True)

try:
    from pypardiso import factorized, spsolve

    PYPARDISO = True
except ImportError:
    from scipy.sparse.linalg import factorized, spsolve

    PYPARDISO = False
try:
    from presamples import PackagesDataLoader
except ImportError:
    PackagesDataLoader = None
try:
    from bw2data import __version__ as _bw2data_version
    from bw2data import get_activity, prepare_lca_inputs

    if not _bw2data_version >= (4, 0):
        raise ImportError
except ImportError:
    prepare_lca_inputs = get_activity = None


from .dense_lca import DenseLCA
from .iterative_lca import IterativeLCA
from .lca import LCA
from .least_squares import LeastSquaresLCA
from .multi_lca import MultiLCA
