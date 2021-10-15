__all__ = [
    "ComparativeMonteCarlo",
    "DenseLCA",
    "direct_solving_worker",
    # "DirectSolvingMixin",
    # "DirectSolvingMonteCarloLCA",
    "GraphTraversal",
    "LCA",
    "LeastSquaresLCA",
    # "load_calculation_package",
    # "MonteCarloLCA",
    "MultiLCA",
    # "MultiMonteCarlo",
    # "ParallelMonteCarlo",
    # "ParameterVectorLCA",
    # "save_calculation_package",
]

from .version import version as __version__

try:
    import json_logging

    json_logging.init_non_web(enable_json=True)
except ImportError:
    pass


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
    from bw2data import prepare_lca_inputs

    if not _bw2data_version >= (4, 0):
        raise ImportError
except ImportError:
    prepare_lca_inputs = None


from .dense_lca import DenseLCA
from .graph_traversal import GraphTraversal
from .lca import LCA
from .least_squares import LeastSquaresLCA
from .multi_lca import MultiLCA

# from .utils import save_calculation_package, load_calculation_package

try:
    from .mc_vector import ParameterVectorLCA
    from .monte_carlo import (
        ComparativeMonteCarlo,
        DirectSolvingMixin,
        DirectSolvingMonteCarloLCA,
        MonteCarloLCA,
        MultiMonteCarlo,
        ParallelMonteCarlo,
        direct_solving_worker,
    )
except ImportError:
    None
