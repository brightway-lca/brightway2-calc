# flake8: noqa
__all__ = [
    "CachingLCA",
    "DenseLCA",
    "LCA",
    "LeastSquaresLCA",
    "IterativeLCA",
    "MethodConfig",
    "MultiLCA",
]

__version__ = "2.0"


import platform
import warnings

ARM = {"arm", "arm64", "aarch64_be", "aarch64", "armv8b", "armv8l"}
AMD_INTEL = {"ia64", "i386", "i686", "x86_64"}
UMFPACK_WARNING = """
It seems like you have an ARM architecture, but haven't installed scikit-umfpack:

    https://pypi.org/project/scikit-umfpack/

Installing it could give you much faster calculations.
"""
PYPARDISO_WARNING = """
It seems like you have an AMD/INTEL x64 architecture, but haven't installed pypardiso:

    https://pypi.org/project/pypardiso/

Installing it could give you much faster calculations.
"""

PYPARDISO, UMFPACK = False, False

try:
    from pypardiso import factorized, spsolve

    PYPARDISO = True
except ImportError:
    pltf = platform.machine().lower()

    if pltf in ARM:
        try:
            import scikits.umfpack

            UMFPACK = True
        except ImportError:
            warnings.warn(UMFPACK_WARNING)
    elif pltf in AMD_INTEL:
        warnings.warn(PYPARDISO_WARNING)

    from scipy.sparse.linalg import factorized, spsolve
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


from .caching_lca import CachingLCA
from .dense_lca import DenseLCA
from .iterative_lca import IterativeLCA
from .lca import LCA
from .least_squares import LeastSquaresLCA
from .method_config import MethodConfig
from .multi_lca import MultiLCA
