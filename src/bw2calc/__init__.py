# flake8: noqa
__all__ = [
    "CachingLCA",
    "DenseLCA",
    "LCA",
    "LeastSquaresLCA",
    "IterativeLCA",
    "MethodConfig",
    "MultiLCA",
    "FastScoresOnlyMultiLCA",
]

__version__ = "2.3.2"


import platform
import warnings

from packaging.version import Version

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

    try:
        import scikits.umfpack

        UMFPACK = True
    except ModuleNotFoundError:
        if pltf in ARM:
            warnings.warn(UMFPACK_WARNING)
        elif pltf in AMD_INTEL:
            warnings.warn(PYPARDISO_WARNING)
        else:
            warnings.warn("No fast sparse solver found")
    except ImportError as e:
        warnings.warn(f"scikit-umfpack found but couldn't be imported. Error: {e}")

    from scipy.sparse.linalg import factorized, spsolve
try:
    from presamples import PackagesDataLoader
except ImportError:
    PackagesDataLoader = None


from bw2calc.caching_lca import CachingLCA
from bw2calc.dense_lca import DenseLCA
from bw2calc.fast_scores import FastScoresOnlyMultiLCA
from bw2calc.iterative_lca import IterativeLCA
from bw2calc.lca import LCA
from bw2calc.least_squares import LeastSquaresLCA
from bw2calc.method_config import MethodConfig
from bw2calc.multi_lca import MultiLCA
