import datetime
from pathlib import Path
from typing import Any

import bw_processing as bwp
import numpy as np
from bw_processing.io_helpers import generic_directory_filesystem
from fsspec import AbstractFileSystem
from fsspec.implementations.zip import ZipFileSystem

from bw2calc.errors import InconsistentGlobalIndex


def get_seed(seed=None):
    """Get valid Numpy random seed value"""
    # https://groups.google.com/forum/#!topic/briansupport/9ErDidIBBFM
    random = np.random.RandomState(seed)
    return random.randint(0, 2147483647)


def consistent_global_index(packages, matrix="characterization_matrix"):
    global_list = [
        resource.get("global_index")
        for package in packages
        for resource in package.filter_by_attribute("matrix", matrix)
        .filter_by_attribute("kind", "indices")
        .resources
    ]
    if len(set(global_list)) > 1:
        raise InconsistentGlobalIndex(
            f"Multiple global index values found: {global_list}. If multiple LCIA datapackages"
            + " are present, they must use the same value for ``GLO``, the global location, in "
            + " order for filtering for site-generic LCIA to work correctly."
        )
    return global_list[0] if global_list else None


def convert_tuple_to_list(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def wrap_functional_unit(dct):
    """Transform functional units for effective logging.
    Turns ``Activity`` objects into their keys."""
    data = []
    for key, amount in dct.items():
        if isinstance(key, int):
            data.append({"id": key, "amount": amount})
        else:
            try:
                data.append({"database": key[0], "code": key[1], "amount": amount})
            except TypeError:
                data.append({"key": key, "amount": amount})
    return data


def get_datapackage(obj):
    if isinstance(obj, bwp.DatapackageBase):
        return obj
    elif isinstance(obj, AbstractFileSystem):
        return bwp.load_datapackage(obj)
    elif isinstance(obj, Path) and obj.suffix.lower() == ".zip":
        return bwp.load_datapackage(ZipFileSystem(obj))
    elif isinstance(obj, Path) and obj.is_dir():
        return bwp.load_datapackage(generic_directory_filesystem(dirpath=obj))
    elif isinstance(obj, str) and obj.lower().endswith(".zip") and Path(obj).is_file():
        return bwp.load_datapackage(ZipFileSystem(Path(obj)))
    elif isinstance(obj, str) and Path(obj).is_dir():
        return bwp.load_datapackage(generic_directory_filesystem(dirpath=Path(obj)))

    else:
        raise TypeError("Unknown input type for loading datapackage: {}: {}".format(type(obj), obj))


def utc_now() -> datetime.datetime:
    """Get current datetime compatible with Py 3.8 to 3.12"""
    if hasattr(datetime, "UTC"):
        return datetime.datetime.now(datetime.UTC)
    else:
        return datetime.datetime.utcnow()


def relative_residual(matrix: Any, solution: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Relative residual ``||matrix @ solution - rhs|| / ||rhs||`` for each column of ``rhs``.

    Direct solvers don't always fail loudly on a singular or badly conditioned matrix; they can
    return numbers which look fine but don't solve the system. The residual costs one sparse
    matrix multiplication, so it is cheap insurance against silently wrong results.

    Columns of ``rhs`` which are all zero use the absolute residual instead, as there is nothing
    to be relative to.

    Returns a 1-d array with one value per column of ``rhs``."""
    rhs = np.asarray(rhs).reshape(rhs.shape[0], -1)
    solution = np.asarray(solution).reshape(rhs.shape)
    residual = np.linalg.norm(matrix @ solution - rhs, axis=0)
    scale = np.linalg.norm(rhs, axis=0)
    return np.divide(residual, scale, out=residual.copy(), where=scale > 0)
