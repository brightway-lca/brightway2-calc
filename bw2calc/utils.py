import datetime
from pathlib import Path

import bw_processing as bwp
import numpy as np
from bw_processing.io_helpers import generic_directory_filesystem
from fsspec import AbstractFileSystem
from fsspec.implementations.zip import ZipFileSystem

from .errors import InconsistentGlobalIndex


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
