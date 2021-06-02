from .errors import InconsistentGlobalIndex
from fs.base import FS
from fs.osfs import OSFS
from fs.zipfs import ZipFS
from pathlib import Path
import bw_processing as bwp
import numpy as np


def get_seed(seed=None):
    """Get valid Numpy random seed value"""
    # https://groups.google.com/forum/#!topic/briansupport/9ErDidIBBFM
    random = np.random.RandomState(seed)
    return random.randint(0, 2147483647)


def consistent_global_index(packages, matrix="characterization"):
    global_list = {p.metadata.get("global_index") for p in (obj.filter_by_attribute("matrix", matrix) for obj in packages)}
    if len(global_list.difference({None})) > 1:
        raise InconsistentGlobalIndex(f"Multiple global index values found ({global_list}). If multiple LCIA datapackages are present, they must use the same value for ``GLO``, the global location, in order for filtering for site-generic LCIA to work correctly.")


def wrap_functional_unit(dct):
    """Transform functional units for effective logging.
    Turns ``Activity`` objects into their keys."""
    data = []
    for key, amount in dct.items():
        if isinstance(key, int):
            data.append({"id": key, "amount": amount})
        else:
            try:
                data.append({'database': key[0],
                             'code': key[1],
                             'amount': amount})
            except TypeError:
                data.append({'key': key,
                             'amount': amount})
    return data


def get_datapackage(obj):
    if isinstance(obj, bwp.DatapackageBase):
        return obj
    elif isinstance(obj, FS):
        return bwp.load_datapackage(obj)
    elif isinstance(obj, Path) and obj.suffix.lower() == ".zip":
        return bwp.load_datapackage(ZipFS(obj))
    elif isinstance(obj, Path) and obj.is_dir():
        return bwp.load_datapackage(OSFS(obj))
    else:
        raise TypeError("Unknown input type for loading datapackage: {}: {}".format(type(obj), obj))
