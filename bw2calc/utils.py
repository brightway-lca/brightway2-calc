# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

from .errors import AllArraysEmpty
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np
import os
import tarfile
import tempfile

MAX_INT_32 = 4294967295
MAX_SIGNED_INT_32 = 2147483647


def load_arrays(objs):
    """Load the numpy arrays from list of objects ``objs``.

    Currently accepts ``str`` filepaths, ``BytesIO``,
     ``numpy.ndarray`` arrays. Creates copies of objects"""

    arrays = []

    is_array = lambda x: isinstance(x, np.ndarray)
    is_filepath = lambda x: isinstance(x, (str, Path))
    is_other = lambda x: not is_array(x) and not is_filepath(x)

    for obj in (o for o in objs if is_array(o)):
        arrays.append(obj.copy())

    for obj in sorted([o for o in objs if is_filepath(o)]):
        arrays.append(np.load(obj, allow_pickle=True))

    for obj in sorted([o for o in objs if is_other(o)]):
        arrays.append(np.load(obj, allow_pickle=True))

    if all(arr.shape[0] == 0 for arr in arrays):
        raise AllArraysEmpty

    return np.hstack(arrays)


def extract_uncertainty_fields(array):
    """Extract the core set of fields needed for uncertainty analysis from a parameter array"""
    fields = ["uncertainty_type", "amount", 'loc',
              'scale', 'shape', 'minimum', 'maximum', 'negative']
    return array[fields].copy()


try:
    from bw2data import (
        config,
        Database,
        databases,
        geomapping,
        mapping,
        Method,
        methods,
        Normalization,
        normalizations,
        projects,
        Weighting,
        weightings,
    )

    from bw2data.utils import TYPE_DICTIONARY, safe_filename

    TYPE_DICTIONARY["generic production"] = 11
    TYPE_DICTIONARY["generic consumption"] = 12

    global_index = geomapping[config.global_location]

    # Extension packages should extend OBJECT_MAPPING
    # with their additional classes

    OBJECT_MAPPING = {
        'database': (Database, databases),
        'method': (Method, methods),
        'normalization': (Normalization, normalizations),
        'weighting': (Weighting, weightings),
    }

    def get_database_filepath(functional_unit):
        """Get filepaths for all databases in supply chain of `functional_unit`"""
        dbs = set.union(*[Database(key[0]).find_graph_dependents() for key in functional_unit])
        return [Database(obj).filepath_processed() for obj in dbs]

    def get_filepaths(name, kind):
        """Get filepath for datastore object `name` of kind `kind`"""
        if kind == "demand":
            return get_database_filepath(name)
        if name is None:
            return None
        data_store, metadata = OBJECT_MAPPING[kind]
        assert name in metadata, "Can't find {} object {}".format(kind, name)
        return [data_store(name).filepath_processed()]

    def clean_databases():
        databases.clean()

    def save_calculation_package(name, demand, **kwargs):
        """Save a calculation package for later use in an independent LCA.

        Args:
            * name (str): Name of file to create. Will have datetime appended.
            * demand (dict): Demand dictionary.
            * kwargs: Any additional keyword arguments, e.g. ``method``, ``iterations``...

        Returns the filepath of the calculation package archive.

        """
        _ = lambda x: [os.path.basename(y) for y in x]

        filepaths = get_filepaths(demand, "demand")
        data = {
            'demand': {mapping[k]: v for k, v in demand.items()},
            'database_filepath': _(filepaths),
            'adjust_filepaths': ['database_filepath'],
        }
        for key, value in kwargs.items():
            if key in OBJECT_MAPPING:
                data[key] = _(get_filepaths(value, key))
                data['adjust_filepaths'].append(key)
                filepaths.extend(get_filepaths(value, key))
            else:
                data[key] = value

        config_fp = os.path.join(
            projects.output_dir,
            "{}.config.json".format(safe_filename(name))
        )
        with open(config_fp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        archive = os.path.join(
            projects.output_dir,
            "{}.{}.tar.gz".format(
                safe_filename(name, False),
                datetime.datetime.now().strftime("%d-%B-%Y-%I-%M%p")
            )
        )
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(
                config_fp,
                arcname=os.path.basename(config_fp)
            )
            for filepath in filepaths:
                tar.add(
                    filepath,
                    arcname=os.path.basename(filepath)
                )

        os.remove(config_fp)
        return archive
except ImportError:
    get_filepaths = mapping = global_index = None

    def clean_databases():
        pass

    def save_calculation_package(*arg, **kwargs):
        raise NotImplemented

    # Maximum value for unsigned integer stored in 4 bytes
    TYPE_DICTIONARY = {
        "unknown": -1,
        "production": 0,
        "technosphere": 1,
        "biosphere": 2,
        "substitution": 3,
        # Added for single-matrix LCAs, which combine LCI and LCIA.
        # consumption values need to be multiplied by -1 in the matrix.
        "generic production": 11,
        "generic consumption": 12,
    }


def get_seed(seed=None):
    """Get valid Numpy random seed value"""
    # https://groups.google.com/forum/#!topic/briansupport/9ErDidIBBFM
    random = np.random.RandomState(seed)
    return random.randint(0, MAX_SIGNED_INT_32)


def load_calculation_package(fp):
    """Load a calculation package created by ``save_calculation_package``.

    NumPy arrays are saved to a temporary directory, and file paths are adjusted.

    ``fp`` is the absolute file path of a calculation package file.

    Returns a dictionary suitable for passing to an LCA object, e.g. ``LCA(**load_calculation_package(fp))``.

    """
    assert os.path.exists(fp), "Can't find file: {}".format(fp)

    temp_dir = tempfile.mkdtemp()
    with tarfile.open(fp, 'r|gz') as tar:
        tar.extractall(temp_dir)

    config_fps = [x for x in os.listdir(temp_dir) if x.endswith(".config.json")]
    assert len(config_fps) == 1, "Can't find configuration file"
    config = json.load(open(os.path.join(temp_dir, config_fps[0])))
    config['demand'] = {int(k): v for k, v in config['demand'].items()}

    for field in config.pop('adjust_filepaths'):
        config[field] = [os.path.join(temp_dir, fn) for fn in config[field]]

    return config


def wrap_functional_unit(dct):
    """Transform functional units for effective logging.

    Turns ``Activity`` objects into their keys."""
    data = []
    for key, amount in dct.items():
        try:
            data.append({'database': key[0],
                         'code': key[1],
                         'amount': amount})
        except:
            data.append({'key': key,
                         'amount': amount})
    return data


def md5(filepath, blocksize=65536):
    """Generate MD5 hash for file at `filepath`"""
    hasher = hashlib.md5()
    fo = open(filepath, 'rb')
    buf = fo.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = fo.read(blocksize)
    return hasher.hexdigest()
