# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

from .errors import MalformedFunctionalUnit
import itertools
import numpy as np
import datetime
import os
import json
import tarfile
import tempfile
try:
    import cPickle as pickle
except ImportError:
    import pickle


def load_arrays(paths):
    """Load the numpy arrays in list of filepaths ``paths``."""
    assert all(os.path.isfile(fp) for fp in paths)
    return np.hstack([np.load(path) for path in sorted(paths)])


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

    from bw2data.utils import TYPE_DICTIONARY, MAX_INT_32, safe_filename

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
    MAX_INT_32 = 4294967295
    TYPE_DICTIONARY = {
        "unknown": -1,
        "production": 0,
        "technosphere": 1,
        "biosphere": 2,
        "substitution": 3,
    }

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
