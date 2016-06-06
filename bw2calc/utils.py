# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

from .errors import MalformedFunctionalUnit
import itertools
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle


def load_arrays(paths):
    """Load the numpy arrays in list of filepaths ``paths``."""
    assert all(os.path.exists(fp) for fp in paths)
    return np.hstack([pickle.load(open(path, "rb")) for path in sorted(paths)])


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
        Weighting,
        weightings,
    )

    from bw2data.utils import TYPE_DICTIONARY, MAX_INT_32

    global_index = geomapping[config.global_location]

    OBJECT_MAPPING = {
        'database': (Database, databases),
        'method': (Method, methods),
        'normalization': (Normalization, normalizations),
        'weighting': (Weighting, weightings),
    }

    def get_filepaths(name, kind):
        """Get filepath for datastore object `name` of kind `kind`"""
        if name is None:
            return None
        data_store, metadata = OBJECT_MAPPING[kind]
        assert name in metadata, "Can't find {} object {}".format(kind, name)
        return [data_store(name).filepath_processed()]

    def get_database_filepaths(functional_unit, database_list):
        """Get filepaths for all databases in supply chain of `functional_unit`"""
        dbs = set.union(*[Database(key[0]).find_graph_dependents() for key in functional_unit])
        return [Database(obj).filepath_processed() for obj in dbs]

    def clean_databases():
        databases.clean()
except ImportError:
    get_filepaths = get_database_filepaths = mapping = global_index = None

    def clean_databases():
        pass

    # Maximum value for unsigned integer stored in 4 bytes
    MAX_INT_32 = 4294967295
    TYPE_DICTIONARY = {
        "unknown": -1,
        "production": 0,
        "technosphere": 1,
        "biosphere": 2,
        "substitution": 3,
    }
