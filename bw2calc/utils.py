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
    return np.hstack([pickle.load(open(path, "rb")) for path in paths])


def extract_uncertainty_fields(array):
    """Extract the core set of fields needed for uncertainty analysis from a parameter array"""
    fields = ["uncertainty_type", "amount", 'loc',
              'scale', 'shape', 'minimum', 'maximum', 'negative']
    return array[fields].copy()


try:
    from bw2data import (
        Database,
        databases,
        methods,
        mapping,
        Method,
        Normalization,
        projects,
        Weighting,
    )
    from bw2data.utils import TYPE_DICTIONARY, MAX_INT_32

    class Translate(object):
        def dependent_database_filepaths(self, demand):
            try:
                return {Database(obj).filepath_processed() for obj in itertools.chain(
                    *[Database(key[0]).find_graph_dependents() for key in demand])
                }
            except TypeError:
                raise MalformedFunctionalUnit("The given functional unit is "
                    "not a valid activity key: {}".format(demand))

        def independent(self, demand, databases):
            if not all(isinstance(obj, int) for obj in demand):
                if databases:
                    raise MalformedFunctionalUnit("Can't specify database array filepaths for normal demand")
                elif any(isinstance(obj, int) for obj in demand):
                    raise MalformedFunctionalUnit("Can't specify hybrid demand (independent and normal)")
                return False
            else:
                if not databases:
                    raise MalformedFunctionalUnit("Must specify database array filepaths")
                return True

        def __call__(self, demand, filepaths, method, weighting, norm):
            # Write all modified databases to fresh parameter arrays
            databases.clean()
            # Validity checks
            if self.independent(demand, filepaths):
                return True, filepaths, method, weighting, norm
            else:
                return (
                    False,
                    self.dependent_database_filepaths(demand),
                    [Method(method).filepath_processed()] if method else [],
                    [Weighting(weighting).filepath_processed()] if weighting else [],
                    [Normalization(norm).filepath_processed()] if norm else [],
                )

    translate = Translate()
except ImportError:
    # bw2data not present. Assume demand is ID numbers
    # Validity check
    def translate(demand, databases, method, weighting, normalization):
        return True, databases, method, weighting, normalization

    mapping = None
    # Maximum value for unsigned integer stored in 4 bytes
    MAX_INT_32 = 4294967295
    TYPE_DICTIONARY = {
        "unknown": -1,
        "production": 0,
        "technosphere": 1,
        "biosphere": 2,
        "substitution": 3,
    }
