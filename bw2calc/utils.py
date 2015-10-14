# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle


try:
    from bw2data.utils import TYPE_DICTIONARY, MAX_INT_32
except ImportError:
    # Maximum value for unsigned integer stored in 4 bytes
    MAX_INT_32 = 4294967295
    TYPE_DICTIONARY = {
        "unknown": -1,
        "production": 0,
        "technosphere": 1,
        "biosphere": 2,
        "substitution": 3,
    }


def load_arrays(paths):
    """Load the numpy arrays in list of filepaths ``paths``."""
    return np.hstack([pickle.load(open(path), "rb")) for path in paths])


def extract_uncertainty_fields(array):
    """Extract the core set of fields needed for uncertainty analysis from a parameter array"""
    fields = ["uncertainty_type", "amount", 'loc',
              'scale', 'shape', 'minimum', 'maximum', 'negative']
    return array[fields].copy()
