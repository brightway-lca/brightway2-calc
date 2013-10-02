# -*- coding: utf-8 -*
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle


def load_arrays(dirpath, names):
    return np.hstack([pickle.load(open(os.path.join(
        dirpath, "processed", "%s.pickle" % name), "rb")
    ) for name in names])


def extract_uncertainty_fields(array):
    """Extract the core set of fields needed for uncertainty analysis from a parameter array"""
    fields = ["uncertainty_type", "amount", 'loc', 'scale', 'shape', 'minimum', 'maximum', 'negative']
    return array[fields].copy()
