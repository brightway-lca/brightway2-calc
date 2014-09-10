# -*- coding: utf-8 -*
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle


def load_arrays(dirpath, names):
    """Load and concatenate the numpy arrays ``names`` in directory ``dirpath``."""
    return np.hstack([pickle.load(open(os.path.join(
        dirpath,
        u"processed", name if u".pickle" in name else u"%s.pickle" % name
    ), "rb")
    ) for name in names])


def extract_uncertainty_fields(array):
    """Extract the core set of fields needed for uncertainty analysis from a parameter array"""
    fields = ["uncertainty_type", "amount", 'loc',
              'scale', 'shape', 'minimum', 'maximum', 'negative']
    return array[fields].copy()
