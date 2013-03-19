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
