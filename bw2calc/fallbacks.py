# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

from .utils import MAX_INT_32
import itertools
import numpy as np


def indexer(array_from, array_to, mapping):
    """
Chosen algorithm is faster than using rankdata, or simple iteration.
Modifies numpy arrays in place.

map(dict.get, data) seems faster, but doesn't work with missing values.

Setup code to try new approaches::

    from numpy import *
    from time import time
    a = random.random_integers(5000, size=100000)
    indices = sorted(list(set([int(x) for x in a])))
    mapping = dict(zip(indices, range(len(indices))))
    b = zeros(a.shape[0])

    """
    array_to[:] = np.array([mapping.get(x, MAX_INT_32) for x in array_from])


def dicter(array):
    """
Create dictionary from the sorted, unique values in a numpy array

Already more than fast enough. Setup code for other approaches::

        %%timeit import numpy as np; array = np.random.randint(0, 5000, size=50000)
        dict(zip(np.sort(np.unique(array)), itertools.count()))

    """
    return dict(zip(
        (int(x) for x in np.sort(np.unique(array))),
        itertools.count()
    ))
