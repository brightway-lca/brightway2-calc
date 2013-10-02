# -*- coding: utf-8 -*
from __future__ import division
from .. import ParameterVectorLCA
from .percentages import get_percentages
import math
import multiprocessing
import numpy as np


def replace_one(old_values, new_values, index):
    p = old_values.copy()
    p[index] = new_values[index]
    return p


def _prescreen_worker(args):
    start, stop, params, percentages, kwargs = args
    lca = ParameterVectorLCA(**kwargs)
    lca.load_data()
    return start, stop, [
        lca(replace_one(params, percentages, index))
        for index in range(start, stop)]


class OATSensitivity(object):
    """Evaluate LCA model parameters using a simple one at a time (OAT) test to screen out unimportant parameters."""
    def __init__(self, kwargs, percentage=0.9, seed=None, cpus=None):
        self.kwargs = kwargs
        self.percentage = percentage
        self.seed = seed
        self.cpus = cpus or multiprocessing.cpu_count()
        self.lca = ParameterVectorLCA(**kwargs)
        self.lca.load_data()
        self.params = self.lca.params.copy()
        self.chunk_size = int(math.ceil(self.params.shape[0] / self.cpus))
        self.ref_score = self.lca(self.params['amount'].copy())

    def screen(self, cutoff=100):
        self.percentages = get_percentages(self.percentage, self.params,
                                      seed=self.seed, num_cpus=self.cpus)
        pool = multiprocessing.Pool(self.cpus)
        args = [[
            self.chunk_size * index,
            self.chunk_size * (index + 1),
            self.params['amount'].copy(),
            self.percentages.copy(),
            self.kwargs] for index in range(self.cpus)]
        args[-1][1] = self.params.shape[0]
        results = pool.map(_prescreen_worker, args)
        self.delta = np.zeros(self.params.shape[0])
        for start, stop, values in results:
            self.delta[start:stop] = values
        self.deltas = np.abs((self.delta - self.ref_score) / self.ref_score)
        self.top_indices = np.argsort(self.deltas)[:-(cutoff + 1):-1]
        return self.top_indices

    def get_mask(self):
        self.mask = np.zeros(self.params.shape, dtype=bool)
        self.mask[self.top_indices] = True
        return self.mask
