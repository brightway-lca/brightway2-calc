# -*- coding: utf-8 -*
from __future__ import division
from ..mc_vector import ParameterVectorLCA
from scipy import stats
import math
import multiprocessing
import numpy as np


def _ctv_worker(args):
    kwargs, iterations, mask = args
    lca = ParameterVectorLCA(**kwargs)
    lca.load_data()
    results = np.zeros(iterations)
    inputs = np.zeros((mask.sum(), iterations))
    reference_vector = lca.params['amount']

    for x in xrange(iterations):
        sample = lca.rng.next()
        vector = reference_vector.copy()
        vector[mask] = sample[mask]
        inputs[:, x] = sample[mask]
        results[x] = lca(vector)

    return inputs, results


class ContributionToVariance(object):
    def __init__(self, kwargs, mask=None, iterations=10000, cpus=None):
        self.kwargs = kwargs
        self.iterations = iterations
        self.cpus = cpus or multiprocessing.cpu_count()
        self.chunk_size = int(math.ceil(self.iterations / self.cpus))
        self.lca = ParameterVectorLCA(**kwargs)
        self.lca.load_data()
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.ones(self.lca.params.shape, dtype=bool)
        self.n_params = self.mask.sum()
        self.parameter_indices = np.arange(self.mask.shape[0])[self.mask]

    def get_correlation_coefficient(self, x, y):
        ranks_x = stats.rankdata(x)
        ranks_y = stats.rankdata(y)
        return stats.kendalltau(ranks_x, ranks_y)[0]

    def calculate(self):
        pool = multiprocessing.Pool(self.cpus)
        args = [(self.kwargs, self.chunk_size, self.mask.copy()) for x in xrange(self.cpus)]
        results = pool.map(_ctv_worker, args)

        self.inputs = np.hstack([x[0] for x in results])
        self.results = np.hstack([x[1] for x in results])

        ctv = np.zeros(self.n_params)
        for x in xrange(ctv.shape[0]):
            ctv[x] = self.get_correlation_coefficient(
                self.inputs[x, :],
                self.results
            )

        ctv[np.isnan(ctv)] = 0
        self.ctv = ctv ** 2 / (ctv ** 2).sum()
        return self.ctv

    def sort_ctv(self):
        assert hasattr(self, "ctv")
        self.sorted_results = [
            (self.ctv[index], self.parameter_indices[index], index)
            for index in range(self.n_params)]
        self.sorted_results.sort(reverse=True)
        self.sorted_results = self.sorted_results
        return self.sorted_results
