# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from time import time


class SpeedTest:
    """Compare speed of sparse matrix operations on this machine compared to a reference machine."""

    size = 1000
    density = 0.02
    seed = 42

    def test(self):
        sm = self.get_sparse_matrix()
        v = self.get_demand_vector()
        now = time()
        for x in range(100):
            spsolve(sm, v)
        return time() - now

    def ratio(self):
        """On the reference machine, this takes about 5.85 seconds"""
        return 5.85 / self.test()

    def get_sparse_matrix(self):
        """Adapted from scipy to use seeded RNG"""
        self.size = 1000
        k = self.density * self.size ** 2
        rng = np.random.RandomState(self.seed)
        i = rng.randint(self.size, size=k)
        j = rng.randint(self.size, size=k)
        data = rng.rand(k)
        return coo_matrix((data, (i, j)), shape=(self.size, self.size)).tocsr()

    def get_demand_vector(self):
        v = np.zeros((self.size,))
        v[42] = 1
        return v
