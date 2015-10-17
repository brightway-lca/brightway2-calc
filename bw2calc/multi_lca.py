# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from eight import *

from .lca import LCA
try:
    from bw2data import calculation_setups
except ImportError:
    calculation_setups = None
import numpy as np


class MultiLCA(object):
    """Wrapper class for performing LCA calculations with many functional units and LCIA methods.

    Needs to be passed a ``calculation_setup`` name."""
    def __init__(self, cs_name):
        if not calculation_setups:
            raise ImportError
        assert cs_name in calculation_setups
        try:
            cs = calculation_setups[cs_name]
        except KeyError:
            raise ValueError(
                "{} is not a known `calculation_setup`.".format(cs_name)
            )
        self.activities = cs['inv']
        self.methods = cs['ia']
        self.lca = LCA(demand=self.all, method=self.methods[0])
        self.lca.lci(factorize=True)
        self.method_matrices = []
        self.results = np.zeros((len(self.activities), len(self.methods)))
        for method in self.methods:
            self.lca.method = method
            self.lca.load_lcia_data()
            self.method_matrices.append(self.lca.characterization_matrix)

        for row, activity in enumerate(self.activities):
            self.lca.redo_lci({activity[0]: float(activity[1])})
            for col, cf_matrix in enumerate(self.method_matrices):
                self.lca.characterization_matrix = cf_matrix
                self.lca.lcia_calculation()
                self.results[row, col] = self.lca.score

    @property
    def all(self):
        return {k: float(v) for k, v in self.activities}
