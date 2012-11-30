# -*- coding: utf-8 -*
from __future__ import division
from brightway2 import config as base_config
from brightway2 import databases, methods, mapping
from bw2data.proxies import OneDimensionalArrayProxy, \
    CompressedSparseMatrixProxy
from bw2data.utils import MAX_INT_32
from fallbacks import dicter
from scipy.sparse.linalg import factorized, spsolve
from scipy import sparse
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from bw2speedups import indexer
except ImportError:
    from fallbacks import indexer


class LCA(object):
    def __init__(self, demand, method=None, config=None):
        self.config = config or base_config
        if isinstance(demand, (basestring, tuple, list)):
            raise ValueError("Demand must be a dictionary")
        self.demand = demand
        self.method = method
        self.databases = self.get_databases()

    def get_databases(self):
        """Get list of databases for functional unit"""
        return set.union(*[set(databases[key[0]]["depends"] + [key[0]]) for key \
            in self.demand])

    def load_databases(self):
        params = np.hstack([pickle.load(open(os.path.join(
            self.config.dir, "processed", "%s.pickle" % name), "rb")
            ) for name in self.databases])
        # Technosphere
        self.tech_params = params[np.where(params['technosphere'] == True)]
        self.bio_params = params[np.where(params['technosphere'] == False)]
        self.technosphere_dict = self.build_dictionary(np.hstack((
            self.tech_params['input'], self.tech_params['output'],
            self.bio_params['output'])))
        self.add_matrix_indices(self.tech_params['input'], self.tech_params['row'],
            self.technosphere_dict)
        self.add_matrix_indices(self.tech_params['output'], self.tech_params['col'],
            self.technosphere_dict)
        # Biosphere
        self.biosphere_dict = self.build_dictionary(self.bio_params['input'])
        self.add_matrix_indices(self.bio_params['input'], self.bio_params['row'],
            self.biosphere_dict)
        self.add_matrix_indices(self.bio_params['output'], self.bio_params['col'],
            self.technosphere_dict)

    def load_method(self):
        params = pickle.load(open(os.path.join(self.config.dir, "processed",
            "%s.pickle" % methods[self.method]['abbreviation']), "rb"))
        self.add_matrix_indices(params['flow'], params['index'],
            self.biosphere_dict)
        # Eliminate references to biosphere flows that don't appear in this
        # assessment; they are masked with MAX_INT_32 values
        self.cf_params = params[np.where(params['index'] != MAX_INT_32)]

    def build_technosphere_matrix(self, vector=None):
        vector = self.tech_params['amount'] if vector is None else vector
        count = len(self.technosphere_dict)
        indices = range(count)
        # Add ones along the diagonal
        data = np.hstack((-1 * vector, np.ones((count,))))
        rows = np.hstack((self.tech_params['row'], indices))
        cols = np.hstack((self.tech_params['col'], indices))
        # coo_matrix construction is coo_matrix((values, (rows, cols)),
        # (row_count, col_count))
        self.technosphere_matrix = CompressedSparseMatrixProxy(
            sparse.coo_matrix((data, (rows, cols)), (count, count)).tocsr(),
            self.technosphere_dict, self.technosphere_dict)

    def build_biosphere_matrix(self, vector=None):
        vector = self.bio_params['amount'] if vector is None else vector
        row_count = len(self.biosphere_dict)
        col_count = len(self.technosphere_dict)
        # coo_matrix construction is coo_matrix((values, (rows, cols)),
        # (row_count, col_count))
        self.biosphere_matrix = CompressedSparseMatrixProxy(
            sparse.coo_matrix((vector, (self.bio_params['row'],
            self.bio_params['col'])), (row_count, col_count)).tocsr(),
            self.biosphere_dict, self.technosphere_dict)

    def decompose_technosphere(self):
        self.solver = factorized(self.technosphere_matrix.data.tocsc())

    def build_demand_array(self, demand=None):
        demand = demand or self.demand
        self.demand_array = OneDimensionalArrayProxy(
            np.zeros(len(self.technosphere_dict)),
            self.technosphere_dict)
        for key in demand:
            self.demand_array[mapping[key]] = demand[key]

    def build_characterization_matrix(self, vector=None):
        vector = self.cf_params['amount'] if vector is None else vector
        count = len(self.biosphere_dict)
        self.characterization_matrix = CompressedSparseMatrixProxy(
            sparse.coo_matrix((vector, (self.cf_params['index'], self.cf_params['index'])),
            (count, count)).tocsr(),
            self.biosphere_dict, self.biosphere_dict)

    def build_dictionary(self, array):
        """Build a dictionary from the sorted, unique elements of an array"""
        return dicter(array)

    def add_matrix_indices(self, array_from, array_to, mapping):
        """Map ``array_from`` keys to ``array_to`` values using ``mapping``"""
        indexer(array_from, array_to, mapping)

    def solve_linear_system(self):
        """
Master solution function for linear system a*x=b.

This is separate from 'solve_system' so that it can be easily subclassed or
replaced. Can be passed the technosphere matrix and demand vector, but is
normally can be called directly; normal parameters are assumed.

Default uses UMFpack, and stores the context so that solving subsequent times
are quick. "To most numerical analysts, matrix inversion is a sin." - Nicolas
Higham, Accuracy and Stability of Numerical Algorithms, 2002, p. 260.
        """
        if hasattr(self, "solver"):
            return self.solver(self.demand_array.data)
        else:
            return spsolve(
                self.technosphere_matrix.data,
                self.demand_array.data)

    def lci(self, factorize=False):
        """Life cycle inventory"""
        self.load_databases()
        self.build_technosphere_matrix()
        self.build_biosphere_matrix()
        self.build_demand_array()
        if factorize:
            self.decompose_technosphere()
        self.lci_calculation()

    def lci_calculation(self):
        self.supply_array = OneDimensionalArrayProxy(
            self.solve_linear_system(),
            self.technosphere_dict)
        count = len(self.technosphere_dict)
        self.inventory = CompressedSparseMatrixProxy(
            self.biosphere_matrix.data * \
            sparse.spdiags([self.supply_array.data], [0], count, count),
            self.biosphere_dict, self.technosphere_dict)

    def redo_lci(self, demand):
        """Redo LCI with same databases but different demand"""
        assert hasattr(self, "inventory"), "Must do lci first"
        self.build_demand_array(demand)
        self.lci_calculation()

    def lcia(self):
        """Life cycle impact assessment"""
        assert hasattr(self, "inventory"), "Must do lci first"
        assert self.method, "Must specify a method to perform LCIA"
        self.load_method()
        self.build_characterization_matrix()
        self.lcia_calculation()

    def lcia_calculation(self):
        self.characterized_inventory = CompressedSparseMatrixProxy(
            self.characterization_matrix.data * self.inventory.data,
            self.biosphere_dict, self.technosphere_dict)

    def redo_lcia(self, demand=None):
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        if demand:
            self.redo_lci(demand)
        self.lcia_calculation()

    @property
    def score(self):
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        return float(self.characterized_inventory.sum())

    def reverse_dict(self):
        """Construct reverse dicts from row and col indices to processes"""
        rev_mapping = dict([(v, k) for k, v in mapping.iteritems()])
        rev_tech = dict([(v, rev_mapping[k]) for k, v in \
            self.technosphere_dict.iteritems()])
        rev_bio = dict([(v, rev_mapping[k]) for k, v in \
            self.biosphere_dict.iteritems()])
        return rev_tech, rev_bio
