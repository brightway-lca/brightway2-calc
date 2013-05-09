# -*- coding: utf-8 -*
from __future__ import division
from brightway2 import config as base_config
from brightway2 import databases, methods, mapping
from scipy.sparse.linalg import factorized, spsolve
from scipy import sparse
import numpy as np
import os
from .matrices import MatrixBuilder
from .matrices import TechnosphereBiosphereMatrixBuilder as TBMBuilder


class LCA(object):
    #############
    ### Setup ###
    #############

    def __init__(self, demand, method=None, config=None):
        self.dirpath = (config or base_config).dir
        if isinstance(demand, (basestring, tuple, list)):
            raise ValueError("Demand must be a dictionary")
        self.demand = demand
        self.method = method
        self.databases = self.get_databases(demand)

    def get_databases(self, demand):
        """Get list of databases for functional unit"""
        return set.union(
            *[set(databases[key[0]]["depends"] + [key[0]]) for key in demand])

    def build_demand_array(self, demand=None):
        demand = demand or self.demand
        self.demand_array = np.zeros(len(self.technosphere_dict))
        for key in demand:
            self.demand_array[self.technosphere_dict[mapping[key]]] = \
                demand[key]

    #########################
    ### Data manipulation ###
    #########################

    def fix_dictionaries(self):
        """
Fix technosphere and biosphere dictionaries from this:

.. code-block:: python

    {mapping integer id: matrix row/column index}

To this:

.. code-block:: python

    {(database, key): matrix row/column index}

        """
        rev_mapping = {v: k for k, v in mapping.iteritems()}
        self.technosphere_dict = {
            rev_mapping[k]: v for k, v in self.technosphere_dict.iteritems()}
        self.biosphere_dict = {
            rev_mapping[k]: v for k, v in self.biosphere_dict.iteritems()}

    def reverse_dict(self):
        """Construct reverse dicts from row and col indices to processes"""
        rev_tech = {v: k for k, v in self.technosphere_dict.iteritems()}
        rev_bio = {v: k for k, v in self.biosphere_dict.iteritems()}
        return rev_tech, rev_bio

    ######################
    ### Data retrieval ###
    ######################

    def load_lci_data(self, builder=TBMBuilder):
        self.bio_params, self.tech_params, \
            self.biosphere_dict, self.technosphere_dict, \
            self.biosphere_matrix, self.technosphere_matrix = \
            builder.build(self.dirpath, self.databases)

    def load_lcia_data(self, builder=MatrixBuilder):
        self.cf_params, d, d, self.characterization_matrix = builder.build(
            self.dirpath, [methods[self.method]['abbreviation']],
            "amount", "flow", "index", row_dict=self.biosphere_dict,
            one_d=True)

    ####################
    ### Calculations ###
    ####################

    def decompose_technosphere(self):
        self.solver = factorized(self.technosphere_matrix.tocsc())

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
            return self.solver(self.demand_array)
        else:
            return spsolve(
                self.technosphere_matrix,
                self.demand_array)

    def lci(self, factorize=False,
            builder=TBMBuilder):
        """Life cycle inventory"""
        self.load_lci_data(builder)
        self.build_demand_array()
        if factorize:
            self.decompose_technosphere()
        self.lci_calculation()

    def lci_calculation(self):
        self.supply_array = self.solve_linear_system()
        count = len(self.technosphere_dict)
        self.inventory = self.biosphere_matrix * \
            sparse.spdiags([self.supply_array], [0], count, count)

    def lcia(self, builder=MatrixBuilder):
        """Life cycle impact assessment"""
        assert hasattr(self, "inventory"), "Must do lci first"
        assert self.method, "Must specify a method to perform LCIA"
        self.load_lcia_data(builder)
        self.lcia_calculation()

    def lcia_calculation(self):
        self.characterized_inventory = \
            self.characterization_matrix * self.inventory

    @property
    def score(self):
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        return float(self.characterized_inventory.sum())

    #########################
    ### Redo calculations ###
    #########################

    def rebuild_technosphere_matrix(self, vector):
        self.technosphere_matrix = MatrixBuilder.build_matrix(
            self.tech_params, self.technosphere_dict, self.technosphere_dict,
            "row", "col", new_data=vector)

    def rebuild_biosphere_matrix(self, vector):
        self.biosphere_matrix = MatrixBuilder.build_matrix(
            self.bio_params, self.biosphere_dict, self.technosphere_dict,
            "row", "col", new_data=vector)

    def rebuild_characterization_matrix(self, vector):
        self.characterization_matrix = MatrixBuilder.build_diagonal_matrix(
            self.cf_params, self.biosphere_dict,
            "index", "index", new_data=vector)

    def redo_lci(self, demand):
        """Redo LCI with same databases but different demand"""
        assert hasattr(self, "inventory"), "Must do lci first"
        self.build_demand_array(demand)
        self.lci_calculation()

    def redo_lcia(self, demand=None):
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        if demand:
            self.redo_lci(demand)
        self.lcia_calculation()
