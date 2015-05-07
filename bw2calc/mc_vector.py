# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from eight import *

from .monte_carlo import IterativeMonteCarlo
from .utils import extract_uncertainty_fields as euf
from stats_arrays.random import MCRandomNumberGenerator
import numpy as np


class ParameterVectorLCA(IterativeMonteCarlo):
    """A Monte Carlo class where all uncertain parameters are stored in a single large array.

    Useful for sensitivity analysis and easy manipulation."""
    def load_data(self):
        self.load_lci_data()
        positions = {
            "tech": (0, self.tech_params.shape[0]),
            "bio": (
                self.tech_params.shape[0],
                self.tech_params.shape[0] + self.bio_params.shape[0]
            )
        }
        params = (euf(self.tech_params), euf(self.bio_params))

        if self.lcia:
            self.load_lcia_data()
            positions["cf"] = (
                positions["bio"][1],
                positions["bio"][1] + self.cf_params.shape[0]
            )
            params = params + (euf(self.cf_params),)

        if self.weighting:
            self.load_weighting_data()
            positions["weighting"] = (
                positions["bio"][1],
                positions["bio"][1] + self.cf_params.shape[0]
            )
            params = params + (euf(self.weighting_params),)

        self.positions = positions
        self.params = np.hstack(params)
        self.rng = MCRandomNumberGenerator(self.params, seed=self.seed)

    def __call__(self, vector=None):
        return self.next(vector)

    def next(self, vector=None):
        """Generate a new Monte Carlo iteration."""
        if vector is not None:
            self.sample = vector
        else:
            self.sample = self.rng.next()

        self.rebuild_technosphere_matrix(self.sample[
            self.positions["tech"][0]:self.positions["tech"][1]
            ])
        self.rebuild_biosphere_matrix(self.sample[
            self.positions["bio"][0]:self.positions["bio"][1]
            ])
        if self.lcia:
            self.rebuild_characterization_matrix(self.sample[
                self.positions["cf"][0]:self.positions["cf"][1]
            ])
        if self.weighting:
            self.weighting_value = self.sample[
                self.positions["weighting"][0]:self.positions["weighting"][1]
            ]

        if not hasattr(self, "demand_array"):
            self.build_demand_array()

        self.lci_calculation()
        if self.lcia:
            self.lcia_calculation()
            if self.weighting:
                self.weighting_calculation()
            return self.score
        else:
            return self.supply_array
