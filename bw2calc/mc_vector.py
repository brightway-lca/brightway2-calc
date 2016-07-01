# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from eight import *
from future.utils import implements_iterator

from .monte_carlo import IterativeMonteCarlo
from .utils import extract_uncertainty_fields as euf
from stats_arrays.random import MCRandomNumberGenerator
import numpy as np


@implements_iterator
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

        # `euf` is extract_uncertainty_fields; needed because we are
        # "gluing" together arrays with different column numbers and labels
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

    def rebuild_all(self, vector=None):
        """Rebuild the LCI/LCIA matrices from a new Monte Carlo sample or provided vector."""
        if not hasattr(self, "positions"):
            self.load_data()

        if vector is not None and not isinstance(vector, np.ndarray):
            raise ValueError("`vector` must be a 1-d numpy array")

        if vector is not None:
            assert vector.shape == self.params.shape, \
                "Incorrect `vector` shape. Is {}, but should be {}".format(
                    vector.shape, self.params.shape
                )
        # Copy to break references and avoid later manipulation by RNG
        self.sample = (self.rng.next() if vector is None else vector).copy()
        self.rebuild_technosphere_matrix(self.tech_sample)
        self.rebuild_biosphere_matrix(self.bio_sample)
        if self.lcia:
            self.rebuild_characterization_matrix(self.cf_sample)
        if self.weighting:
            self.weighting_value = self.weighting_sample

    def __next__(self):
        """Generate a new Monte Carlo iteration."""
        self.rebuild_all()

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

    @property
    def tech_sample(self):
        return self.sample[self.positions["tech"][0]:self.positions["tech"][1]]

    @property
    def bio_sample(self):
        return self.sample[self.positions["bio"][0]:self.positions["bio"][1]]

    @property
    def cf_sample(self):
        return self.sample[self.positions["cf"][0]:self.positions["cf"][1]]

    @property
    def weighting_sample(self):
        return self.sample[self.positions["weighting"][0]:self.positions["weighting"][1]]
