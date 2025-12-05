import warnings

import numpy as np
import xarray

from bw2calc.fast_supply_arrays import FastSupplyArraysMixin

from . import PYPARDISO, UMFPACK
from .multi_lca import MultiLCA


class FastScoresOnlyMultiLCA(MultiLCA, FastSupplyArraysMixin):
    """Use chunking and pre-calculate as much as possible to optimize speed for multiple LCA
    calculations.

    If using pardiso via pypardiso:

    - Feed multiple demands at once as a tensor into the solver function
    - Skip some identity checks on the technosphere matrix

    """

    def __init__(self, *args, chunk_size: int = 50, **kwargs):
        # Extract chunk_size before passing to super() to avoid it being consumed
        # by MultiLCA.__init__, then manually initialize mixin attributes
        super().__init__(*args, **kwargs)
        self.set_chunk_size(chunk_size)

        if UMFPACK:
            warnings.warn(
                """Using UMFPACK - the speedups in `FastSupplyArraysMixin` work better when using PARDISO"""  # noqa: E501
            )

    def lci(self) -> None:
        raise NotImplementedError(
            "LCI and LCIA aren't separate in `FastScoresOnlyMultiLCA`; use `next()` to calculate scores."  # noqa: E501
        )

    def lci_calculation(self) -> None:
        raise NotImplementedError(
            "LCI and LCIA aren't separate in `FastScoresOnlyMultiLCA`; use `next()` to calculate scores."  # noqa: E501
        )

    def lcia(self) -> None:
        raise NotImplementedError(
            "LCI and LCIA aren't separate in `FastScoresOnlyMultiLCA`; use `next()` to calculate scores."  # noqa: E501
        )

    def lcia_calculation(self) -> None:
        raise NotImplementedError(
            "LCI and LCIA aren't separate in `FastScoresOnlyMultiLCA`; use `next()` to calculate scores."  # noqa: E501
        )

    def build_precalculated(self) -> None:
        """Multiply the characterization, and normalization and weighting matrices if present, by
        the biosphere matrix. When done outside the calculation loop, this only needs to be done
        once."""
        self.precalculated = self.characterization_matrices @ self.biosphere_matrix
        if hasattr(self, "normalization_matrices"):
            self.precalculated = self.normalization_matrices @ self.precalculated
        if hasattr(self, "weighting_matrices"):
            self.precalculated = self.weighting_matrices @ self.precalculated
        self.precalculated = {
            key: np.asarray(matrix.sum(axis=0)) for key, matrix in self.precalculated.items()
        }

    def _calculation(self) -> xarray.DataArray:
        # Calls lci_calculation() and lcia_calculation in parent class, but we don't have
        # these as separate methods, so need to override to change behaviour.
        return self.calculate()

    def _load_datapackages(self) -> None:
        self.load_lci_data()
        self.build_demand_array()
        self.load_lcia_data()
        if self.config.get("normalizations"):
            self.load_normalization_data()
        if self.config.get("weightings"):
            self.load_weighting_data()

    def calculate(self) -> xarray.DataArray:
        """The actual LCI calculation.

        Separated from ``lci`` to be reusable in cases where the matrices are already built, e.g.
        ``redo_lci`` and Monte Carlo classes.

        """
        if not (PYPARDISO or UMFPACK):
            raise ValueError(
                "`FastScoresOnlyMultiLCA` only supported with PARDISO and UMFPACK solvers"
            )

        if not hasattr(self, "technosphere_matrix"):
            self._load_datapackages()
            self.build_precalculated()

        self.supply_array = self.calculate_supply_arrays(list(self.demand_arrays.values()))

        lcia_array = np.vstack(list(self.precalculated.values()))
        scores = lcia_array @ self.supply_array

        self._set_scores(
            xarray.DataArray(
                scores,
                coords=[[str(x) for x in self.precalculated], list(self.demand_arrays)],
                dims=["LCIA", "processes"],
            )
        )
        return self._scores

    def _get_scores(self) -> xarray.DataArray:
        if not hasattr(self, "_scores"):
            raise ValueError("Scores not calculated yet")
        return self._scores

    def _set_scores(self, arr: xarray.DataArray) -> None:
        self._scores = arr

    scores = property(fget=_get_scores, fset=_set_scores)
