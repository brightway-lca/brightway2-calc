import warnings

import numpy as np
import xarray

from . import PYPARDISO, UMFPACK, factorized
from .multi_lca import MultiLCA

if PYPARDISO:
    from pypardiso.pardiso_wrapper import PyPardisoSolver
else:
    PyPardisoSolver = None


class FastScoresOnlyMultiLCA(MultiLCA):
    """Use chunking and pre-calculate as much as possible to optimize speed for multiple LCA
    calculations.

    If using pardiso via pypardiso:

    - Feed multiple demands at once as a tensor into the solver function
    - Skip some identity checks on the technosphere matrix

    """

    def __init__(self, *args, chunk_size: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

        if chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {chunk_size}")

        if UMFPACK:
            warnings.warn(
                """Using UMFPACK - the speedups in `FastScoresOnlyMultiLCA` work much better when using PARDISO"""  # noqa: E501
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

    def __next__(self) -> xarray.DataArray:
        # The technosphere matrix will change, so remove any existing LU factorization
        if UMFPACK:
            if hasattr(self, "solver"):
                delattr(self, "solver")
        elif PYPARDISO:
            # This is global state in the pypardiso library - use built-in reset function
            from pypardiso.scipy_aliases import pypardiso_solver

            pypardiso_solver.free_memory()
        else:
            raise ValueError("No suitable solver installed")
        super().__next__()

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
        if not hasattr(self, "technosphere_matrix"):
            self._load_datapackages()
            self.build_precalculated()

        if PYPARDISO:
            return self._calculate_pardiso()
        elif UMFPACK:
            return self._calculate_umfpack()
        else:
            raise ValueError(
                "`FastScoresOnlyMultiLCA` only supported with PARDISO and UMFPACK solvers"
            )

    def _calculate_umfpack(self) -> xarray.DataArray:
        solver = factorized(self.technosphere_matrix.tocsc())
        self.supply_array = np.zeros((self.technosphere_matrix.shape[0], len(self.demand_arrays)))

        for index, (label, arr) in enumerate(self.demand_arrays.items()):
            self.supply_array[:, index] = solver(arr)

        lcia_array = np.vstack(list(self.precalculated.values()))
        scores = lcia_array @ self.supply_array

        self._set_scores(
            xarray.DataArray(
                scores,
                coords=[[str(x) for x in self.precalculated], list(self.demand_arrays)],
                dims=["LCIA", "processes"],
            )
        )
        return self._get_scores()

    def _calculate_pardiso(self) -> xarray.DataArray:
        demand_array = np.vstack([arr for arr in self.demand_arrays.values()]).T
        supply_arrays = []

        solver = PyPardisoSolver()
        solver.factorize(self.technosphere_matrix)

        num_chunks = demand_array.shape[1] // self.chunk_size + 1
        for demand_chunk in np.array_split(demand_array, num_chunks, axis=1):
            b = solver._check_b(self.technosphere_matrix, demand_chunk)
            solver.set_phase(33)
            supply_arrays.append(solver._call_pardiso(self.technosphere_matrix, b))

        self.supply_array = np.hstack(supply_arrays)

        lcia_array = np.vstack(list(self.precalculated.values()))
        scores = lcia_array @ self.supply_array

        self._set_scores(
            xarray.DataArray(
                scores,
                coords=[[str(x) for x in self.precalculated], list(self.demand_arrays)],
                dims=["LCIA", "processes"],
            )
        )
        return self._get_scores()

    def _get_scores(self) -> xarray.DataArray:
        if not hasattr(self, "_scores"):
            raise ValueError("Scores not calculated yet")
        return self._scores

    def _set_scores(self, arr: xarray.DataArray) -> None:
        self._scores = arr

    scores = property(fget=_get_scores, fset=_set_scores)
