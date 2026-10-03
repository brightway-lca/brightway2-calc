import warnings
from typing import Optional

import numpy as np
import xarray

from bw2calc import PYPARDISO, UMFPACK, factorized
from bw2calc.errors import InaccurateSolution
from bw2calc.fast_supply_arrays import FastSupplyArraysMixin
from bw2calc.multi_lca import MultiLCA
from bw2calc.utils import relative_residual

if PYPARDISO:
    from pypardiso.pardiso_wrapper import PyPardisoSolver
else:
    PyPardisoSolver = None

DIRECTIONS = ("auto", "forward", "adjoint")
STOCHASTIC_MATRICES = (
    "technosphere_matrix",
    "biosphere_matrix",
    "characterization_matrix",
    "normalization_matrix",
    "weighting_matrix",
)


class FastScoresOnlyMultiLCA(MultiLCA, FastSupplyArraysMixin):
    """Use chunking and pre-calculate as much as possible to optimize speed for multiple LCA
    calculations.

    If using pardiso via pypardiso:

    - Feed multiple demands at once as a tensor into the solver function
    - Skip some identity checks on the technosphere matrix

    Scores can be calculated in two directions, which give the same results:

    - **forward**: solve ``A s = f`` once per demand, then multiply each supply array by the
      characterized biosphere rows ``c^T B``. Cost scales with the number of demands.
    - **adjoint**: solve the transposed system ``A^T Y = B^T C`` once per impact category (or
      normalization and weighting combination). Column ``k`` of ``Y`` is the score of one unit of
      *every* product for impact category ``k``, so the score of any demand ``f`` is ``f^T Y``.
      Cost scales with the number of impact categories, and is much faster when there are many
      more demands than impact categories, e.g. scoring every product in a database.

    The adjoint direction also stores ``product_scores``, the score of one unit of every product
    in the technosphere matrix. It doesn't calculate supply arrays, so ``supply_array`` isn't
    available.

    Parameters
    ----------
    chunk_size : int
        Number of demands to solve at once in the forward direction with PARDISO.
    direction : str
        One of ``"auto"``, ``"forward"``, or ``"adjoint"``. ``"auto"`` uses the adjoint direction
        when there are more demands than impact category combinations. The adjoint direction
        doesn't yet support Monte Carlo or other iterated calculations (``use_arrays`` or
        ``use_distributions``); ``"auto"`` will use the forward direction in this case.
    residual_tolerance : float, optional
        Maximum relative residual of the adjoint solve before raising ``InaccurateSolution``.
        ``None`` disables the check.

    """

    def __init__(
        self,
        *args,
        chunk_size: int = 50,
        direction: str = "auto",
        residual_tolerance: Optional[float] = 1e-8,
        **kwargs,
    ):
        # Extract chunk_size before passing to super() to avoid it being consumed
        # by MultiLCA.__init__, then manually initialize mixin attributes
        super().__init__(*args, **kwargs)
        self.set_chunk_size(chunk_size)

        if direction not in DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}; must be one of {DIRECTIONS}")
        if direction == "adjoint" and self._is_stochastic():
            raise NotImplementedError(
                "The adjoint direction doesn't support `use_arrays` or `use_distributions` yet"
            )
        self.direction = direction
        self.residual_tolerance = residual_tolerance

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

        lcia_array = np.vstack(list(self.precalculated.values()))

        if self._use_adjoint():
            scores = self._calculate_adjoint(lcia_array)
        else:
            self.supply_array = self.calculate_supply_arrays(list(self.demand_arrays.values()))
            scores = lcia_array @ self.supply_array

        self._set_scores(
            xarray.DataArray(
                scores,
                coords=[[str(x) for x in self.precalculated], list(self.demand_arrays)],
                dims=["LCIA", "processes"],
            )
        )
        return self._scores

    def _is_stochastic(self) -> bool:
        """Will matrix values change between iterations?"""
        return any(any(self.check_selective_use(label)) for label in STOCHASTIC_MATRICES)

    def _use_adjoint(self) -> bool:
        if self.direction == "auto":
            return not self._is_stochastic() and len(self.demand_arrays) > len(self.precalculated)
        return self.direction == "adjoint"

    def _calculate_adjoint(self, lcia_array: np.ndarray) -> np.ndarray:
        """Solve ``A^T Y = (c^T B)^T`` for all impact category combinations at once.

        Sets ``self.product_scores`` and returns scores with dimensions ``[LCIA, demands]``."""
        transposed = self.technosphere_matrix.T
        rhs = np.ascontiguousarray(lcia_array.T)
        product_scores = self.solve_adjoint(rhs)

        if self.residual_tolerance is not None:
            residuals = relative_residual(transposed, product_scores, rhs)
            if residuals.max(initial=0) > self.residual_tolerance:
                worst = int(residuals.argmax())
                raise InaccurateSolution(
                    f"Adjoint solve for {list(self.precalculated)[worst]} has a relative residual "
                    f"of {residuals[worst]:.3e}, above the tolerance of "
                    f"{self.residual_tolerance:.1e}. The technosphere matrix could be singular "
                    "or badly conditioned."
                )

        # Don't leave stale results from an earlier forward calculation
        if hasattr(self, "supply_array"):
            delattr(self, "supply_array")

        reversed_product = self.dicts.product.reversed
        self.product_scores = xarray.DataArray(
            product_scores.T,
            coords=[
                [str(x) for x in self.precalculated],
                [reversed_product[index] for index in range(product_scores.shape[0])],
            ],
            dims=["LCIA", "products"],
        )

        if not self.demand_arrays:
            return np.zeros((lcia_array.shape[0], 0))
        demand_matrix = np.vstack(list(self.demand_arrays.values()))
        return (demand_matrix @ product_scores).T

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        """Solve the transposed technosphere system ``A^T Y = rhs``.

        ``rhs`` has dimensions ``[activities, columns]``; returns an array with dimensions
        ``[products, columns]``.

        Uses its own PARDISO solver instance rather than the global one used by ``spsolve``, so
        the factorization of the transposed matrix doesn't replace the factorization of the
        technosphere matrix itself."""
        rhs = np.asarray(rhs).reshape(rhs.shape[0], -1)
        transposed = self.technosphere_matrix.T

        if rhs.shape[1] == 0:
            return np.zeros((transposed.shape[1], 0))

        if PYPARDISO:
            matrix = transposed.tocsr()
            solver = PyPardisoSolver()
            # `solve` on its own does the analysis and factorization phases each time
            solver.factorize(matrix)
            try:
                solution = solver.solve(matrix, rhs)
            finally:
                solver.free_memory(everything=True)
        elif UMFPACK:
            solve = factorized(transposed.tocsc())
            solution = np.column_stack([solve(column) for column in rhs.T])
        else:
            raise ValueError(
                "`FastScoresOnlyMultiLCA` only supported with PARDISO and UMFPACK solvers"
            )

        # pypardiso squeezes its result, which drops dimensions for a single column or product
        return np.asarray(solution).reshape(transposed.shape[1], rhs.shape[1])

    def _get_scores(self) -> xarray.DataArray:
        if not hasattr(self, "_scores"):
            raise ValueError("Scores not calculated yet")
        return self._scores

    def _set_scores(self, arr: xarray.DataArray) -> None:
        self._scores = arr

    scores = property(fget=_get_scores, fset=_set_scores)
