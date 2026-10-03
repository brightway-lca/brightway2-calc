import warnings
from typing import Optional

import numpy as np
import xarray
from scipy import sparse

from bw2calc import PYPARDISO, UMFPACK, factorized
from bw2calc.errors import InaccurateSolution
from bw2calc.fast_supply_arrays import FastSupplyArraysMixin
from bw2calc.multi_lca import MultiLCA
from bw2calc.utils import as_columns, relative_residual

if PYPARDISO:
    from pypardiso.pardiso_wrapper import PyPardisoError, PyPardisoSolver
else:
    PyPardisoError, PyPardisoSolver = None, None

DIRECTIONS = ("auto", "forward", "adjoint")
DEFAULT_DIRECTION_WARNING = """The default `direction` of `FastScoresOnlyMultiLCA` will change from "forward" to "auto" in a future release.
This calculation has more demands ({demands}) than impact category combinations ({categories}), so "auto" would use the faster adjoint direction.
The scores are the same, but the adjoint direction doesn't calculate `supply_array`.
Pass `direction="forward"` to keep the current behaviour, or `direction="auto"` to get the speedup now."""  # noqa: E501
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
    in the technosphere matrix, labelled with the integer product ids of the datapackages. It
    doesn't calculate supply arrays, so ``supply_array`` isn't available.

    Parameters
    ----------
    chunk_size : int
        Number of demands to solve at once in the forward direction with PARDISO.
    direction : str, optional
        One of ``"auto"``, ``"forward"``, or ``"adjoint"``. ``"auto"`` uses the adjoint direction
        when there are more demands than impact category combinations. The default is currently
        ``"forward"``, but will change to ``"auto"`` in a future release; a ``FutureWarning`` is
        raised if this would change the direction of a calculation. The adjoint direction
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
        direction: Optional[str] = None,
        residual_tolerance: Optional[float] = 1e-8,
        **kwargs,
    ):
        # Extract chunk_size before passing to super() to avoid it being consumed
        # by MultiLCA.__init__, then manually initialize mixin attributes
        super().__init__(*args, **kwargs)
        self.set_chunk_size(chunk_size)

        self.direction = "forward" if direction is None else direction
        self._default_direction = direction is None
        self.residual_tolerance = residual_tolerance

        if UMFPACK:
            warnings.warn(
                """Using UMFPACK - the speedups in `FastSupplyArraysMixin` work better when using PARDISO"""  # noqa: E501
            )

    @property
    def direction(self) -> str:
        return self._direction

    @direction.setter
    def direction(self, value: str) -> None:
        if value not in DIRECTIONS:
            raise ValueError(f"Invalid direction: {value}; must be one of {DIRECTIONS}")
        if value == "adjoint" and self._is_stochastic():
            raise NotImplementedError(
                "The adjoint direction doesn't support `use_arrays` or `use_distributions` yet"
            )
        self._direction = value
        self._default_direction = False

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

        # Don't leave results from an earlier calculation if this one fails or changes direction
        for attr in ("_scores", "supply_array", "product_scores"):
            if hasattr(self, attr):
                delattr(self, attr)

        lcia_array = np.vstack(list(self.precalculated.values()))

        if self._default_direction and self._auto_is_adjoint():
            warnings.warn(
                DEFAULT_DIRECTION_WARNING.format(
                    demands=len(self.demand_arrays), categories=len(self.precalculated)
                ),
                FutureWarning,
            )
            # Once per instance is enough
            self._default_direction = False

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

    def _auto_is_adjoint(self) -> bool:
        return not self._is_stochastic() and len(self.demand_arrays) > len(self.precalculated)

    def _use_adjoint(self) -> bool:
        if self.direction == "auto":
            return self._auto_is_adjoint()
        return self.direction == "adjoint"

    def _calculate_adjoint(self, lcia_array: np.ndarray) -> np.ndarray:
        """Solve ``A^T Y = (c^T B)^T`` for all impact category combinations at once.

        Sets ``self.product_scores`` and returns scores with dimensions ``[LCIA, demands]``."""
        transposed = self.technosphere_matrix.T
        rhs = np.ascontiguousarray(lcia_array.T)
        product_scores = self.solve_adjoint(rhs)

        if self.residual_tolerance is not None:
            residuals = relative_residual(transposed, product_scores, rhs)
            # Negated so that NaN residuals, e.g. from a singular matrix, also fail
            if not np.all(residuals <= self.residual_tolerance):
                worst = int(np.where(np.isfinite(residuals), residuals, np.inf).argmax())
                raise InaccurateSolution(
                    f"Adjoint solve for {list(self.precalculated)[worst]} has a relative residual "
                    f"of {residuals[worst]:.3e}, above the tolerance of "
                    f"{self.residual_tolerance:.1e}. The technosphere matrix could be singular "
                    "or badly conditioned."
                )

        # Use the integer ids even if `dicts.product` was remapped; `(database, code)` tuples
        # can't be used as coordinates
        reversed_product = {index: key for key, index in self.dicts.product.original.items()}
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
        # Sparse, as a dense `[demands, products]` copy is huge when scoring every product
        demand_matrix = sparse.vstack(
            [sparse.csr_matrix(arr) for arr in self.demand_arrays.values()], format="csr"
        )
        return (demand_matrix @ product_scores).T

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        """Solve the transposed technosphere system ``A^T Y = rhs``.

        ``rhs`` has dimensions ``[activities, columns]``, or is 1-d for a single column; returns
        an array with dimensions ``[products, columns]``.

        Uses its own PARDISO solver instance rather than the global one used by ``spsolve``, so
        the factorization of the transposed matrix doesn't replace the factorization of the
        technosphere matrix itself."""
        transposed = self.technosphere_matrix.T
        rhs = as_columns(rhs)
        if rhs.shape[0] != transposed.shape[0]:
            raise ValueError(
                f"`rhs` has {rhs.shape[0]} rows, but the technosphere matrix has "
                f"{transposed.shape[0]} activities"
            )

        if rhs.shape[1] == 0:
            return np.zeros((transposed.shape[1], 0))

        if PYPARDISO:
            matrix = transposed.tocsr()
            solver = PyPardisoSolver()
            try:
                # `solve` on its own does the analysis and factorization phases each time
                solver.factorize(matrix)
                solution = solver.solve(matrix, rhs)
            finally:
                try:
                    solver.free_memory(everything=True)
                except PyPardisoError:
                    # Nothing to release if factorization failed; don't hide the original error
                    pass
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
