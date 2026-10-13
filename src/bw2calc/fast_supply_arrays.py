from typing import Iterator, Union

import numpy as np
from scipy import sparse

from bw2calc import PYPARDISO, UMFPACK, factorized

if PYPARDISO:
    from pypardiso.pardiso_wrapper import PyPardisoSolver
else:
    PyPardisoSolver = None


class FastSupplyArraysMixin:
    """Mixin class for `LCA` which allows for fast calculations of multiple inventories.

    Requires either `scikits-umfpack` or `pypardiso`. Must be mixed into an `LCA` class."""

    chunk_size = 50

    def set_chunk_size(self, chunk_size: int) -> None:
        if chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {chunk_size}")

        self.chunk_size = chunk_size

    def calculate_supply_arrays(
        self, demands: Union[list[np.ndarray], sparse.spmatrix, sparse.sparray]
    ) -> np.ndarray:
        """Calculate multiple supply arrays in a single calculation.

        Much faster than individual calculations, especially when using PARDISO.

        ``demands`` is either a list of 1-d demand arrays, or a sparse matrix with dimensions
        ``[demands, products]``. A sparse matrix is only made dense one chunk of ``chunk_size``
        demands at a time.

        Returns a numpy array with dimensions `[process scaling amounts, demands]`. `demands` are
        given in the same order as in `demands`."""
        if PYPARDISO:
            return self._calculate_pardiso(demands)
        elif UMFPACK:
            return self._calculate_umfpack(demands)
        else:
            raise ValueError(
                "`FastSupplyArraysMixin` only supported with PARDISO and UMFPACK solvers"
            )

    def _demand_chunks(self, demands, chunk_size: int) -> Iterator[np.ndarray]:
        """Yield dense demand arrays with dimensions ``[products, chunk_size]``."""
        if sparse.issparse(demands):
            demands = sparse.csr_matrix(demands)
            for start in range(0, demands.shape[0], chunk_size):
                yield demands[start : start + chunk_size].toarray().T
        else:
            for start in range(0, len(demands), chunk_size):
                yield np.vstack(demands[start : start + chunk_size]).T

    def _calculate_umfpack(self, demands) -> np.ndarray:
        # There is no speed up here, but it's convenient to have the same API
        supply_array = np.zeros((self.technosphere_matrix.shape[0], demands_count(demands)))
        if not supply_array.shape[1]:
            # Return empty array with correct shape when no demands provided
            return supply_array

        solver = factorized(self.technosphere_matrix.tocsc())
        index = 0
        for chunk in self._demand_chunks(demands, self.chunk_size):
            for arr in chunk.T:
                supply_array[:, index] = solver(arr)
                index += 1

        return supply_array

    def _calculate_pardiso(self, demands) -> np.ndarray:
        if not demands_count(demands):
            # Return empty array with correct shape when no demands provided
            return np.zeros((self.technosphere_matrix.shape[0], 0))

        supply_arrays = []

        solver = PyPardisoSolver()
        solver.factorize(self.technosphere_matrix)

        for demand_chunk in self._demand_chunks(demands, self.chunk_size):
            b = solver._check_b(self.technosphere_matrix, demand_chunk)
            solver.set_phase(33)
            supply_arrays.append(solver._call_pardiso(self.technosphere_matrix, b))

        return np.hstack(supply_arrays)


def demands_count(demands) -> int:
    return demands.shape[0] if sparse.issparse(demands) else len(demands)
