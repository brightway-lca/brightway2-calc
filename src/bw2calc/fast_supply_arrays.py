import numpy as np

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

    def calculate_supply_arrays(self, demand_arrays: list[np.ndarray]) -> np.ndarray:
        """Calculate multiple supply arrays in a single calculation.

        Much faster than individual calculations, especially when using PARDISO.

        Returns a numpy array with dimensions `[process scaling amounts, demands]`. `demands` are
        given in the same order as `demand_arrays`."""
        if PYPARDISO:
            return self._calculate_pardiso(demand_arrays)
        elif UMFPACK:
            return self._calculate_umfpack(demand_arrays)
        else:
            raise ValueError(
                "`FastSupplyArraysMixin` only supported with PARDISO and UMFPACK solvers"
            )

    def _calculate_umfpack(self, demands: list[np.ndarray]) -> np.ndarray:
        # There is no speed up here, but it's convenient to have the same API
        solver = factorized(self.technosphere_matrix.tocsc())
        supply_array = np.zeros((self.technosphere_matrix.shape[0], len(demands)))

        for index, arr in enumerate(demands):
            supply_array[:, index] = solver(arr)

        return supply_array

    def _calculate_pardiso(self, demands: list[np.ndarray]) -> np.ndarray:
        demand_array = np.vstack([arr for arr in demands]).T
        supply_arrays = []

        solver = PyPardisoSolver()
        solver.factorize(self.technosphere_matrix)

        num_chunks = demand_array.shape[1] // self.chunk_size + 1
        for demand_chunk in np.array_split(demand_array, num_chunks, axis=1):
            b = solver._check_b(self.technosphere_matrix, demand_chunk)
            solver.set_phase(33)
            supply_arrays.append(solver._call_pardiso(self.technosphere_matrix, b))

        return np.hstack(supply_arrays)
