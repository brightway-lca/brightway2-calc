from numpy.linalg import solve

from .lca import LCA


class DenseLCA(LCA):
    """Convert the `technosphere_matrix` to a numpy array and solve with `numpy.linalg`."""

    def solve_linear_system(self):
        return solve(self.technosphere_matrix.toarray(), self.demand_array)
