from typing import Optional

import numpy as np
from scipy.sparse.linalg import cgs

from . import spsolve
from .lca import LCA


class IterativeLCA(LCA):
    """
    Solve `Ax=b` using iterative techniques instead of
    [LU factorization](http://en.wikipedia.org/wiki/LU_decomposition).
    """

    def __init__(self, *args, iter_solver=cgs, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_solver = iter_solver
        self.guess = None

    def solve_linear_system(self, demand: Optional[np.ndarray] = None) -> None:
        if demand is None:
            demand = self.demand_array
        if not self.iter_solver or self.guess is None:
            self.guess = spsolve(self.technosphere_matrix, demand)
            if not self.guess.shape:
                self.guess = self.guess.reshape((1,))
            return self.guess
        else:
            solution, status = self.iter_solver(
                self.technosphere_matrix,
                demand,
                x0=self.guess,
                atol="legacy",
                maxiter=1000,
            )
            if status != 0:
                return spsolve(self.technosphere_matrix, demand)
            return solution
