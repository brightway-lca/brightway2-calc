from typing import Optional

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator, gmres

from bw2calc.lca import LCA


class JacobiGMRESLCA(LCA):
    """Solve ``Ax=b`` with GMRES using a Jacobi preconditioner.

    The preconditioner is the inverse of the technosphere diagonal, i.e. ``D^-1``.
    This prior decomposition can significantly improve convergence for certain types of
    matrices, especially those with dominant diagonal entries.

    :param demand: Functional unit mapping passed through to :class:`bw2calc.lca.LCA`.
    :type demand: dict
    :param data_objs: Datapackages passed through to :class:`bw2calc.lca.LCA`.
    :type data_objs: iterable
    :param rtol:
        Relative tolerance for GMRES convergence. Convergence is checked against a threshold
        comparable to ``max(rtol * ||b||, atol)``.
    :type rtol: float
    :param atol: Absolute tolerance floor for GMRES convergence.
    :type atol: float
    :param restart: Number of iterations between GMRES restarts. ``None`` uses SciPy defaults.
    :type restart: int or None
    :param maxiter: Maximum number of outer GMRES iterations.
    :type maxiter: int or None
    :param use_guess:
        If ``True``, reuse the previous solution as ``x0`` for subsequent solves in the same
        instance.
    :type use_guess: bool
    """

    def __init__(
        self,
        *args,
        rtol: float = 1e-8,
        atol: float = 0.0,
        restart: Optional[int] = 50,
        maxiter: Optional[int] = 300,
        use_guess: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # GMRES convergence controls.
        self.rtol = rtol
        self.atol = atol
        self.restart = restart
        self.maxiter = maxiter
        # When enabled, reuse the previous solution as GMRES initial guess (x0).
        self.use_guess = use_guess
        # Cache whether matrix structure cleanup was already done.
        self._matrix_prepared = False
        # Cache the Jacobi preconditioner to avoid rebuilding between solves.
        self._cached_preconditioner: Optional[LinearOperator] = None
        # Last successful solution vector, used as warm start when `use_guess=True`.
        self.guess = None

    def __next__(self) -> None:
        # Matrix values can change across iteration steps, so invalidate caches.
        self._matrix_prepared = False
        self._cached_preconditioner = None
        super().__next__()

    def load_lci_data(self, nonsquare_ok=False) -> None:
        super().load_lci_data(nonsquare_ok=nonsquare_ok)
        # New matrices imply stale solver-side caches.
        self._matrix_prepared = False
        self._cached_preconditioner = None
        self.guess = None

    def _prepare_matrix(self) -> None:
        # Sparse cleanup is done once per matrix build, then reused.
        if self._matrix_prepared:
            return
        if not sps.isspmatrix(self.technosphere_matrix):
            raise TypeError("technosphere_matrix must be a SciPy sparse matrix")

        # GMRES works best with canonical sparse structure.
        self.technosphere_matrix = self.technosphere_matrix.tocsc(copy=False)
        self.technosphere_matrix.sum_duplicates()
        self.technosphere_matrix.eliminate_zeros()
        self.technosphere_matrix.sort_indices()
        self._matrix_prepared = True

    def _build_jacobi_preconditioner(self) -> Optional[LinearOperator]:
        # Reuse preconditioner when solving multiple demands on same matrix.
        if self._cached_preconditioner is not None:
            return self._cached_preconditioner

        diagonal = self.technosphere_matrix.diagonal()
        # Cannot build Jacobi inverse if any diagonal entry is zero.
        if np.any(diagonal == 0):
            return None

        inverse_diagonal = 1.0 / diagonal
        # LinearOperator form avoids materializing a dense diagonal inverse matrix.
        self._cached_preconditioner = LinearOperator(
            shape=self.technosphere_matrix.shape,
            matvec=lambda x: inverse_diagonal * x,
            dtype=self.technosphere_matrix.dtype,
        )
        return self._cached_preconditioner

    def solve_linear_system(self, demand: Optional[np.ndarray] = None) -> np.ndarray:
        if demand is None:
            demand = self.demand_array

        self._prepare_matrix()
        preconditioner = self._build_jacobi_preconditioner()
        # Warm start can reduce Krylov iterations for related successive solves.
        x0 = self.guess if (self.use_guess and self.guess is not None) else None

        try:
            # SciPy modern API (`rtol` + `atol`).
            solution, _ = gmres(
                self.technosphere_matrix,
                demand,
                x0=x0,
                rtol=self.rtol,
                atol=self.atol,
                restart=self.restart,
                maxiter=self.maxiter,
                M=preconditioner,
            )
        except TypeError:
            # Backward compatibility for SciPy versions using `tol`.
            solution, _ = gmres(
                self.technosphere_matrix,
                demand,
                x0=x0,
                tol=self.rtol,
                atol=self.atol,
                restart=self.restart,
                maxiter=self.maxiter,
                M=preconditioner,
            )

        # Match return conventions used elsewhere in bw2calc.
        solution = np.asarray(solution)
        if not solution.shape:
            solution = solution.reshape((1,))

        if self.use_guess:
            # Keep latest solution for the next warm-started call.
            self.guess = solution

        return solution
