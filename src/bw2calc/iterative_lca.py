import logging
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator, bicgstab

from bw2calc.lca import LCA

logger = logging.getLogger("bw2calc")


class IterativeLCA(LCA):
    """Solve ``Ax=b`` using iterative techniques instead of
    `LU factorization <http://en.wikipedia.org/wiki/LU_decomposition>`_.

    Any solver with the signature of the ``scipy.sparse.linalg`` iterative solvers can be
    used, i.e. ``solver(A, b, x0=..., rtol=..., atol=..., maxiter=..., M=...)`` returning
    ``(solution, info)``. The default is
    `BiCGSTAB <https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method>`_
    with a Jacobi preconditioner, i.e. the inverse of the technosphere diagonal ``D^-1``.

    Benchmarks and choice of default solver
    ---------------------------------------

    Monte Carlo on ecoinvent 3.10.1 cutoff (23,523 x 23,523, 293k nonzeros), demand 1 kWh
    ``market for electricity, low voltage`` (CH), CML v4.8 GWP100, ``use_distributions=True``.
    Apple M1 Max, Python 3.12, SciPy 1.18.1, scikit-umfpack. pypardiso isn't available on
    ARM, so the direct baseline is UMFPACK; on x86 with pypardiso the speedups will be
    smaller. See https://github.com/brightway-lca/brightway2-calc/pull/161.

    Full Monte Carlo iterations (30 iterations, each including about 65 ms of sampling and
    matrix rebuilding):

    ==============================================  ===============  ====================
    Class                                           ms / iteration   Max rel. score diff.
    ==============================================  ===============  ====================
    ``LCA`` (UMFPACK)                               387              --
    ``JacobiGMRESLCA``                              153              9e-7
    BiCGSTAB + Jacobi, warm start, residual check   84               3e-7
    ==============================================  ===============  ====================

    Solve time only (median over 20 Monte Carlo technosphere matrices, ``rtol=1e-8``,
    errors are the maximum over the samples relative to UMFPACK):

    ============================================  ==========  ============  ===============
    Solver                                        ms / solve  Rel. error x  Rel. score err.
    ============================================  ==========  ============  ===============
    UMFPACK ``spsolve`` (CSR input)               350         --            --
    GMRES + Jacobi, no ``x0``                     126         3e-8          7e-7
    GMRES + Jacobi, ``x0`` = static solution      95          4e-8          7e-7
    LGMRES + Jacobi, ``x0`` = static              100         2e-8          3e-7
    GCROT(m,k) + Jacobi, ``x0`` = static          122         3e-8          7e-7
    **BiCGSTAB + Jacobi, x0 = static**            **33**      2e-8          3e-7
    TFQMR + Jacobi, ``x0`` = static               40          **2e-1**      **5e-2**
    GMRES, preconditioner = static LU             136         9e-8          1e-6
    BiCGSTAB, preconditioner = static LU          118         1e-7          1e-6
    Plain iteration with static LU                103         1e-7          1e-6
    GMRES / BiCGSTAB + static ``spilu``           26k--99k    --            no convergence
    PETSc BiCGSTAB + ILU(0)                       32          1e-8          1e-7
    PETSc BiCGSTAB + Jacobi                       36          2e-8          3e-7
    ============================================  ==========  ============  ===============

    * BiCGSTAB + Jacobi is about 10x faster than UMFPACK per solve, and about 3x faster
      than GMRES + Jacobi. BiCGSTAB does fixed work per iteration (two matrix-vector
      products, constant memory), while GMRES orthogonalizes against a growing basis, has
      to restart, and runs a Python-level inner loop in SciPy.
    * Jacobi suits the technosphere matrix: dividing by the production amounts on the
      diagonal gives ``D^-1 A = I - T'`` with the eigenvalues of ``T'`` well inside the
      unit circle, so Krylov methods converge in a few dozen iterations. It costs one
      elementwise division per iteration and is trivially rebuilt for every sample.
    * Warm starts help only modestly (about 25% for GMRES). Monte Carlo matrices differ a
      lot from the static one, but the static solution is only about 4% from each sampled
      solution, so a good ``x0`` saves just the first few iterations.
    * Reusing the static LU as a preconditioner converges in 3-4 iterations, but each
      application costs two triangular solves (about 27 ms). Static ``spilu`` is singular
      or useless. Subspace recycling (LGMRES, GCROT(m,k)) gave no gain.
    * PETSc is marginally faster, but a heavy dependency for about 1 ms per solve.
    * Not yet checked: ecoinvent 3.12, other demands and databases (where small diagonals
      or strong loops could slow Jacobi convergence), the x86 + pypardiso baseline, and
      ``rtol`` / ``restart`` sweeps.

    Correctness checks
    ------------------

    Convergence flags can't be trusted: TFQMR reported ``info == 0`` with a solution 20%
    off, and BiCGSTAB can break down (``info < 0``), stagnate, or return non-finite values.
    Every iterative solution is therefore checked by computing the true residual
    ``||Ax - b||`` against ``residual_factor * max(rtol * ||b||, atol)``. The slack given
    by ``residual_factor`` allows for solvers which test convergence on a preconditioned
    residual. If the solver doesn't converge, or the solution fails the check, the system
    is solved again with the direct solver, and a debug message is logged on the
    ``bw2calc`` logger. In the benchmark above, this caught one BiCGSTAB result in 30
    Monte Carlo iterations.

    Warm starts
    -----------

    With ``use_guess=True``, the previous solution is used as the initial guess ``x0`` for
    the next solve. With ``direct_first_solve=True``, the first solve uses the direct
    solver, so that the iterative solver always starts from the static solution.

    Subclasses can change the preconditioner by overriding :meth:`build_preconditioner`
    (return ``None`` for no preconditioning), and add solver arguments by overriding
    :meth:`solver_kwargs`.

    :param demand: Functional unit mapping passed through to :class:`bw2calc.lca.LCA`.
    :type demand: dict
    :param data_objs: Datapackages passed through to :class:`bw2calc.lca.LCA`.
    :type data_objs: iterable
    :param iter_solver: Iterative solver function. Defaults to
        :func:`scipy.sparse.linalg.bicgstab`.
    :type iter_solver: callable
    :param rtol: Relative tolerance for convergence.
    :type rtol: float
    :param atol: Absolute tolerance floor for convergence.
    :type atol: float
    :param maxiter: Maximum number of solver iterations.
    :type maxiter: int or None
    :param use_guess: If ``True``, use the previous solution as ``x0`` for the next solve.
    :type use_guess: bool
    :param direct_first_solve: If ``True``, use the direct solver when there is no guess
        yet.
    :type direct_first_solve: bool
    :param residual_factor: Slack factor for the residual check on iterative solutions.
    :type residual_factor: float
    """

    def __init__(
        self,
        *args,
        iter_solver: Optional[Callable] = None,
        rtol: float = 1e-8,
        atol: float = 0.0,
        maxiter: Optional[int] = 1000,
        use_guess: bool = True,
        direct_first_solve: bool = True,
        residual_factor: float = 10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.iter_solver = iter_solver if iter_solver is not None else bicgstab
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.use_guess = use_guess
        self.direct_first_solve = direct_first_solve
        self.residual_factor = residual_factor
        # Prepared CSC copy used by the iterative solver; don't replace
        # `technosphere_matrix`, as Monte Carlo iteration mutates the matrix held by
        # `technosphere_mm`. `None` means "not prepared yet".
        self._prepared_technosphere_matrix = None
        # Cache the preconditioner to avoid rebuilding between solves.
        self._cached_preconditioner: Optional[LinearOperator] = None
        # Last solution vector, used as warm start when `use_guess=True`.
        self.guess = None

    def __next__(self) -> None:
        # Matrix values can change across iteration steps, so invalidate caches.
        self._clear_matrix_caches()
        super().__next__()

    def load_lci_data(self, nonsquare_ok=False) -> None:
        super().load_lci_data(nonsquare_ok=nonsquare_ok)
        # New matrices imply stale solver-side caches.
        self._clear_matrix_caches()
        self.guess = None

    def _clear_matrix_caches(self) -> None:
        self._prepared_technosphere_matrix = None
        self._cached_preconditioner = None

    def _prepare_matrix(self) -> None:
        # Sparse cleanup is done once per matrix build, then reused.
        if self._prepared_technosphere_matrix is not None:
            return
        if not sps.issparse(self.technosphere_matrix):
            raise TypeError("technosphere_matrix must be a SciPy sparse matrix")

        # Iterative solvers work best with canonical sparse structure. CSR is fine for
        # the matrix-vector products they need, and is what `LCA` builds. Always copy:
        # with `copy=False`, a `technosphere_matrix` which is already CSR would be
        # returned as-is, and `eliminate_zeros()` would then strip structural zeros from
        # the matrix owned by `technosphere_mm`, which needs them to update in place.
        matrix = self.technosphere_matrix.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        self._prepared_technosphere_matrix = matrix

    def build_preconditioner(self) -> Optional[LinearOperator]:
        """Return a preconditioner ``M`` approximating ``A^-1``, or ``None`` for no
        preconditioning. The result is cached until the technosphere matrix changes.

        The default is the Jacobi preconditioner ``D^-1``, the inverse of the
        technosphere diagonal. Returns ``None`` if any diagonal entry is zero."""
        self._prepare_matrix()
        matrix = self._prepared_technosphere_matrix
        diagonal = matrix.diagonal()
        # Cannot build Jacobi inverse if any diagonal entry is zero.
        if np.any(diagonal == 0):
            return None

        inverse_diagonal = 1.0 / diagonal
        # LinearOperator form avoids materializing a dense diagonal inverse matrix.
        return LinearOperator(
            shape=matrix.shape,
            matvec=lambda x: inverse_diagonal * x,
            dtype=matrix.dtype,
        )

    def _get_preconditioner(self) -> Optional[LinearOperator]:
        # Reuse preconditioner when solving multiple demands on same matrix.
        if self._cached_preconditioner is None:
            self._cached_preconditioner = self.build_preconditioner()
        return self._cached_preconditioner

    def solver_kwargs(self) -> dict:
        """Additional keyword arguments for ``iter_solver``."""
        return {}

    def _call_solver(self, matrix, demand, x0, preconditioner):
        kwargs = dict(
            x0=x0,
            atol=self.atol,
            maxiter=self.maxiter,
            M=preconditioner,
            **self.solver_kwargs(),
        )
        try:
            # SciPy modern API (`rtol` + `atol`).
            return self.iter_solver(matrix, demand, rtol=self.rtol, **kwargs)
        except TypeError:
            # Backward compatibility for SciPy versions using `tol`.
            return self.iter_solver(matrix, demand, tol=self.rtol, **kwargs)

    def _solution_is_acceptable(self, matrix, demand, solution) -> bool:
        if solution.shape != demand.shape or not np.all(np.isfinite(solution)):
            return False
        residual = np.linalg.norm(matrix @ solution - demand)
        threshold = max(self.rtol * np.linalg.norm(demand), self.atol)
        return residual <= self.residual_factor * threshold

    def _direct_solve(self, demand: np.ndarray) -> np.ndarray:
        return super().solve_linear_system(demand)

    def solve_linear_system(self, demand: Optional[np.ndarray] = None) -> np.ndarray:
        if demand is None:
            demand = self.demand_array

        x0 = self.guess if (self.use_guess and self.guess is not None) else None

        if x0 is None and self.direct_first_solve:
            solution = self._direct_solve(demand)
        else:
            self._prepare_matrix()
            matrix = self._prepared_technosphere_matrix
            solution, info = self._call_solver(
                matrix, demand, x0, self._get_preconditioner()
            )
            solution = np.asarray(solution)

            if info != 0 or not self._solution_is_acceptable(matrix, demand, solution):
                # A silent fallback would look like a working but inexplicably slow
                # iterative LCA, so make it visible that the solver isn't being used.
                logger.debug(
                    "Iterative solver failed (info=%s); falling back to the direct solver",
                    info,
                    extra={
                        "info": info,
                        "solver": getattr(self.iter_solver, "__name__", repr(self.iter_solver)),
                        "rtol": self.rtol,
                        "maxiter": self.maxiter,
                    },
                )
                solution = self._direct_solve(demand)

        # Match return conventions used elsewhere in bw2calc.
        solution = np.asarray(solution)
        if not solution.shape:
            solution = solution.reshape((1,))

        if self.use_guess:
            # Keep latest solution for the next warm-started call.
            self.guess = solution

        return solution
