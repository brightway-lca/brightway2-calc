import logging
import time
import weakref
from collections import Counter
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator, bicgstab
from threadpoolctl import ThreadpoolController

from bw2calc import PYPARDISO, factorized
from bw2calc.lca import LCA

if PYPARDISO:
    from pypardiso import PyPardisoSolver
    from pypardiso.pardiso_wrapper import PyPardisoError
    from pypardiso.scipy_aliases import pypardiso_solver

logger = logging.getLogger("bw2calc")


class IterativeLCA(LCA):
    """Solve ``Ax=b`` using iterative techniques instead of
    `LU factorization <http://en.wikipedia.org/wiki/LU_decomposition>`_.

    Any solver with the signature of the ``scipy.sparse.linalg`` iterative solvers can be
    used, i.e. ``solver(A, b, x0=..., rtol=..., atol=..., maxiter=..., M=..., callback=...)``
    returning ``(solution, info)``. ``callback`` is called with any arguments in each
    iteration, and is used for the time limit. The default is
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
    * Not yet checked: other databases, and ``rtol`` / ``restart`` sweeps.

    x86 + pypardiso
    ---------------

    Full Monte Carlo iterations (median of 50) on an AMD Ryzen 9 5950X (16 cores),
    pypardiso 0.4.7, same demand and method as above, with the solving strategy and
    checks below (``JacobiGMRESLCA`` with ``direct_first_solve=True``). "4 threads" is ``MKL_NUM_THREADS=OMP_NUM_THREADS=4``; "no limit" lets
    MKL use 16 threads:

    ==========  ==========  ===================  ================  ==================
    ecoinvent   Threads     ``LCA`` (pypardiso)  ``IterativeLCA``  ``JacobiGMRESLCA``
    ==========  ==========  ===================  ================  ==================
    3.10        4           416 ms               74 ms (5.6x)      86 ms (4.8x)
    3.10        no limit    326 ms               79 ms (4.1x)      92 ms (3.5x)
    3.12        4           584 ms               92 ms (6.3x)      104 ms (5.6x)
    3.12        no limit    428 ms               95 ms (4.5x)      109 ms (3.9x)
    ==========  ==========  ===================  ================  ==================

    Scores match ``LCA`` to within 1e-6 (``IterativeLCA``) and 1e-5 (``JacobiGMRESLCA``).
    Monte Carlo for eight ill-conditioned demands with strong waste, recycling, or
    construction loops gave similar speedups and scores within 1e-6, all solved
    iteratively; for two nuclear construction demands on 3.12, GMRES failed and the
    fallback stage took over (about 280 ms per iteration). New demands on an unchanged
    matrix take the same time as ``LCA``, 11-14 ms.

    Solving strategy
    ----------------

    The first solve is direct, and its factorization of the technosphere matrix is kept as
    the *reference factorization*. Each later solve then works through these stages:

    #. If the technosphere matrix is still the reference matrix, e.g. for a new demand on
       the same system, solve directly with the reference factorization. This is as fast
       as ``LCA``, and much faster than any iterative solver.
    #. Otherwise, run ``iter_solver`` (BiCGSTAB with a Jacobi preconditioner by default).
    #. If that fails, run ``fallback_solver`` (BiCGSTAB by default), preconditioned with
       the reference factorization. Each iteration costs two solves with the reference
       factorization, but it converges in a few iterations, even for ill-conditioned
       demands, as long as the matrix is close to the reference, as in Monte Carlo.
    #. If that fails too, solve directly with a new factorization.

    Each iterative stage stops after ``time_limit`` seconds; by default, the time the
    first direct solve took (at least 0.1 s), as after that a direct solve would have been
    faster. After
    ``max_consecutive_failures`` failures in a row, a stage is skipped until new matrices
    are loaded. ``solver_stats`` counts how many solves each stage did.

    With pypardiso, the reference factorization has its own ``PyPardisoSolver``, separate
    from the global one used by ``spsolve``, so direct solves don't replace it. Its memory
    is released when new matrices are loaded, or when the object is garbage collected.
    The factorization a direct fallback leaves in the global solver is released right
    after the solve, so that only the reference factorization stays in memory.

    Correctness checks
    ------------------

    Convergence flags can't be trusted: TFQMR reported ``info == 0`` with a solution 20%
    off, and BiCGSTAB can break down (``info < 0``), stagnate, or return non-finite values.

    A small residual isn't enough either. For demands with strong waste or recycling
    loops, the technosphere matrix is ill-conditioned, and a relative residual
    ``||Ax - b|| / ||b||`` of 1e-8 was compatible with a solution 20% off (GMRES,
    ``market for waste polystyrene``, ecoinvent 3.12). It also goes the other way: for
    some construction demands, even the direct solution has a relative residual above
    1e-7, so a residual check rejects correct iterative solutions.

    Each iterative solution is therefore checked by estimating its relative forward error
    with the reference factorization, ``||A0^-1 (b - Ax)|| / ||x||``, where ``A0`` is the
    reference matrix. ``A0^-1`` amplifies the residual in the same directions as
    ``A^-1``, so this tracks the true error closely as long as ``A`` is close to ``A0``:
    within about 2x over 80 Monte Carlo solves on ecoinvent 3.12. It costs one solve with
    the reference factorization. Solutions with an estimated error above ``max_error`` are
    rejected. Without a reference factorization (``direct_first_solve=False``), or with
    ``max_error=None``, the true residual is checked against
    ``residual_factor * max(rtol * ||b||, atol)`` instead.

    Every rejected solution is logged as a debug message on the ``bw2calc`` logger.

    Warm starts
    -----------

    With ``use_guess=True``, the previous solution is used as the initial guess ``x0`` for
    the next iterative solve.

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
    :param direct_first_solve: If ``True``, solve the first system directly and keep its
        factorization as the reference factorization. If ``False``, there is no reference
        factorization: every solve is iterative, solutions are checked by their residual,
        and failures go straight to the direct solver.
    :type direct_first_solve: bool
    :param residual_factor: Slack factor for the residual check, used when there is no
        forward error check.
    :type residual_factor: float
    :param max_error: Maximum estimated relative forward error of an iterative solution.
        ``None`` checks the residual instead.
    :type max_error: float or None
    :param time_limit: Time limit in seconds for each iterative stage. ``"auto"`` uses the
        time of the first direct solve, at least 0.1 s, or no limit without one. ``None``
        means no limit.
    :type time_limit: float, str, or None
    :param fallback_solver: Iterative solver preconditioned with the reference
        factorization, used when ``iter_solver`` fails. Defaults to
        :func:`scipy.sparse.linalg.bicgstab`.
    :type fallback_solver: callable
    :param fallback_maxiter: Maximum number of iterations for ``fallback_solver``.
    :type fallback_maxiter: int or None
    :param max_consecutive_failures: Skip an iterative stage after this many failures in a
        row, until new matrices are loaded. ``None`` never skips a stage.
    :type max_consecutive_failures: int or None
    """

    STAGES = ("iterative", "fallback_iterative")
    MIN_AUTO_TIME_LIMIT = 0.1

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
        max_error: Optional[float] = 1e-6,
        time_limit: Union[float, str, None] = "auto",
        fallback_solver: Optional[Callable] = None,
        fallback_maxiter: Optional[int] = 100,
        max_consecutive_failures: Optional[int] = 3,
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
        self.max_error = max_error
        self.time_limit = time_limit
        self.fallback_solver = fallback_solver if fallback_solver is not None else bicgstab
        self.fallback_maxiter = fallback_maxiter
        self.max_consecutive_failures = max_consecutive_failures
        # Prepared CSR copy used by the iterative solver; don't replace
        # `technosphere_matrix`, as Monte Carlo iteration mutates the matrix held by
        # `technosphere_mm`. `None` means "not prepared yet".
        self._prepared_technosphere_matrix = None
        # Cache the preconditioner to avoid rebuilding between solves.
        self._cached_preconditioner: Optional[LinearOperator] = None
        # Last solution vector, used as warm start when `use_guess=True`.
        self.guess = None
        # Number of solves done by each stage: "reference", "iterative",
        # "fallback_iterative", and "direct".
        self.solver_stats = Counter()
        self._clear_reference()

    def __next__(self) -> None:
        # Matrix values can change across iteration steps, so invalidate caches.
        self._clear_matrix_caches()
        super().__next__()

    def load_lci_data(self, nonsquare_ok=False) -> None:
        super().load_lci_data(nonsquare_ok=nonsquare_ok)
        # New matrices imply stale solver-side caches.
        self._clear_matrix_caches()
        self._clear_reference()
        self.guess = None

    def _clear_matrix_caches(self) -> None:
        self._prepared_technosphere_matrix = None
        self._cached_preconditioner = None

    def _clear_reference(self) -> None:
        finalizer = getattr(self, "_reference_finalizer", None)
        if finalizer is not None:
            # Release the reference factorization now instead of when `self` is collected.
            finalizer()
        self._reference_finalizer = None
        self._reference_matrix = None
        self._reference_solve = None
        self._reference_time = None
        self._consecutive_failures = {stage: 0 for stage in self.STAGES}

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

    def _build_reference(self, demand: np.ndarray) -> np.ndarray:
        """Factorize the current technosphere matrix, keep the factorization as the
        reference, and solve ``demand`` with it."""
        start = time.perf_counter()
        matrix = self.technosphere_matrix.tocsr(copy=True)
        if PYPARDISO:
            # Use a separate solver instead of pypardiso's global one, which `spsolve`
            # uses and `LCA.__next__` resets, so the reference factorization survives
            # direct solves, and the time measured is always a real factorization.
            solver = PyPardisoSolver()
            solver.factorize(matrix)
            self._reference_finalizer = weakref.finalize(self, _free_pardiso, solver)
            self._reference_finalizer.atexit = False

            def solve(b):
                return solver.solve(matrix, b)

        else:
            # UMFPACK factorization needs CSC sparse matrix; see
            # https://github.com/brightway-lca/brightway2-calc/issues/132
            solve = factorized(matrix.tocsc())
        self._reference_solve = lambda b: np.asarray(solve(b)).reshape(np.shape(b))
        solution = self._reference_solve(demand)
        self._reference_time = time.perf_counter() - start
        self._reference_matrix = matrix
        return solution

    def _is_reference_matrix(self) -> bool:
        reference = self._reference_matrix
        if reference is None:
            return False
        matrix = self.technosphere_matrix.tocsr()
        return (
            matrix.shape == reference.shape
            and np.array_equal(matrix.indptr, reference.indptr)
            and np.array_equal(matrix.indices, reference.indices)
            and np.array_equal(matrix.data, reference.data)
        )

    def _reference_preconditioner(self) -> LinearOperator:
        return LinearOperator(
            shape=self._reference_matrix.shape,
            matvec=self._reference_solve,
            dtype=self._reference_matrix.dtype,
        )

    def _stage_time_limit(self) -> Optional[float]:
        if self.time_limit == "auto":
            if self._reference_time is None:
                return None
            # The floor keeps timer jitter from failing solves of small systems.
            return max(self._reference_time, self.MIN_AUTO_TIME_LIMIT)
        return self.time_limit

    def _call_solver(self, solver, matrix, demand, kwargs):
        try:
            # SciPy modern API (`rtol` + `atol`).
            return solver(matrix, demand, rtol=self.rtol, **kwargs)
        except TypeError:
            # Backward compatibility for SciPy versions using `tol`.
            return solver(matrix, demand, tol=self.rtol, **kwargs)

    def estimate_error(self, matrix, demand, solution) -> float:
        """Estimate the relative forward error ``||x - A^-1 b|| / ||x||`` of ``solution``
        as ``||A0^-1 (b - Ax)|| / ||x||``, where ``A0`` is the reference matrix."""
        correction = self._reference_solve(demand - matrix @ solution)
        norm = np.linalg.norm(solution)
        if norm == 0:
            return 0.0 if not np.any(correction) else np.inf
        return float(np.linalg.norm(correction) / norm)

    def _solution_is_acceptable(self, matrix, demand, solution) -> bool:
        if solution.shape != demand.shape or not np.all(np.isfinite(solution)):
            return False
        if self._reference_solve is not None and self.max_error is not None:
            return self.estimate_error(matrix, demand, solution) <= self.max_error
        residual = np.linalg.norm(matrix @ solution - demand)
        threshold = max(self.rtol * np.linalg.norm(demand), self.atol)
        return residual <= self.residual_factor * threshold

    def _direct_solve(self, demand: np.ndarray) -> np.ndarray:
        return super().solve_linear_system(demand)

    def _run_stage(self, stage: str, matrix, demand, x0) -> Optional[np.ndarray]:
        """Run one iterative stage, and return its solution, or ``None`` on failure."""
        if stage == "iterative":
            solver = self.iter_solver
            kwargs = dict(
                maxiter=self.maxiter, M=self._get_preconditioner(), **self.solver_kwargs()
            )
        else:
            solver = self.fallback_solver
            kwargs = dict(maxiter=self.fallback_maxiter, M=self._reference_preconditioner())
        kwargs.update(x0=x0, atol=self.atol)

        time_limit = self._stage_time_limit()
        if time_limit is not None:
            deadline = time.perf_counter() + time_limit

            def callback(*args):
                if time.perf_counter() > deadline:
                    raise _TimeLimitExceeded

            kwargs["callback"] = callback

        try:
            solution, info = self._call_solver(solver, matrix, demand, kwargs)
        except _TimeLimitExceeded:
            solution, info = None, "time limit"
        else:
            solution = np.asarray(solution)
            if info == 0 and self._solution_is_acceptable(matrix, demand, solution):
                return solution

        # A silent fallback would look like a working but inexplicably slow
        # iterative LCA, so make it visible that the solver isn't being used.
        logger.debug(
            "Iterative stage %s failed (info=%s)",
            stage,
            info,
            extra={
                "stage": stage,
                "info": info,
                "solver": getattr(solver, "__name__", repr(solver)),
                "rtol": self.rtol,
                "time_limit": time_limit,
            },
        )
        return None

    def _stage_skipped(self, stage: str) -> bool:
        if stage == "fallback_iterative" and self._reference_solve is None:
            return True
        return (
            self.max_consecutive_failures is not None
            and self._consecutive_failures[stage] >= self.max_consecutive_failures
        )

    def _iterative_solve(self, demand: np.ndarray) -> np.ndarray:
        self._prepare_matrix()
        matrix = self._prepared_technosphere_matrix
        x0 = self.guess if self.use_guess else None

        for stage in self.STAGES:
            if self._stage_skipped(stage):
                continue
            with _single_threaded_openblas():
                solution = self._run_stage(stage, matrix, demand, x0)
            if solution is not None:
                self._consecutive_failures[stage] = 0
                self.solver_stats[stage] += 1
                return solution
            self._consecutive_failures[stage] += 1
            if self._stage_skipped(stage):
                logger.info(
                    "Iterative stage %s failed %s times in a row; skipping it until new "
                    "matrices are loaded",
                    stage,
                    self._consecutive_failures[stage],
                )

        self.solver_stats["direct"] += 1
        solution = self._direct_solve(demand)
        if PYPARDISO and self._reference_solve is not None:
            # The direct solve left a factorization of this matrix in pypardiso's global
            # solver, next to the reference factorization. It is almost never reused:
            # solves on the reference matrix use the reference factorization, and Monte
            # Carlo moves on to a new matrix. `LCA.__next__` only frees its LU factors,
            # so release all of it now; about 100 MB for ecoinvent.
            _free_pardiso(pypardiso_solver)
        return solution

    def solve_linear_system(self, demand: Optional[np.ndarray] = None) -> np.ndarray:
        if demand is None:
            demand = self.demand_array

        if self._reference_solve is None and self.direct_first_solve:
            solution = self._build_reference(demand)
            self.solver_stats["reference"] += 1
        elif self._is_reference_matrix():
            solution = self._reference_solve(demand)
            self.solver_stats["reference"] += 1
        else:
            solution = self._iterative_solve(demand)

        # Match return conventions used elsewhere in bw2calc.
        solution = np.asarray(solution)
        if not solution.shape:
            solution = solution.reshape((1,))

        if self.use_guess:
            # Keep latest solution for the next warm-started call.
            self.guess = solution

        return solution


_openblas_controller = None


def _single_threaded_openblas():
    """Limit OpenBLAS, which NumPy uses for the solvers' vector operations, to one thread.

    The vectors are too small to gain from threads: with 32 OpenBLAS threads and MKL
    limited to one thread, BiCGSTAB was still 1.6x slower than with one OpenBLAS thread.

    With pypardiso it is much worse, as both thread pools busy-wait between calls instead
    of sleeping, and together they have more threads than the CPU. The contention goes
    both ways: the MKL threads Pardiso leaves spinning after each solve slow BiCGSTAB, and
    spinning OpenBLAS threads slow the Pardiso solves of the error check and the fallback
    stage. On a Ryzen 9 5950X (16 cores, 32 threads), ecoinvent 3.10 Monte Carlo, the
    median solve took 22 ms with this limit, and 220-310 ms without it (BiCGSTAB 17 ms to
    197 ms, the Pardiso error check 4 ms to 43 ms). ``KMP_BLOCKTIME=0``, which stops MKL
    threads spinning, doesn't fix this: BiCGSTAB went back to 34 ms, but the Pardiso error
    check went up to 81 ms, as MKL threads now had to wake up against the spinning
    OpenBLAS threads, and the solve still took 140-150 ms.

    Only OpenBLAS is limited. MKL isn't, so Pardiso keeps its threads."""
    global _openblas_controller
    if _openblas_controller is None:
        # Created on first use, when NumPy and SciPy have loaded their BLAS libraries.
        _openblas_controller = ThreadpoolController().select(internal_api="openblas")
    return _openblas_controller.limit(limits=1)


class _TimeLimitExceeded(Exception):
    """Raised from a solver callback to stop an iterative stage."""


def _free_pardiso(solver) -> None:
    """Release all the memory Pardiso holds for ``solver``."""
    try:
        solver.free_memory(everything=True)
    except PyPardisoError:
        # Best effort, like `LCA._delete_solver_state`.
        logger.debug("Couldn't release Pardiso memory")
