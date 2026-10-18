import inspect
import logging
import time
import warnings
import weakref
from collections import Counter
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator, bicgstab
from threadpoolctl import ThreadpoolController

from bw2calc import PYPARDISO, factorized
from bw2calc.errors import InvalidReferenceFlags
from bw2calc.lca import LCA
from bw2calc.lca_base import _release_pardiso_solver
from bw2calc.utils import production_pivot_map, reference_entries

if PYPARDISO:
    from pypardiso import PyPardisoSolver

logger = logging.getLogger("bw2calc")


class IterativeLCA(LCA):
    """Solve ``Ax=b`` using iterative techniques instead of
    `LU factorization <http://en.wikipedia.org/wiki/LU_decomposition>`_.

    Any solver with the signature of the ``scipy.sparse.linalg`` iterative solvers can be
    used, i.e. ``solver(A, b, x0=..., rtol=..., atol=..., maxiter=..., M=..., callback=...)``
    returning ``(solution, info)``. ``callback`` is called with any arguments in each
    iteration, and is used for the time limit. The default is
    `BiCGSTAB <https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method>`_
    with a Jacobi preconditioner, which divides by the production amounts; see
    `Jacobi preconditioner`_ below.

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
    checks below (``JacobiGMRESLCA`` with ``direct_first_solve=True``). "4 threads" is
    ``MKL_NUM_THREADS=OMP_NUM_THREADS=4``; "no limit" lets MKL use 16 threads:

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
    ``max_consecutive_failures`` failures in a row, a stage is skipped for the next
    ``retry_after`` solves, and then tried again; if it fails again, it is skipped again.
    In Monte Carlo, a run of hard samples then doesn't turn a stage off for all later
    samples. ``solver_stats`` counts how many solves each stage did.

    With pypardiso, the reference factorization has its own ``PyPardisoSolver``, separate
    from the object's solver for direct solves, so direct solves don't replace it. Its
    memory is released when new matrices are loaded, or when the object is garbage
    collected. The factorization a direct fallback leaves in the object's solver is
    released right after the solve, so that only the reference factorization stays in
    memory.

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
    rejected, and solutions below it are accepted even if the solver reports that it didn't
    converge; if the time limit stops the solver, its last iterate is checked. Without a
    reference factorization (``direct_first_solve=False``), or with ``max_error=None``,
    the true residual is checked against
    ``residual_factor * max(rtol * ||b||, atol)`` instead.

    Every rejected solution is logged as a debug message on the ``bw2calc`` logger.

    Jacobi preconditioner
    ---------------------

    Scaling each activity to one unit of its reference product turns the technosphere
    matrix into ``I - T``, with the inputs per unit of output in ``T``, and eigenvalues
    well inside the unit circle. The preconditioner therefore divides by the *production
    pivots* ``A[p(j), j]``, where ``p(j)`` is the row of the reference product of activity
    column ``j``: ``(M r)[j] = r[p(j)] / A[p(j), j]``. This is Jacobi preconditioning after
    the row permutation ``p``; nothing is reordered, each application is one gather and
    one multiplication.

    ``p`` comes from the reference exchange flags in the datapackages (bw_processing's
    ``reference_array``). For databases where processes and their products share an id,
    ``p(j) = j``, and this is the plain diagonal. For databases with separate product
    nodes, product rows are numbered independently of activity columns, so the diagonal
    holds arbitrary entries, often zeros, and only the pivot map gives a useful
    preconditioner. Permuting rows doesn't change the solution, so a wrong pairing only
    costs speed, never correctness.

    The preconditioner doesn't rescale anything. The solver uses ``M`` only to choose its
    search directions (in SciPy's BiCGSTAB: ``phat = M @ p``, then ``x += alpha * phat``),
    so the iterate ``x`` is always the supply in the original units, and every product
    with ``A`` uses the unscaled technosphere matrix. ``technosphere_matrix`` and
    ``biosphere_matrix`` are never changed, and ``supply_array``, the inventory ``B @ x``,
    and the impact assessment are calculated as for :class:`~bw2calc.lca.LCA`. A change of
    variables ``x = D^-1 y`` would need ``B D^-1`` (or converting ``y`` back to ``x``), but
    none is made. The forward error check also uses the original ``A``, so a poor ``M``
    can make a stage slower or fail, but can't make it accept a wrong solution.

    ``jacobi_pairing`` chooses how ``p`` is found:

    * ``"auto"`` (default): from the reference exchange flags, if they pair every column
      with exactly one row, and every row with exactly one column. Otherwise, e.g. for
      datapackages without flags, the diagonal is used, and a warning gives the reason.
    * ``"reference"``: from the flags, raising
      :class:`~bw2calc.errors.InvalidReferenceFlags` if they can't be used.
    * ``"diagonal"``: always the diagonal, ignoring the flags.

    In Monte Carlo, ``p`` and the pivot values are handled separately:

    #. ``load_lci_data()`` builds the matrices and clears ``p``.
    #. The first solve is direct, and keeps the reference factorization. No preconditioner
       is built.
    #. ``next()`` draws new values into the same matrix structure. The prepared matrix
       copy and the cached preconditioner are discarded; ``p`` is kept, as Monte Carlo
       changes values, not structure.
    #. The iterative stage makes a prepared copy of the sampled matrix, builds ``p`` from
       the flags if this is the first time, and reads this sample's production amounts
       ``A[p(j), j]``. Production amounts can be uncertain, so they are read from every
       sample. BiCGSTAB then solves the sampled system with this ``M``, starting from the
       previous solution, and the result goes through the forward error check and, if
       needed, the fallback stages as usual.
    #. Further demands on the same sample reuse the cached ``M``.

    If a pivot is zero or not finite, e.g. a sampled production amount of zero, which is
    removed from the prepared copy, that sample's iterative stage runs without a
    preconditioner. ``preconditioner_source`` tells which preconditioner the last
    iterative stage used: ``"production_pivots"``, ``"diagonal"``, or ``None``.

    Monte Carlo on BAFU 2026 v1 (11,947 x 11,947, 114k nonzeros), demand 1 kWh
    ``Electricity, low voltage, at grid`` (CH), 30 iterations, Apple M1 Max, no
    scikit-umfpack. "Separate products" gives the product rows ids which sort
    independently of the process ids, as separate ``product`` nodes do; this leaves
    11,939 zeros on the diagonal:

    =================  ==================  ==============  ==========  ===================
    Product ids        Preconditioner      ms / iteration  Iterations  Stage
    =================  ==================  ==============  ==========  ===================
    Shared             -- (``LCA``)        200             --          --
    Shared             Diagonal            43              20          iterative
    Shared             Production pivots   42              20          iterative
    Separate products  -- (``LCA``)        183             --          --
    Separate products  None (zero diag.)   122             1000        fallback_iterative
    Separate products  Production pivots   44              20          iterative
    =================  ==================  ==============  ==========  ===================

    With the diagonal, the zeros left the iterative stage without a preconditioner, so it
    ran to ``maxiter`` and the fallback stage solved every sample. Supply arrays were
    within 2e-8 of ``LCA`` in all cases.

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
        :func:`scipy.sparse.linalg.bicgstab`. ``None`` or ``False`` turns off iterative
        solving: changed matrices are always solved directly.
    :type iter_solver: callable or None
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
        :func:`scipy.sparse.linalg.bicgstab`. ``None`` turns off the fallback stage.
    :type fallback_solver: callable or None
    :param fallback_maxiter: Maximum number of iterations for ``fallback_solver``.
    :type fallback_maxiter: int or None
    :param max_consecutive_failures: Skip an iterative stage after this many failures in a
        row. ``None`` never skips a stage.
    :type max_consecutive_failures: int or None
    :param retry_after: Number of solves a failing stage is skipped for before it is tried
        again. ``None`` skips it until new matrices are loaded.
    :type retry_after: int or None
    :param jacobi_pairing: How the Jacobi preconditioner pairs activity columns with
        product rows: ``"auto"``, ``"reference"``, or ``"diagonal"``; see above.
    :type jacobi_pairing: str
    """

    STAGES = ("iterative", "fallback_iterative")
    JACOBI_PAIRINGS = ("auto", "reference", "diagonal")
    MIN_AUTO_TIME_LIMIT = 0.1

    def __init__(
        self,
        *args,
        iter_solver: Optional[Callable] = bicgstab,
        rtol: float = 1e-8,
        atol: float = 0.0,
        maxiter: Optional[int] = 1000,
        use_guess: bool = True,
        direct_first_solve: bool = True,
        residual_factor: float = 10.0,
        max_error: Optional[float] = 1e-6,
        time_limit: Union[float, str, None] = "auto",
        fallback_solver: Optional[Callable] = bicgstab,
        fallback_maxiter: Optional[int] = 100,
        max_consecutive_failures: Optional[int] = 3,
        retry_after: Optional[int] = 20,
        jacobi_pairing: str = "auto",
        **kwargs,
    ):
        if jacobi_pairing not in self.JACOBI_PAIRINGS:
            raise ValueError(
                f"`jacobi_pairing` must be one of {self.JACOBI_PAIRINGS}, got {jacobi_pairing!r}"
            )
        super().__init__(*args, **kwargs)
        self.iter_solver = iter_solver
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.use_guess = use_guess
        self.direct_first_solve = direct_first_solve
        self.residual_factor = residual_factor
        self.max_error = max_error
        self.time_limit = time_limit
        self.fallback_solver = fallback_solver
        self.fallback_maxiter = fallback_maxiter
        self.max_consecutive_failures = max_consecutive_failures
        self.retry_after = retry_after
        self.jacobi_pairing = jacobi_pairing
        # Preconditioner used by the last iterative stage: "production_pivots",
        # "diagonal", or `None`.
        self.preconditioner_source = None
        self._clear_pivot_map()
        # Prepared CSR copy used by the iterative solver; don't replace
        # `technosphere_matrix`, as Monte Carlo iteration mutates the matrix held by
        # `technosphere_mm`. `None` means "not prepared yet".
        self._prepared_technosphere_matrix = None
        # Copy of `technosphere_matrix` the prepared matrix was built from.
        self._prepared_source = None
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
        # The pivot map depends on the structure, which only changes here.
        self._clear_pivot_map()
        self.guess = None

    def _clear_pivot_map(self) -> None:
        # Row of the reference product of each column, or `None` for the diagonal.
        self._pivot_rows = None
        # "production_pivots" or "diagonal"; `None` means "not built yet".
        self._pivot_pairing = None

    def _clear_matrix_caches(self) -> None:
        self._prepared_technosphere_matrix = None
        self._prepared_source = None
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
        # Solves skipped by each stage since it last failed.
        self._skipped_solves = {stage: 0 for stage in self.STAGES}

    def _prepare_matrix(self) -> None:
        # Sparse cleanup is done once per matrix, then reused.
        if not sps.issparse(self.technosphere_matrix):
            raise TypeError("technosphere_matrix must be a SciPy sparse matrix")
        source = self.technosphere_matrix.tocsr()
        # `technosphere_matrix` can also be replaced or edited in place between solves,
        # without `__next__` or `load_lci_data`. A stale copy would solve the old system,
        # and pass the solution checks too, as they use the same copy.
        if self._prepared_technosphere_matrix is not None and _csr_equal(
            source, self._prepared_source
        ):
            return

        # Iterative solvers work best with canonical sparse structure. CSR is fine for
        # the matrix-vector products they need, and is what `LCA` builds. Always copy:
        # with `copy=False`, a `technosphere_matrix` which is already CSR would be
        # returned as-is, and `eliminate_zeros()` would then strip structural zeros from
        # the matrix owned by `technosphere_mm`, which needs them to update in place.
        matrix = source.copy()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        # Keep an unmodified copy to detect changes; in-place edits would change `source`.
        self._prepared_source = source.copy()
        self._prepared_technosphere_matrix = matrix
        self._cached_preconditioner = None

    def build_preconditioner(self) -> Optional[LinearOperator]:
        """Return a preconditioner ``M`` approximating ``A^-1``, or ``None`` for no
        preconditioning. The result is cached until the technosphere matrix changes.

        The default is the Jacobi preconditioner, which divides by the production pivots
        ``A[p(j), j]``, or by the diagonal; see ``jacobi_pairing``. Returns ``None`` if
        any of them is zero or not finite."""
        self._prepare_matrix()
        matrix = self._prepared_technosphere_matrix
        pivot_rows = self._production_pivot_rows()
        if pivot_rows is None:
            pivots = matrix.diagonal()
        else:
            pivots = _csr_entries(matrix, pivot_rows, np.arange(matrix.shape[1]))
        # Monte Carlo samples can have a zero production amount.
        if not np.all(np.isfinite(pivots)) or np.any(pivots == 0):
            return None

        inverse_pivots = 1.0 / pivots
        self.preconditioner_source = self._pivot_pairing
        # `A` maps activities to products, and `M` approximates `A^-1`, so it maps
        # products to activities: activity `j` reads the residual of its product `p(j)`.
        if pivot_rows is None:
            matvec = lambda r: inverse_pivots * r  # noqa: E731
        else:
            matvec = lambda r: inverse_pivots * r[pivot_rows]  # noqa: E731
        # `scipy.sparse.linalg.LinearOperator` wraps `matvec`, so the solver only ever
        # calls `M @ r` and `M` is never stored as a matrix. `M` is diagonal (after
        # permuting rows), so `matvec` needs only the 1-D array `inverse_pivots`; building
        # `M` as a matrix would need a sparse copy with its own index arrays, or, for a
        # dense one, `n^2` entries.
        return LinearOperator(shape=matrix.shape, matvec=matvec, dtype=matrix.dtype)

    def _get_preconditioner(self) -> Optional[LinearOperator]:
        # Reuse preconditioner when solving multiple demands on same matrix.
        if self._cached_preconditioner is None:
            # Set by `build_preconditioner` when it returns a Jacobi preconditioner.
            self.preconditioner_source = None
            self._cached_preconditioner = self.build_preconditioner()
        return self._cached_preconditioner

    def _production_pivot_rows(self) -> Optional[np.ndarray]:
        """Return the pivot map ``p`` as an array of row indices, or ``None`` for the
        diagonal. Built once per ``load_lci_data``."""
        if self._pivot_pairing is None:
            self._pivot_rows, self._pivot_pairing = self._build_pivot_map()
            logger.debug("Jacobi preconditioner pairing: %s", self._pivot_pairing)
        return self._pivot_rows

    def _build_pivot_map(self) -> tuple:
        if self.jacobi_pairing == "diagonal":
            return None, "diagonal"
        rows, cols = reference_entries(self.technosphere_mm)
        try:
            if not len(rows):
                raise InvalidReferenceFlags("No reference exchange flags")
            pivot_rows = production_pivot_map(rows, cols, self.technosphere_matrix.shape)
        except InvalidReferenceFlags as error:
            if self.jacobi_pairing == "reference":
                raise
            # With separate product nodes, the diagonal can hold arbitrary entries or
            # zeros, making iterative solves slow or sending them to the fallback stages.
            warnings.warn(
                f"{error}; using the diagonal for the Jacobi preconditioner. Pass "
                "`jacobi_pairing='diagonal'` to silence this warning."
            )
            return None, "diagonal"
        if np.array_equal(pivot_rows, np.arange(len(pivot_rows))):
            # Same preconditioner as the diagonal, without the gather in each iteration.
            pivot_rows = None
        return pivot_rows, "production_pivots"

    def solver_kwargs(self) -> dict:
        """Additional keyword arguments for ``iter_solver``."""
        return {}

    def _build_reference(self, demand: np.ndarray) -> np.ndarray:
        """Factorize the current technosphere matrix, keep the factorization as the
        reference, and solve ``demand`` with it."""
        start = time.perf_counter()
        matrix = self.technosphere_matrix.tocsr(copy=True)
        solver = None
        try:
            if PYPARDISO:
                # Use a separate solver instead of the object's own one, which direct
                # solves use and `LCA.__next__` resets, so the reference factorization
                # survives direct solves, and the time measured is always a real
                # factorization.
                solver = PyPardisoSolver()
                solver.factorize(matrix)

                def solve(b):
                    return solver.solve(matrix, b)

            else:
                # UMFPACK factorization needs CSC sparse matrix; see
                # https://github.com/brightway-lca/brightway2-calc/issues/132
                solve = factorized(matrix.tocsc())

            def reference_solve(b):
                return np.asarray(solve(b)).reshape(np.shape(b))

            solution = reference_solve(demand)
        except Exception:
            if solver is not None:
                _release_pardiso_solver(solver)
            raise

        # Only keep the reference once it works; a partial reference would send later
        # solves to the fallback stage without a reference matrix.
        if solver is not None:
            self._reference_finalizer = weakref.finalize(self, _release_pardiso_solver, solver)
            self._reference_finalizer.atexit = False
        self._reference_solve = reference_solve
        self._reference_time = time.perf_counter() - start
        self._reference_matrix = matrix
        return solution

    def _is_reference_matrix(self) -> bool:
        reference = self._reference_matrix
        if reference is None:
            return False
        return _csr_equal(self.technosphere_matrix.tocsr(), reference)

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
        kwargs = {**kwargs, _tolerance_keyword(solver): self.rtol}
        return solver(matrix, demand, **kwargs)

    def estimate_error(self, matrix, demand, solution) -> float:
        """Estimate the relative forward error ``||x - A^-1 b|| / ||x||`` of ``solution``
        as ``||A0^-1 (b - Ax)|| / ||x||``, where ``A0`` is the reference matrix."""
        correction = self._reference_solve(demand - matrix @ solution)
        norm = np.linalg.norm(solution)
        if norm == 0:
            return 0.0 if not np.any(correction) else np.inf
        return float(np.linalg.norm(correction) / norm)

    def _checks_forward_error(self) -> bool:
        return self._reference_solve is not None and self.max_error is not None

    def _solution_is_acceptable(self, matrix, demand, solution) -> bool:
        if solution.shape != demand.shape or not np.all(np.isfinite(solution)):
            return False
        if self._checks_forward_error():
            return self.estimate_error(matrix, demand, solution) <= self.max_error
        residual = np.linalg.norm(matrix @ solution - demand)
        threshold = max(self.rtol * np.linalg.norm(demand), self.atol)
        return residual <= self.residual_factor * threshold

    def _direct_solve(self, demand: np.ndarray) -> np.ndarray:
        # Not `LCA.solve_linear_system`, which uses `self.solver` from
        # `lci(factorize=True)` if it exists, even if the matrix has been edited since.
        return self._spsolve(self.technosphere_matrix, demand)

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

        # The forward error check is what we trust, so with it, accept any solution which
        # passes it, whatever the solver reports. Some systems can't reach `rtol`, e.g.
        # construction demands where even the direct solution has a relative residual
        # above 1e-7, and the solver would run until `maxiter` or the time limit, and then
        # throw away a good solution. Without it, the solver's own convergence test is
        # part of the check.
        trust_info = not self._checks_forward_error()
        last_iterate = None

        time_limit = self._stage_time_limit()
        if time_limit is not None:
            deadline = time.perf_counter() + time_limit

            def callback(*args):
                nonlocal last_iterate
                # Most SciPy solvers pass the current iterate; GMRES with "pr_norm"
                # passes a residual norm.
                if args and isinstance(args[0], np.ndarray) and args[0].shape == demand.shape:
                    last_iterate = args[0]
                if time.perf_counter() > deadline:
                    raise _TimeLimitExceeded

            kwargs["callback"] = callback

        try:
            solution, info = self._call_solver(solver, matrix, demand, kwargs)
        except _TimeLimitExceeded:
            solution, info = (None if trust_info else last_iterate), "time limit"
        if solution is not None:
            solution = np.array(solution)
            if (info == 0 or not trust_info) and self._solution_is_acceptable(
                matrix, demand, solution
            ):
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
        # Without `iter_solver`, every changed matrix is solved directly, as before
        # `IterativeLCA` had a fallback stage.
        if not self.iter_solver:
            return True
        if stage == "fallback_iterative" and (
            not self.fallback_solver or self._reference_solve is None
        ):
            return True
        if (
            self.max_consecutive_failures is None
            or self._consecutive_failures[stage] < self.max_consecutive_failures
        ):
            return False
        # Skipped stages can't succeed to reset their failure count, so try them again
        # from time to time; Monte Carlo samples can get easier again.
        return self.retry_after is None or self._skipped_solves[stage] < self.retry_after

    def _iterative_solve(self, demand: np.ndarray) -> np.ndarray:
        self._prepare_matrix()
        matrix = self._prepared_technosphere_matrix
        x0 = self.guess if self.use_guess else None

        for stage in self.STAGES:
            if self._stage_skipped(stage):
                self._skipped_solves[stage] += 1
                continue
            with _single_threaded_openblas():
                solution = self._run_stage(stage, matrix, demand, x0)
            self._skipped_solves[stage] = 0
            if solution is not None:
                self._consecutive_failures[stage] = 0
                self.solver_stats[stage] += 1
                return solution
            self._consecutive_failures[stage] += 1
            if self._stage_skipped(stage):
                logger.info(
                    "Iterative stage %s failed %s times in a row; skipping it for %s solves",
                    stage,
                    self._consecutive_failures[stage],
                    "all further" if self.retry_after is None else self.retry_after,
                )

        self.solver_stats["direct"] += 1
        solution = self._direct_solve(demand)
        if self._pardiso_solver is not None and self._reference_solve is not None:
            # The direct solve left a factorization of this matrix in the object's own
            # solver, next to the reference factorization. It is almost never reused:
            # solves on the reference matrix use the reference factorization, and Monte
            # Carlo moves on to a new matrix. `LCA.__next__` only frees its LU factors,
            # so release all of it now; about 100 MB for ecoinvent.
            _release_pardiso_solver(self._pardiso_solver)
            # Nothing left for `_delete_solver_state` to release.
            self._used_pypardiso = False
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


def _tolerance_keyword(solver) -> str:
    """Return the name of the relative tolerance argument of ``solver``.

    SciPy added ``rtol`` in 1.12 and removed ``tol`` in 1.14. Retrying with ``tol`` after
    a ``TypeError`` would also hide ``TypeError`` raised inside the solver, behind an
    error about ``tol``, so check the signature instead."""
    try:
        parameters = inspect.signature(solver).parameters
    except (TypeError, ValueError):
        # No signature available, e.g. for some builtins; assume the modern API.
        return "rtol"
    if "rtol" not in parameters and "tol" in parameters:
        return "tol"
    return "rtol"


def _csr_entries(matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Return ``matrix[rows[k], cols[k]]`` for each ``k``, with zero for entries missing
    from the structure.

    ``matrix`` must be CSR in canonical form (sorted indices, no duplicates). Faster than
    fancy indexing, as entries in row-major order have sorted keys
    ``row * n_cols + col``."""
    n_cols = matrix.shape[1]
    entry_rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr))
    keys = entry_rows * n_cols + matrix.indices
    targets = np.asarray(rows, dtype=np.int64) * n_cols + np.asarray(cols, dtype=np.int64)
    if not len(keys):
        return np.zeros(len(targets), dtype=matrix.dtype)
    offsets = np.minimum(np.searchsorted(keys, targets), len(keys) - 1)
    return np.where(keys[offsets] == targets, matrix.data[offsets], 0)


def _csr_equal(a, b) -> bool:
    """Check that two CSR matrices have the same structure and values."""
    return (
        a.shape == b.shape
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.data, b.data)
    )


class _TimeLimitExceeded(Exception):
    """Raised from a solver callback to stop an iterative stage."""
