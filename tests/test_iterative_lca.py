import time
from pathlib import Path

import bw_processing as bwp
import numpy as np
import pytest
from scipy.sparse.linalg import bicgstab, cgs

from bw2calc import LCA, PYPARDISO, IterativeLCA, JacobiGMRESLCA
from bw2calc.errors import InvalidReferenceFlags
from bw2calc.utils import production_pivot_map, reference_entries

fixture_dir = Path(__file__).resolve().parent / "fixtures"
basic = [fixture_dir / "basic_fixture.zip"]
mc_basic = [fixture_dir / "mc_basic.zip"]


def test_iterative_default_solver_is_bicgstab():
    lca = IterativeLCA({1: 1}, data_objs=basic)
    assert lca.iter_solver is bicgstab


@pytest.mark.parametrize("iter_solver", [None, False])
def test_iterative_without_solver_solves_directly(iter_solver):
    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA({3: 1}, iter_solver=iter_solver, **kwargs)
    reference.lci()
    lca.lci()
    for _ in range(3):
        next(reference)
        next(lca)
        assert np.allclose(lca.supply_array, reference.supply_array)

    assert lca.solver_stats == {"reference": 1, "direct": 3}


def test_iterative_without_fallback_solver_skips_fallback_stage():
    def failing(matrix, demand, **kwargs):
        return np.zeros_like(demand), 1

    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    lca = IterativeLCA({3: 1}, iter_solver=failing, fallback_solver=None, **kwargs)
    lca.lci()
    next(lca)

    assert lca.solver_stats == {"reference": 1, "direct": 1}


def test_iterative_lci_matches_lca():
    reference = LCA({1: 1}, data_objs=basic)
    reference.lci()

    lca = IterativeLCA({1: 1}, data_objs=basic, direct_first_solve=False)
    lca.lci()

    assert np.allclose(lca.supply_array, reference.supply_array)


def test_iterative_custom_solver():
    reference = LCA({1: 1}, data_objs=basic)
    reference.lci()

    lca = IterativeLCA({1: 1}, data_objs=basic, iter_solver=cgs, direct_first_solve=False)
    lca.lci()

    assert np.allclose(lca.supply_array, reference.supply_array)


def test_iterative_direct_first_solve_skips_solver():
    calls = []

    def solver(matrix, demand, **kwargs):
        calls.append(kwargs["x0"])
        return bicgstab(matrix, demand, **kwargs)

    lca = IterativeLCA({3: 1}, data_objs=mc_basic, iter_solver=solver, use_distributions=True)
    lca.lci()
    assert calls == []

    first = lca.supply_array.copy()
    next(lca)
    assert len(calls) == 1
    assert np.allclose(calls[0], first)


def test_iterative_uses_reference_factorization_for_unchanged_matrix():
    calls = []

    def solver(matrix, demand, **kwargs):
        calls.append(kwargs["x0"])
        return bicgstab(matrix, demand, **kwargs)

    reference = LCA({2: 1}, data_objs=basic)
    reference.lci()

    lca = IterativeLCA({1: 1}, data_objs=basic, iter_solver=solver)
    lca.lci()
    lca.lci(demand={2: 1})

    assert calls == []
    assert lca.solver_stats == {"reference": 2}
    assert np.allclose(lca.supply_array, reference.supply_array)


def test_iterative_monte_carlo_matches_lca():
    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA({3: 1}, **kwargs)
    reference.lci()
    lca.lci()

    for _ in range(5):
        next(reference)
        next(lca)
        assert np.allclose(lca.supply_array, reference.supply_array)

    assert lca.solver_stats == {"reference": 1, "iterative": 5}


def test_iterative_falls_back_when_solver_reports_failure():
    def solver(matrix, demand, **kwargs):
        return np.zeros_like(demand), -10

    reference = LCA({1: 1}, data_objs=basic)
    reference.lci()

    lca = IterativeLCA({1: 1}, data_objs=basic, iter_solver=solver, direct_first_solve=False)
    lca.lci()

    assert np.allclose(lca.supply_array, reference.supply_array)


def test_iterative_falls_back_on_false_convergence():
    """BiCGSTAB can claim convergence with a wrong answer; the residual check catches it."""

    def solver(matrix, demand, **kwargs):
        return np.ones_like(demand), 0

    reference = LCA({1: 1}, data_objs=basic)
    reference.lci()

    lca = IterativeLCA({1: 1}, data_objs=basic, iter_solver=solver, direct_first_solve=False)
    lca.lci()

    assert np.allclose(lca.supply_array, reference.supply_array)


def test_iterative_falls_back_on_non_finite_solution():
    def solver(matrix, demand, **kwargs):
        return np.full_like(demand, np.nan), 0

    reference = LCA({1: 1}, data_objs=basic)
    reference.lci()

    lca = IterativeLCA({1: 1}, data_objs=basic, iter_solver=solver, direct_first_solve=False)
    lca.lci()

    assert np.allclose(lca.supply_array, reference.supply_array)


def test_iterative_does_not_modify_technosphere_matrix():
    lca = IterativeLCA({1: 1}, data_objs=basic, direct_first_solve=False)
    lca.lci()

    assert lca.technosphere_matrix is lca.technosphere_mm.matrix


def test_iterative_uses_jacobi_preconditioner_by_default():
    calls = []

    def solver(matrix, demand, **kwargs):
        calls.append(kwargs["M"])
        return bicgstab(matrix, demand, **kwargs)

    lca = IterativeLCA({1: 1}, data_objs=basic, iter_solver=solver, direct_first_solve=False)
    lca.lci()

    diagonal = lca.technosphere_matrix.diagonal()
    x = np.arange(1, len(diagonal) + 1, dtype=float)
    assert calls[0] is not None
    assert np.allclose(calls[0].matvec(x), x / diagonal)


def loop_datapackage(values):
    """Two activities which consume almost all of each other's output, a strong loop.

    The technosphere matrix is ``[[1, -a], [-a, 1]]``, with ``a`` taking one of ``values``
    in each iteration. For ``a`` close to one it is ill-conditioned: its eigenvalues are
    ``1 - a`` for ``[1, 1]`` and ``1 + a`` for ``[1, -1]``."""
    dp = bwp.create_datapackage(sequential=True)
    n = len(values)
    dp.add_persistent_array(
        matrix="technosphere_matrix",
        indices_array=np.array([(1, 101), (2, 102), (1, 102), (2, 101)], dtype=bwp.INDICES_DTYPE),
        data_array=np.vstack([np.ones(n), np.ones(n), values, values]),
        flip_array=np.array([False, False, True, True]),
        name="loop-technosphere",
    )
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        indices_array=np.array([(10, 101), (10, 102)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([1.0, 2.0]),
        name="loop-biosphere",
    )
    return dp


LOOP_VALUES = 1 - np.array([1.0, 1.1, 1.2, 1.3]) * 1e-6
# Along the eigenvector with eigenvalue `1 + a`, so the exact solution is well scaled.
LOOP_DEMAND = {1: 1, 2: -1}


def loop_lca(cls, **kwargs):
    return cls(LOOP_DEMAND, data_objs=[loop_datapackage(LOOP_VALUES)], use_arrays=True, **kwargs)


def small_residual_wrong_solver(matrix, demand, **kwargs):
    """Return the exact solution plus an error along the ill-conditioned direction.

    The error of about 0.2% only gives a relative residual of about 1e-9."""
    exact = np.linalg.solve(matrix.toarray(), demand)
    return exact + 1e-3, 0


def test_iterative_residual_check_misses_error_in_ill_conditioned_system():
    """Documents why the forward error check is needed."""
    lca = loop_lca(IterativeLCA, iter_solver=small_residual_wrong_solver, max_error=None)
    reference = loop_lca(LCA)
    lca.lci()
    reference.lci()
    next(lca)
    next(reference)

    assert lca.solver_stats["iterative"] == 1
    assert not np.allclose(lca.supply_array, reference.supply_array, rtol=1e-4, atol=0)


def test_iterative_forward_error_check_rejects_small_residual_wrong_solution():
    lca = loop_lca(IterativeLCA, iter_solver=small_residual_wrong_solver)
    reference = loop_lca(LCA)
    lca.lci()
    reference.lci()

    for _ in range(len(LOOP_VALUES) - 1):
        next(lca)
        next(reference)
        assert np.allclose(lca.supply_array, reference.supply_array, rtol=1e-6, atol=0)

    assert lca.solver_stats["iterative"] == 0
    assert lca.solver_stats["fallback_iterative"] + lca.solver_stats["direct"] == 3


def test_iterative_estimate_error_tracks_true_error():
    lca = loop_lca(IterativeLCA)
    lca.lci()
    next(lca)
    lca._prepare_matrix()
    matrix = lca._prepared_technosphere_matrix
    exact = np.linalg.solve(matrix.toarray(), lca.demand_array)
    wrong = exact + 1e-3

    true_error = np.linalg.norm(wrong - exact) / np.linalg.norm(wrong)
    estimate = lca.estimate_error(matrix, lca.demand_array, wrong)
    assert 0.5 < estimate / true_error < 2
    assert lca.estimate_error(matrix, lca.demand_array, exact) < 1e-8


def test_iterative_falls_back_to_reference_preconditioned_solver():
    def solver(matrix, demand, **kwargs):
        return np.ones_like(demand), 0

    fallback_calls = []

    def fallback(matrix, demand, **kwargs):
        fallback_calls.append(kwargs["M"])
        return bicgstab(matrix, demand, **kwargs)

    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA({3: 1}, iter_solver=solver, fallback_solver=fallback, **kwargs)
    reference.lci()
    lca.lci()
    next(reference)
    next(lca)

    assert np.allclose(lca.supply_array, reference.supply_array)
    assert lca.solver_stats == {"reference": 1, "fallback_iterative": 1}
    # The fallback is preconditioned with the reference factorization, i.e. `A0^-1`.
    x = np.array([1.0, 2.0])
    assert np.allclose(lca._reference_matrix @ fallback_calls[0].matvec(x), x)


def test_iterative_time_limit_stops_stage():
    def slow_solver(matrix, demand, callback=None, **kwargs):
        for _ in range(1000):
            time.sleep(0.001)
            callback(None)
        raise AssertionError("time limit not applied")

    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA({3: 1}, iter_solver=slow_solver, time_limit=0.01, **kwargs)
    reference.lci()
    lca.lci()
    start = time.perf_counter()
    next(reference)
    next(lca)

    assert time.perf_counter() - start < 0.5
    assert np.allclose(lca.supply_array, reference.supply_array)
    assert lca.solver_stats["fallback_iterative"] == 1


def exact_unconverged_solver(matrix, demand, **kwargs):
    """Return the exact solution, but report that `maxiter` was reached."""
    return np.linalg.solve(matrix.toarray(), demand), 1


def test_iterative_accepts_unconverged_solution_passing_forward_error_check():
    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA({3: 1}, iter_solver=exact_unconverged_solver, **kwargs)
    reference.lci()
    lca.lci()
    next(reference)
    next(lca)

    assert np.allclose(lca.supply_array, reference.supply_array)
    assert lca.solver_stats == {"reference": 1, "iterative": 1}


def test_iterative_rejects_unconverged_solution_without_forward_error_check():
    lca = IterativeLCA(
        {1: 1}, data_objs=basic, iter_solver=exact_unconverged_solver, direct_first_solve=False
    )
    lca.lci()

    assert lca.solver_stats == {"direct": 1}


def test_iterative_accepts_last_iterate_at_time_limit():
    def slow_solver(matrix, demand, callback=None, **kwargs):
        exact = np.linalg.solve(matrix.toarray(), demand)
        for _ in range(1000):
            time.sleep(0.001)
            callback(exact)
        raise AssertionError("time limit not applied")

    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA({3: 1}, iter_solver=slow_solver, time_limit=0.01, **kwargs)
    reference.lci()
    lca.lci()
    next(reference)
    next(lca)

    assert np.allclose(lca.supply_array, reference.supply_array)
    assert lca.solver_stats == {"reference": 1, "iterative": 1}


def test_iterative_auto_time_limit_is_first_direct_solve_time():
    lca = IterativeLCA({1: 1}, data_objs=basic)
    assert lca._stage_time_limit() is None
    lca.lci()
    assert lca._reference_time > 0
    assert lca._stage_time_limit() == max(lca._reference_time, IterativeLCA.MIN_AUTO_TIME_LIMIT)


@pytest.mark.parametrize("time_limit", [None, 2.5])
def test_iterative_explicit_time_limit(time_limit):
    lca = IterativeLCA({1: 1}, data_objs=basic, time_limit=time_limit)
    lca.lci()
    assert lca._stage_time_limit() == time_limit


def test_iterative_skips_stage_after_consecutive_failures():
    calls = []

    def solver(matrix, demand, **kwargs):
        calls.append(1)
        return np.ones_like(demand), 0

    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA({3: 1}, iter_solver=solver, max_consecutive_failures=2, **kwargs)
    reference.lci()
    lca.lci()
    for _ in range(4):
        next(reference)
        next(lca)
        assert np.allclose(lca.supply_array, reference.supply_array)

    assert len(calls) == 2
    assert lca.solver_stats == {"reference": 1, "fallback_iterative": 4}

    # New matrices give the stage another chance.
    lca.load_lci_data()
    lca.lci_calculation()
    next(lca)
    assert len(calls) == 3


@pytest.mark.parametrize("retry_after", [3, None])
def test_iterative_retries_skipped_stage(retry_after):
    """A run of hard Monte Carlo samples mustn't turn the stage off for all later ones."""
    calls = []

    def solver(matrix, demand, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            return np.ones_like(demand), 0
        return bicgstab(matrix, demand, **kwargs)

    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA(
        {3: 1}, iter_solver=solver, max_consecutive_failures=2, retry_after=retry_after, **kwargs
    )
    reference.lci()
    lca.lci()
    for _ in range(7):
        next(reference)
        next(lca)
        assert np.allclose(lca.supply_array, reference.supply_array)

    if retry_after is None:
        assert len(calls) == 2
        assert lca.solver_stats == {"reference": 1, "fallback_iterative": 7}
    else:
        # Fails twice, is skipped for three samples, and then succeeds again.
        assert len(calls) == 4
        assert lca.solver_stats == {"reference": 1, "fallback_iterative": 5, "iterative": 2}


def test_iterative_skipped_stage_failing_again_is_skipped_again():
    calls = []

    def solver(matrix, demand, **kwargs):
        calls.append(1)
        return np.ones_like(demand), 0

    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    lca = IterativeLCA(
        {3: 1}, iter_solver=solver, max_consecutive_failures=2, retry_after=3, **kwargs
    )
    lca.lci()
    for _ in range(10):
        next(lca)

    # Samples 1-2 fail, 3-5 are skipped, 6 fails, 7-9 are skipped, 10 fails.
    assert len(calls) == 4


def test_iterative_uses_direct_solver_when_all_stages_fail():
    def failing(matrix, demand, **kwargs):
        return np.zeros_like(demand), 1

    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA({3: 1}, iter_solver=failing, fallback_solver=failing, **kwargs)
    reference.lci()
    lca.lci()
    next(reference)
    next(lca)

    assert np.allclose(lca.supply_array, reference.supply_array)
    assert lca.solver_stats == {"reference": 1, "direct": 1}


@pytest.mark.skipif(not PYPARDISO, reason="needs pypardiso")
def test_iterative_direct_fallback_releases_its_factorization():
    def failing(matrix, demand, **kwargs):
        return np.zeros_like(demand), 1

    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    reference = LCA({3: 1}, **kwargs)
    lca = IterativeLCA({3: 1}, iter_solver=failing, fallback_solver=failing, **kwargs)
    reference.lci()
    lca.lci()
    next(reference)
    next(lca)

    assert lca.solver_stats == {"reference": 1, "direct": 1}
    assert lca._pardiso_solver.factorized_A.nnz == 0
    # So `__next__` doesn't release it a second time.
    assert not lca._used_pypardiso
    # The reference factorization is separate, and still works.
    b = np.array([1.0, 2.0])
    assert np.allclose(lca._reference_matrix @ lca._reference_solve(b), b)


def test_iterative_reference_survives_monte_carlo_iteration():
    """`LCA.__next__` frees the object's direct solver factorization, but not the reference."""
    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    lca = IterativeLCA({3: 1}, **kwargs)
    lca.lci()
    reference_solve = lca._reference_solve
    next(lca)

    assert lca._reference_solve is reference_solve
    b = np.array([1.0, 2.0])
    assert np.allclose(lca._reference_matrix @ reference_solve(b), b)


def test_iterative_reference_factorization_survives_other_direct_solves():
    """The reference must not share the object's solver for direct solves, which
    `LCA.__next__` resets, or be affected by other objects' solves."""
    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    lca = IterativeLCA({3: 1}, **kwargs)
    other = LCA({3: 1}, **kwargs)
    lca.lci()
    other.lci()
    next(other)
    lca._direct_solve(lca.demand_array)

    b = np.array([1.0, 2.0])
    assert np.allclose(lca._reference_matrix @ lca._reference_solve(b), b)


def test_iterative_failed_first_solve_leaves_no_partial_reference():
    def failing(matrix, demand, **kwargs):
        return np.zeros_like(demand), 1

    reference = LCA({3: 1}, data_objs=mc_basic)
    reference.lci()

    lca = IterativeLCA({3: 1}, data_objs=mc_basic, iter_solver=failing)
    lca.load_lci_data()
    lca.build_demand_array()
    with pytest.raises(ValueError):
        lca.solve_linear_system(np.ones(5))

    assert lca._reference_solve is None
    assert lca._reference_finalizer is None
    lca.lci_calculation()
    assert np.allclose(lca.supply_array, reference.supply_array)
    assert lca.solver_stats == {"reference": 1}


def test_iterative_releases_reference_on_new_matrices():
    lca = IterativeLCA({1: 1}, data_objs=basic)
    lca.lci()
    finalizer = lca._reference_finalizer
    lca.load_lci_data()

    assert lca._reference_solve is None
    assert finalizer is None or not finalizer.alive


@pytest.mark.parametrize("cls", [IterativeLCA, JacobiGMRESLCA])
@pytest.mark.parametrize("in_place", [False, True])
def test_iterative_solves_edited_technosphere_matrix(cls, in_place):
    """Replacing or editing `technosphere_matrix` between solves, without `__next__` or
    `load_lci_data`, must not solve the previous matrix."""
    lca = cls({3: 1}, data_objs=mc_basic)
    lca.lci()

    for _ in range(2):
        if in_place:
            data = lca.technosphere_matrix.data
            data[data < 0] *= 2
        else:
            matrix = lca.technosphere_matrix.copy()
            matrix.data[matrix.data < 0] *= 2
            lca.technosphere_matrix = matrix
        lca.lci_calculation()

        expected = np.linalg.solve(lca.technosphere_matrix.toarray(), lca.demand_array)
        assert np.allclose(lca.supply_array, expected)


@pytest.mark.skipif(PYPARDISO, reason="`decompose_technosphere` is a no-op with pypardiso")
@pytest.mark.parametrize("direct_first_solve", [True, False])
def test_iterative_direct_fallback_ignores_stale_factorization(direct_first_solve):
    """`lci(factorize=True)` stores `self.solver`, which `LCA.solve_linear_system` uses
    even after the technosphere matrix was edited."""

    def failing(matrix, demand, **kwargs):
        return np.zeros_like(demand), 1

    lca = IterativeLCA(
        {3: 1},
        data_objs=mc_basic,
        iter_solver=failing,
        fallback_solver=failing,
        direct_first_solve=direct_first_solve,
    )
    lca.lci(factorize=True)
    assert hasattr(lca, "solver")

    direct_solves = lca.solver_stats["direct"]

    data = lca.technosphere_matrix.data
    data[data < 0] *= 2
    lca.lci_calculation()

    expected = np.linalg.solve(lca.technosphere_matrix.toarray(), lca.demand_array)
    assert np.allclose(lca.supply_array, expected)
    assert lca.solver_stats["direct"] == direct_solves + 1


def test_iterative_solver_type_error_is_not_hidden():
    def solver(matrix, demand, x0=None, rtol=1e-5, atol=0.0, maxiter=None, M=None):
        raise TypeError("bug inside the solver")

    lca = IterativeLCA({1: 1}, data_objs=basic, iter_solver=solver, direct_first_solve=False)
    with pytest.raises(TypeError, match="bug inside the solver"):
        lca.lci()


def test_iterative_passes_tol_to_solvers_without_rtol():
    calls = []

    def solver(matrix, demand, x0=None, tol=1e-5, atol=0.0, maxiter=None, M=None):
        calls.append(tol)
        return np.linalg.solve(matrix.toarray(), demand), 0

    lca = IterativeLCA(
        {1: 1}, data_objs=basic, iter_solver=solver, direct_first_solve=False, rtol=1e-9
    )
    lca.lci()

    assert calls == [1e-9]
    assert lca.solver_stats == {"iterative": 1}


def pivot_datapackage(matrix, flags=None, samples=None, sequential=True):
    """Technosphere ``matrix`` with product rows ``1..n`` and activity columns
    ``1001..1000+n``, i.e. separate product and process ids, as given; no flips.

    ``flags`` is a boolean matrix of the reference exchanges, or ``None`` for no
    ``reference_array``. ``samples`` is a list of matrices with the same structure, used
    as data array columns."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    structure = matrix != 0
    if samples is not None:
        for sample in samples:
            structure |= np.asarray(sample) != 0
    if flags is not None:
        structure |= flags
    rows, cols = np.nonzero(structure)
    indices = np.array(list(zip(rows + 1, cols + 1001)), dtype=bwp.INDICES_DTYPE)
    reference = None if flags is None else np.asarray(flags)[rows, cols]

    dp = bwp.create_datapackage(sequential=sequential)
    kwargs = dict(
        matrix="technosphere_matrix",
        indices_array=indices,
        flip_array=np.zeros(len(rows), dtype=bool),
        reference_array=reference,
        name="pivot-technosphere",
    )
    if samples is None:
        dp.add_persistent_vector(data_array=matrix[rows, cols], **kwargs)
    else:
        data = np.stack([np.asarray(s, dtype=float)[rows, cols] for s in samples], axis=1)
        dp.add_persistent_array(data_array=data, **kwargs)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        indices_array=np.array([(1, 1001 + j) for j in range(n)], dtype=bwp.INDICES_DTYPE),
        data_array=np.arange(1.0, n + 1),
        name="pivot-biosphere",
    )
    return dp


def permuted_system(n=40, seed=0, negative=True):
    """Return ``(matrix, flags)`` for a technosphere ``(I - T) D`` with dense, small inputs
    ``T`` and production amounts ``D``, with its rows shuffled.

    As every entry is nonzero, the diagonal holds inputs, not production amounts."""
    rng = np.random.default_rng(seed)
    inputs = rng.random((n, n)) * 0.9 / n
    np.fill_diagonal(inputs, 0)
    production = rng.uniform(1, 5, n)
    if negative:
        # Waste treatment, with a negative reference amount.
        production[0] = -production[0]
    matrix = (np.eye(n) - inputs) * production
    order = rng.permutation(n)
    while np.any(order == np.arange(n)):
        order = rng.permutation(n)
    flags = np.eye(n, dtype=bool)
    return matrix[order], flags[order]


def counting_solver(counts):
    def solver(matrix, demand, callback=None, **kwargs):
        counts.append(0)

        def count(*args):
            counts[-1] += 1
            if callback is not None:
                callback(*args)

        return bicgstab(matrix, demand, callback=count, **kwargs)

    return solver


def swapped_system():
    """Products sort in the opposite order to their processes, so the diagonal is all
    zeros. The explicit zero on the diagonal keeps `[1, 1]` in the structure."""
    matrix = np.array([[0.0, 1.0], [2.0, -0.5]])
    flags = np.array([[False, True], [True, False]])
    return matrix, flags


def test_iterative_production_pivots_solve_zero_diagonal():
    matrix, flags = swapped_system()
    lca = IterativeLCA(
        {1: 1}, data_objs=[pivot_datapackage(matrix, flags)], direct_first_solve=False
    )
    lca.lci()

    assert lca.solver_stats == {"iterative": 1}
    assert lca.preconditioner_source == "production_pivots"
    assert np.allclose(lca.supply_array, np.linalg.solve(matrix, [1, 0]))
    x = np.array([3.0, 4.0])
    # Activity `j` reads the residual of its product row `p(j)`.
    assert np.allclose(lca._get_preconditioner().matvec(x), [4.0 / 2.0, 3.0 / 1.0])


def test_iterative_without_flags_keeps_diagonal_behaviour():
    matrix, _ = swapped_system()
    lca = IterativeLCA({1: 1}, data_objs=[pivot_datapackage(matrix)], direct_first_solve=False)
    lca.lci()

    assert lca._get_preconditioner() is None
    assert lca.preconditioner_source is None
    assert lca._pivot_pairing == "diagonal"
    assert np.allclose(lca.supply_array, np.linalg.solve(matrix, [1, 0]))


def test_iterative_caches_missing_preconditioner(monkeypatch):
    matrix, _ = swapped_system()
    lca = IterativeLCA({1: 1}, data_objs=[pivot_datapackage(matrix)], direct_first_solve=False)
    lca.lci()
    assert lca._get_preconditioner() is None

    # A zero pivot leaves this matrix without a preconditioner; don't look again for
    # every demand on it.
    monkeypatch.setattr(lca, "build_preconditioner", lambda: pytest.fail("rebuilt"))
    assert lca._get_preconditioner() is None
    lca.lci_calculation()
    assert np.allclose(lca.supply_array, np.linalg.solve(matrix, [1, 0]))


def test_iterative_preconditioner_source_set_by_the_iterative_stage():
    matrix, flags = swapped_system()
    lca = IterativeLCA({1: 1}, data_objs=[pivot_datapackage(matrix, flags)])
    # The first solve is direct: no iterative stage has run.
    lca.lci()
    assert lca.preconditioner_source is None

    # Inspecting the preconditioner isn't using it.
    assert lca.build_preconditioner() is not None
    assert lca.preconditioner_source is None


def test_iterative_replaced_matrix_of_another_shape_uses_diagonal():
    import scipy.sparse as sps

    matrix, flags = permuted_system(n=6)
    lca = IterativeLCA(
        {1: 1}, data_objs=[pivot_datapackage(matrix, flags)], direct_first_solve=False
    )
    lca.lci()
    assert lca.preconditioner_source == "production_pivots"

    # The flags in `technosphere_mm` don't fit the new matrix, so `auto` uses the diagonal.
    smaller = np.array([[2.0, -0.5], [0.0, 4.0]])
    lca.technosphere_matrix = sps.csr_matrix(smaller)
    demand = np.array([1.0, 0.0])
    solution = lca.solve_linear_system(demand)

    assert np.allclose(solution, np.linalg.solve(smaller, demand))
    assert lca.preconditioner_source == "diagonal"
    assert lca._pivot_shape == (2, 2)


def test_iterative_production_pivots_need_fewer_iterations_than_wrong_diagonal():
    matrix, flags = permuted_system()
    counts = {}
    for pairing in ("auto", "diagonal"):
        counts[pairing] = []
        lca = IterativeLCA(
            {1: 1},
            data_objs=[pivot_datapackage(matrix, flags)],
            iter_solver=counting_solver(counts[pairing]),
            direct_first_solve=False,
            jacobi_pairing=pairing,
        )
        lca.lci()
        assert np.allclose(lca.supply_array, np.linalg.solve(matrix, lca.demand_array))

    assert lca.preconditioner_source == "diagonal"
    assert len(counts["auto"]) == 1
    assert counts["auto"][0] < counts["diagonal"][0]


def test_iterative_production_pivots_on_diagonal_match_plain_diagonal():
    matrix, flags = permuted_system(negative=False)
    order = np.argmax(flags, axis=0)
    # Undo the row shuffle: processes and their products sort the same way.
    matrix, flags = matrix[order], flags[order]
    lca = IterativeLCA(
        {1: 1}, data_objs=[pivot_datapackage(matrix, flags)], direct_first_solve=False
    )
    lca.lci()

    assert lca.preconditioner_source == "production_pivots"
    # Identity pivot map: same operator as the diagonal, without the gather.
    assert lca._pivot_rows is None
    x = np.arange(1.0, len(matrix) + 1)
    assert np.allclose(lca._get_preconditioner().matvec(x), x / np.diag(matrix))


def broken_flags(kind):
    matrix, flags = permuted_system()
    flags = flags.copy()
    if kind == "missing":
        flags[:, 3] = False
    elif kind == "conflicting":
        # A second reference exchange, in another row, for column 3.
        flags[np.argmin(flags[:, 3]), 3] = True
    else:
        # Column 4 claims the reference product of column 3.
        flags[:, 4] = flags[:, 3]
    return matrix, flags


@pytest.mark.parametrize("kind", ["missing", "conflicting", "shared"])
def test_iterative_unusable_flags_fall_back_to_diagonal(kind, caplog):
    caplog.set_level("INFO", logger="bw2calc")
    matrix, flags = broken_flags(kind)
    lca = IterativeLCA(
        {1: 1}, data_objs=[pivot_datapackage(matrix, flags)], direct_first_solve=False
    )
    lca.lci()

    assert lca._pivot_pairing == "diagonal"
    assert np.allclose(lca.supply_array, np.linalg.solve(matrix, lca.demand_array))
    messages = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert any("using the diagonal" in m and "of 40 columns" in m for m in messages)


def test_iterative_no_flags_warns_to_update_bw2data():
    matrix = permuted_system()[0]
    lca = IterativeLCA(
        {1: 1}, data_objs=[pivot_datapackage(matrix, None)], direct_first_solve=False
    )
    with pytest.warns(UserWarning, match="Update bw2data"):
        lca.lci()

    assert lca._pivot_pairing == "diagonal"


@pytest.mark.parametrize("kind", ["missing", "conflicting", "shared", "none"])
def test_iterative_reference_pairing_raises_for_unusable_flags(kind):
    if kind == "none":
        matrix, flags = permuted_system()[0], None
    else:
        matrix, flags = broken_flags(kind)
    lca = IterativeLCA(
        {1: 1},
        data_objs=[pivot_datapackage(matrix, flags)],
        direct_first_solve=False,
        jacobi_pairing="reference",
    )
    with pytest.raises(InvalidReferenceFlags):
        lca.lci()


def test_iterative_reference_pairing_raises_before_the_first_solve():
    matrix = permuted_system()[0]
    # Default `direct_first_solve=True`: the first solve wouldn't build the pivot map.
    lca = IterativeLCA({1: 1}, data_objs=[pivot_datapackage(matrix)], jacobi_pairing="reference")
    with pytest.raises(InvalidReferenceFlags, match="No reference exchange flags"):
        lca.load_lci_data()


def test_iterative_diagonal_pairing_ignores_flags():
    matrix, flags = swapped_system()
    lca = IterativeLCA(
        {1: 1},
        data_objs=[pivot_datapackage(matrix, flags)],
        direct_first_solve=False,
        jacobi_pairing="diagonal",
    )
    lca.lci()

    assert lca._pivot_pairing == "diagonal"
    assert lca.solver_stats == {"direct": 1}


def test_iterative_invalid_jacobi_pairing():
    with pytest.raises(ValueError, match="jacobi_pairing"):
        IterativeLCA({1: 1}, data_objs=basic, jacobi_pairing="matching")


def test_iterative_production_pivots_monte_carlo(monkeypatch):
    import bw2calc.iterative_lca as module

    builds = []
    original = module.production_pivot_map
    monkeypatch.setattr(
        module, "production_pivot_map", lambda *args: builds.append(1) or original(*args)
    )

    matrix, flags = permuted_system()
    pivot = (np.argmax(flags[:, 5]), 5)
    samples = [matrix * (1 + 0.1 * k) for k in range(4)]
    # A sampled zero production amount.
    samples[2] = samples[2].copy()
    samples[2][pivot] = 0
    lca = IterativeLCA(
        {1: 1},
        data_objs=[pivot_datapackage(matrix, flags, samples=samples)],
        use_arrays=True,
        direct_first_solve=False,
    )
    lca.lci()

    for k in range(4):
        if k:
            next(lca)
        expected = np.linalg.solve(samples[k], lca.demand_array)
        assert np.allclose(lca.supply_array, expected)
        preconditioner = lca._get_preconditioner()
        if k == 2:
            assert preconditioner is None
            assert lca.preconditioner_source is None
        else:
            # Pivots come from the current sample.
            x = np.zeros(len(matrix))
            x[pivot[0]] = 1
            assert np.isclose(preconditioner.matvec(x)[5], 1 / samples[k][pivot])
            assert lca.preconditioner_source == "production_pivots"

    assert builds == [1]
    assert lca.solver_stats["iterative"] == 3


def test_reference_entries_align_with_masks_and_duplicates():
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        # Duplicate (1, 1001) entries are summed, so `row_matrix` has fewer entries than
        # the flags; only the second one is the reference exchange.
        indices_array=np.array(
            [(1, 1001), (2, 1001), (1, 1001), (2, 1002)], dtype=bwp.INDICES_DTYPE
        ),
        data_array=np.array([0.4, -0.5, 0.6, 1.0]),
        reference_array=np.array([False, False, True, True]),
        name="first",
    )
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        indices_array=np.array([(1, 1002), (2, 1002)], dtype=bwp.INDICES_DTYPE),
        data_array=np.array([-0.1, 2.0]),
        reference_array=np.array([False, True]),
        name="second",
    )
    lca = LCA({1: 1}, data_objs=[dp])
    lca.load_lci_data()

    rows, cols = reference_entries(lca.technosphere_mm)
    pairs = sorted(zip(rows.tolist(), cols.tolist()))
    row, col = lca.dicts.product, lca.dicts.activity
    assert pairs == sorted([(row[1], col[1001]), (row[2], col[1002]), (row[2], col[1002])])
    pivot_rows = production_pivot_map(rows, cols, lca.technosphere_matrix.shape)
    assert pivot_rows.tolist() == [row[1], row[2]]


def test_production_pivot_map_messages():
    rows, cols = np.array([0, 1, 1]), np.array([0, 1, 2])
    with pytest.raises(InvalidReferenceFlags, match="1 columns have no reference exchange"):
        production_pivot_map(rows[:2], cols[:2], (3, 3))
    with pytest.raises(InvalidReferenceFlags, match="1 rows are the reference product"):
        production_pivot_map(rows, cols, (3, 3))
    # One shared row, however many columns claim it.
    with pytest.raises(InvalidReferenceFlags, match="1 rows are the reference product"):
        production_pivot_map(np.array([0, 0, 0]), cols, (3, 3))
    with pytest.raises(InvalidReferenceFlags, match="not square"):
        production_pivot_map(rows, cols, (2, 3))
    with pytest.raises(InvalidReferenceFlags, match="don't fit"):
        production_pivot_map(np.array([0, 1, 3]), cols, (3, 3))
    assert production_pivot_map(np.array([2, 0, 1]), cols, (3, 3)).tolist() == [2, 0, 1]
