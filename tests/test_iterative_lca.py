import time
from pathlib import Path

import bw_processing as bwp
import numpy as np
import pytest
from scipy.sparse.linalg import bicgstab, cgs

from bw2calc import LCA, PYPARDISO, IterativeLCA, JacobiGMRESLCA

fixture_dir = Path(__file__).resolve().parent / "fixtures"
basic = [fixture_dir / "basic_fixture.zip"]
mc_basic = [fixture_dir / "mc_basic.zip"]


def test_iterative_default_solver_is_bicgstab():
    lca = IterativeLCA({1: 1}, data_objs=basic)
    assert lca.iter_solver is bicgstab


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
def test_iterative_direct_fallback_releases_global_factorization():
    from pypardiso.scipy_aliases import pypardiso_solver

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
    assert pypardiso_solver.factorized_A.nnz == 0
    # The reference factorization is separate, and still works.
    b = np.array([1.0, 2.0])
    assert np.allclose(lca._reference_matrix @ lca._reference_solve(b), b)


def test_iterative_reference_survives_monte_carlo_iteration():
    """`LCA.__next__` frees pypardiso's global factorization; `IterativeLCA` mustn't."""
    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    lca = IterativeLCA({3: 1}, **kwargs)
    lca.lci()
    reference_solve = lca._reference_solve
    next(lca)

    assert lca._reference_solve is reference_solve
    b = np.array([1.0, 2.0])
    assert np.allclose(lca._reference_matrix @ reference_solve(b), b)


def test_iterative_reference_factorization_survives_other_direct_solves():
    """The reference must not share pypardiso's global solver, which `spsolve` uses and
    `LCA.__next__` resets."""
    kwargs = dict(data_objs=mc_basic, seed_override=42, use_distributions=True)
    lca = IterativeLCA({3: 1}, **kwargs)
    other = LCA({3: 1}, **kwargs)
    lca.lci()
    other.lci()
    next(other)
    lca._direct_solve(lca.demand_array)

    b = np.array([1.0, 2.0])
    assert np.allclose(lca._reference_matrix @ lca._reference_solve(b), b)


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
