from pathlib import Path

import numpy as np
from scipy.sparse.linalg import bicgstab, cgs

from bw2calc import LCA, IterativeLCA

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

    lca = IterativeLCA({1: 1}, data_objs=basic, iter_solver=solver)
    lca.lci()
    assert calls == []

    first = lca.supply_array.copy()
    lca.solve_linear_system()
    assert len(calls) == 1
    assert np.allclose(calls[0], first)


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
