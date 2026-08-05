from pathlib import Path

import bw_processing as bwp
import numpy as np
import scipy.sparse as sps

from bw2calc import LCA, JacobiGMRESLCA

fixture_dir = Path(__file__).resolve().parent / "fixtures"


def test_jacobi_gmres_lci_matches_lca_basic_fixture():
    packages = [fixture_dir / "basic_fixture.zip"]

    reference = LCA({1: 1}, data_objs=packages)
    reference.lci()

    jacobi = JacobiGMRESLCA({1: 1}, data_objs=packages)
    jacobi.lci()

    assert np.allclose(jacobi.supply_array, reference.supply_array)


def test_jacobi_gmres_returns_no_preconditioner_for_zero_diagonal():
    jacobi = JacobiGMRESLCA.__new__(JacobiGMRESLCA)
    jacobi.technosphere_matrix = sps.csc_matrix([[0.0, 1.0], [1.0, 2.0]])
    jacobi._prepared_technosphere_matrix = None
    jacobi._cached_preconditioner = None

    preconditioner = jacobi._build_jacobi_preconditioner()

    assert preconditioner is None


def test_jacobi_gmres_uses_previous_solution_as_guess(monkeypatch):
    calls = []

    def fake_gmres(matrix, demand, **kwargs):
        calls.append(kwargs.get("x0"))
        return np.array([0.2, 0.6]), 0

    monkeypatch.setattr("bw2calc.jacobi_gmres_lca.gmres", fake_gmres)

    jacobi = JacobiGMRESLCA.__new__(JacobiGMRESLCA)
    jacobi.technosphere_matrix = sps.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    jacobi.rtol = 1e-8
    jacobi.atol = 0.0
    jacobi.restart = 50
    jacobi.maxiter = 300
    jacobi.use_guess = True
    jacobi._prepared_technosphere_matrix = None
    jacobi._cached_preconditioner = None
    jacobi.guess = None

    demand = np.array([1.0, 2.0])
    jacobi.solve_linear_system(demand)
    jacobi.solve_linear_system(demand)

    assert calls[0] is None
    assert np.allclose(calls[1], np.array([0.2, 0.6]))


def test_jacobi_gmres_keeps_monte_carlo_technosphere_current():
    packages = [fixture_dir / "mc_basic.zip"]
    jacobi = JacobiGMRESLCA(
        {3: 1},
        data_objs=packages,
        seed_override=42,
        use_distributions=True,
    )
    jacobi.lci()

    initial_sum = jacobi.technosphere_matrix.sum()
    next(jacobi)

    assert jacobi.technosphere_matrix is jacobi.technosphere_mm.matrix
    assert jacobi.technosphere_matrix.sum() != initial_sum


def test_jacobi_gmres_falls_back_to_direct_solver(monkeypatch):
    packages = [fixture_dir / "basic_fixture.zip"]

    def fake_gmres(*args, **kwargs):
        matrix, demand = args[:2]
        return np.zeros(matrix.shape[1], dtype=demand.dtype), 1

    monkeypatch.setattr("bw2calc.jacobi_gmres_lca.gmres", fake_gmres)

    reference = LCA({1: 1}, data_objs=packages)
    reference.lci()

    jacobi = JacobiGMRESLCA({1: 1}, data_objs=packages)
    jacobi.lci()

    assert np.allclose(reference.supply_array, jacobi.supply_array)


def sequential_technosphere_override(values):
    """Override cell ``(2, 101)`` of `basic_fixture` with a known sequence."""
    dp = bwp.create_datapackage(sequential=True)
    dp.add_persistent_array(
        matrix="technosphere_matrix",
        indices_array=np.array([(2, 101)], dtype=bwp.INDICES_DTYPE),
        data_array=values.reshape((1, -1)),
        flip_array=np.array([True]),
        name="technosphere-override",
    )
    return dp


def test_jacobi_gmres_advances_sequential_technosphere_array():
    """Presampled technosphere values must advance with each iteration.

    `JacobiGMRESLCA` used to detach `technosphere_matrix` from the mapped matrix,
    which silently froze presampled arrays on their first column and produced
    plausible but wrong results. See
    https://github.com/brightway-lca/brightway2-calc/issues/157
    """
    values = np.array([0.5, 0.25, 0.75, 0.125])
    packages = [fixture_dir / "basic_fixture.zip"]

    jacobi = JacobiGMRESLCA(
        {1: 1},
        data_objs=packages + [sequential_technosphere_override(values)],
        use_arrays=True,
    )
    jacobi.lci()
    row, col = jacobi.dicts.product[2], jacobi.dicts.activity[101]

    observed = []
    for _ in values:
        observed.append(-jacobi.technosphere_matrix[row, col])
        next(jacobi)

    assert np.allclose(observed, values)


def test_jacobi_gmres_sequential_supply_matches_lca():
    """The solve must use the advanced values, not just report them."""
    values = np.array([0.5, 0.25, 0.75, 0.125])
    packages = [fixture_dir / "basic_fixture.zip"]

    jacobi = JacobiGMRESLCA(
        {1: 1},
        data_objs=packages + [sequential_technosphere_override(values)],
        use_arrays=True,
    )
    reference = LCA(
        {1: 1},
        data_objs=packages + [sequential_technosphere_override(values)],
        use_arrays=True,
    )
    jacobi.lci()
    reference.lci()

    for _ in values:
        assert np.allclose(jacobi.supply_array, reference.supply_array)
        # A frozen technosphere still varies over iterations if only the first
        # value is used, so compare against `LCA` rather than against itself.
        assert not np.allclose(jacobi.supply_array, 0)
        next(jacobi)
        next(reference)
