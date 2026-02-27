from pathlib import Path

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
    jacobi._matrix_prepared = False
    jacobi._cached_preconditioner = None
    jacobi.guess = None

    demand = np.array([1.0, 2.0])
    jacobi.solve_linear_system(demand)
    jacobi.solve_linear_system(demand)

    assert calls[0] is None
    assert np.allclose(calls[1], np.array([0.2, 0.6]))
