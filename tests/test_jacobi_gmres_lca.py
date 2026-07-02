from pathlib import Path

import numpy as np

from bw2calc import JacobiGMRESLCA, LCA

fixture_dir = Path(__file__).resolve().parent / "fixtures"


def test_jacobi_gmres_matches_direct():
    packages = [fixture_dir / "basic_fixture.zip"]

    reference = LCA({1: 1}, data_objs=packages)
    reference.lci()

    candidate = JacobiGMRESLCA({1: 1}, data_objs=packages, rtol=1e-12, maxiter=100)
    candidate.lci()

    assert np.allclose(reference.supply_array, candidate.supply_array)
    assert np.allclose(reference.inventory.toarray(), candidate.inventory.toarray())


def test_jacobi_gmres_stores_guess():
    packages = [fixture_dir / "basic_fixture.zip"]
    candidate = JacobiGMRESLCA({1: 1}, data_objs=packages)
    candidate.lci()
    assert candidate.guess is not None


def test_jacobi_gmres_keeps_monte_carlo_technosphere_current():
    packages = [fixture_dir / "mc_basic.zip"]
    candidate = JacobiGMRESLCA(
        {3: 1},
        data_objs=packages,
        seed_override=42,
        use_distributions=True,
    )
    candidate.lci()

    initial_sum = candidate.technosphere_matrix.sum()
    next(candidate)

    assert candidate.technosphere_matrix is candidate.technosphere_mm.matrix
    assert candidate.technosphere_matrix.sum() != initial_sum
    assert candidate.technosphere_matrix.sum() == candidate.technosphere_mm.matrix.sum()


def test_jacobi_gmres_falls_back_to_direct_solver(monkeypatch):
    packages = [fixture_dir / "basic_fixture.zip"]

    def fake_gmres(*args, **kwargs):
        matrix, demand = args[:2]
        return np.zeros(matrix.shape[1], dtype=demand.dtype), 1

    monkeypatch.setattr("bw2calc.jacobi_gmres_lca.gmres", fake_gmres)

    reference = LCA({1: 1}, data_objs=packages)
    reference.lci()

    candidate = JacobiGMRESLCA({1: 1}, data_objs=packages)
    candidate.lci()

    assert np.allclose(reference.supply_array, candidate.supply_array)
