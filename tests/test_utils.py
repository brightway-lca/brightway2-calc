import multiprocessing
from pathlib import Path

import bw_processing as bwp
import numpy as np
import pytest
import scipy.sparse as sps
from bw_processing.io_helpers import generic_directory_filesystem
from fsspec.implementations.zip import ZipFileSystem as ZFS
from scipy.sparse.linalg import splu

from bw2calc import utils
from bw2calc.utils import get_datapackage, get_seed, lu_solver

try:
    import scikits.umfpack  # noqa: F401

    HAS_UMFPACK = True
except ImportError:
    HAS_UMFPACK = False

needs_umfpack = pytest.mark.skipif(not HAS_UMFPACK, reason="scikit-umfpack not installed")

fixture_dir = Path(__file__).resolve().parent / "fixtures"


def test_get_seeds_different_under_mp_pool():
    with multiprocessing.Pool(processes=4) as pool:
        results = list(pool.map(get_seed, [None] * 10))
    assert sorted(set(results)) == sorted(results)


def test_consistent_global_index():
    # TODO
    pass


def test_get_datapackage():
    dp = bwp.load_datapackage(ZFS(fixture_dir / "basic_fixture.zip"))
    assert get_datapackage(dp) is dp

    assert get_datapackage(ZFS(fixture_dir / "basic_fixture.zip")).metadata == dp.metadata

    assert get_datapackage(fixture_dir / "basic_fixture.zip").metadata == dp.metadata

    assert get_datapackage(str(fixture_dir / "basic_fixture.zip")).metadata == dp.metadata

    dp = bwp.load_datapackage(generic_directory_filesystem(dirpath=fixture_dir / "basic_fixture"))
    assert get_datapackage(dp) is dp

    assert (
        get_datapackage(
            generic_directory_filesystem(dirpath=fixture_dir / "basic_fixture")
        ).metadata
        == dp.metadata
    )

    assert get_datapackage(str(fixture_dir / "basic_fixture")).metadata == dp.metadata

    with pytest.raises(TypeError):
        get_datapackage(1)


def lu_test_matrix():
    # Not symmetric, so solving with `A^T` by mistake would give a different answer.
    return np.array([[2.0, -0.5, 0.0], [0.0, 1.0, -0.25], [-0.1, 0.0, 4.0]])


@pytest.mark.parametrize("umfpack", [False, pytest.param(True, marks=needs_umfpack)])
@pytest.mark.parametrize("fmt", [sps.csr_matrix, sps.csc_matrix, sps.csr_array, sps.coo_matrix])
def test_lu_solver_solves_matrix_not_its_transpose(monkeypatch, umfpack, fmt):
    monkeypatch.setattr(utils, "UMFPACK", umfpack)
    dense = lu_test_matrix()
    b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(lu_solver(fmt(dense))(b), np.linalg.solve(dense, b))


def test_lu_solver_factorizes_the_transpose(monkeypatch):
    monkeypatch.setattr(utils, "UMFPACK", False)
    factorized = []
    monkeypatch.setattr(utils, "splu", lambda m: factorized.append(m) or splu(m))
    dense = lu_test_matrix()
    lu_solver(sps.csr_matrix(dense))

    # A CSC view of the CSR arrays: no conversion needed before factorizing.
    assert factorized[0].format == "csc"
    assert np.array_equal(factorized[0].toarray(), dense.T)


@pytest.mark.parametrize("umfpack", [False, pytest.param(True, marks=needs_umfpack)])
def test_lu_solver_ignores_later_in_place_edits(monkeypatch, umfpack):
    """Monte Carlo and `IterativeLCA` edit `technosphere_matrix` in place."""
    monkeypatch.setattr(utils, "UMFPACK", umfpack)
    dense = lu_test_matrix()
    matrix = sps.csr_matrix(dense)
    solve = lu_solver(matrix)
    matrix.data *= 3
    b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(solve(b), np.linalg.solve(dense, b))
