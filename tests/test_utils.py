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
    matrix = sps.csr_matrix(dense)
    lu_solver(matrix)

    # A CSC view of the CSR arrays: no conversion or copy before factorizing.
    assert factorized[0].format == "csc"
    assert np.array_equal(factorized[0].toarray(), dense.T)
    assert np.shares_memory(factorized[0].data, matrix.data)
    assert np.shares_memory(factorized[0].indices, matrix.indices)


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


@needs_umfpack
def test_lu_solver_uses_64_bit_umfpack_family(monkeypatch):
    """Never "di", which runs out of memory on large matrices; see scipy/scipy#8278."""
    import scikits.umfpack as umfpack

    monkeypatch.setattr(utils, "UMFPACK", True)
    families = []
    original = umfpack.UmfpackContext
    monkeypatch.setattr(
        umfpack, "UmfpackContext", lambda family: families.append(family) or original(family)
    )
    matrix = sps.csr_matrix(lu_test_matrix())
    assert matrix.indices.dtype == np.int32
    b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(lu_solver(matrix)(b), np.linalg.solve(lu_test_matrix(), b))
    assert families == ["dl"]


@pytest.mark.parametrize("umfpack", [False, pytest.param(True, marks=needs_umfpack)])
def test_lu_solver_leaves_a_non_canonical_matrix_alone(monkeypatch, umfpack):
    """`splu` sorts indices in place; `IterativeLCA` compares its reference copy to the
    technosphere matrix array by array, so the input must not be reordered."""
    monkeypatch.setattr(utils, "UMFPACK", umfpack)
    dense = lu_test_matrix()
    # `dense` in CSR with the column indices of each row in reverse order.
    matrix = sps.csr_matrix(
        (
            np.array([-0.5, 2.0, -0.25, 1.0, 4.0, -0.1]),
            np.array([1, 0, 2, 1, 2, 0]),
            np.array([0, 2, 4, 6]),
        ),
        shape=(3, 3),
    )
    assert np.array_equal(matrix.toarray(), dense)
    assert not matrix.has_canonical_format
    indices, data = matrix.indices.copy(), matrix.data.copy()

    b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(lu_solver(matrix)(b), np.linalg.solve(dense, b))
    assert np.array_equal(matrix.indices, indices)
    assert np.array_equal(matrix.data, data)


@pytest.mark.parametrize("umfpack", [False, pytest.param(True, marks=needs_umfpack)])
def test_lu_solver_accepts_float32(monkeypatch, umfpack):
    monkeypatch.setattr(utils, "UMFPACK", umfpack)
    dense = lu_test_matrix()
    b = np.array([1.0, 2.0, 3.0])
    x = lu_solver(sps.csr_matrix(dense.astype(np.float32)))(b)
    assert x.dtype == np.float64
    assert np.allclose(x, np.linalg.solve(dense, b), rtol=1e-5)
