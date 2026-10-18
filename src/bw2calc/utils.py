import datetime
from pathlib import Path
from typing import Any, Callable

import bw_processing as bwp
import numpy as np
from bw_processing.io_helpers import generic_directory_filesystem
from fsspec import AbstractFileSystem
from fsspec.implementations.zip import ZipFileSystem
from scipy import sparse
from scipy.sparse.linalg import splu

from bw2calc import UMFPACK
from bw2calc.errors import InconsistentGlobalIndex, InvalidReferenceFlags


def get_seed(seed=None):
    """Get valid Numpy random seed value"""
    # https://groups.google.com/forum/#!topic/briansupport/9ErDidIBBFM
    random = np.random.RandomState(seed)
    return random.randint(0, 2147483647)


def consistent_global_index(packages, matrix="characterization_matrix"):
    global_list = [
        resource.get("global_index")
        for package in packages
        for resource in package.filter_by_attribute("matrix", matrix)
        .filter_by_attribute("kind", "indices")
        .resources
    ]
    if len(set(global_list)) > 1:
        raise InconsistentGlobalIndex(
            f"Multiple global index values found: {global_list}. If multiple LCIA datapackages"
            + " are present, they must use the same value for ``GLO``, the global location, in "
            + " order for filtering for site-generic LCIA to work correctly."
        )
    return global_list[0] if global_list else None


def convert_tuple_to_list(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def wrap_functional_unit(dct):
    """Transform functional units for effective logging.
    Turns ``Activity`` objects into their keys."""
    data = []
    for key, amount in dct.items():
        if isinstance(key, int):
            data.append({"id": key, "amount": amount})
        else:
            try:
                data.append({"database": key[0], "code": key[1], "amount": amount})
            except TypeError:
                data.append({"key": key, "amount": amount})
    return data


def get_datapackage(obj):
    if isinstance(obj, bwp.DatapackageBase):
        return obj
    elif isinstance(obj, AbstractFileSystem):
        return bwp.load_datapackage(obj)
    elif isinstance(obj, Path) and obj.suffix.lower() == ".zip":
        return bwp.load_datapackage(ZipFileSystem(obj))
    elif isinstance(obj, Path) and obj.is_dir():
        return bwp.load_datapackage(generic_directory_filesystem(dirpath=obj))
    elif isinstance(obj, str) and obj.lower().endswith(".zip") and Path(obj).is_file():
        return bwp.load_datapackage(ZipFileSystem(Path(obj)))
    elif isinstance(obj, str) and Path(obj).is_dir():
        return bwp.load_datapackage(generic_directory_filesystem(dirpath=Path(obj)))

    else:
        raise TypeError("Unknown input type for loading datapackage: {}: {}".format(type(obj), obj))


def utc_now() -> datetime.datetime:
    """Get current datetime compatible with Py 3.8 to 3.12"""
    if hasattr(datetime, "UTC"):
        return datetime.datetime.now(datetime.UTC)
    else:
        return datetime.datetime.utcnow()


def as_columns(array: Any) -> np.ndarray:
    """Convert ``array`` to a dense 2-d array, treating 1-d input as a single column.

    Accepts lists, NumPy arrays, and SciPy sparse matrices."""
    if sparse.issparse(array):
        array = array.toarray()
    array = np.asarray(array, dtype=float)
    if array.ndim > 2:
        raise ValueError(f"Expected 1-d or 2-d array, got {array.ndim} dimensions")
    return array.reshape(array.shape[0], -1) if array.ndim else array.reshape(1, 1)


def relative_residual(matrix: Any, solution: Any, rhs: Any) -> np.ndarray:
    """Relative residual ``||matrix @ solution - rhs|| / ||rhs||`` for each column of ``rhs``.

    Direct solvers don't always fail loudly on a singular or badly conditioned matrix; they can
    return numbers which look fine but don't solve the system. The residual costs one sparse
    matrix multiplication, so it is cheap insurance against silently wrong results.

    Columns of ``rhs`` which are all zero use the absolute residual instead, as there is nothing
    to be relative to.

    Returns a 1-d array with one value per column of ``rhs``."""
    rhs = as_columns(rhs)
    solution = as_columns(solution)
    residual = np.linalg.norm(matrix @ solution - rhs, axis=0)
    scale = np.linalg.norm(rhs, axis=0)
    return np.divide(residual, scale, out=residual.copy(), where=scale > 0)


def lu_solver(matrix: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Factorize the square sparse ``matrix`` ``A`` with UMFPACK, if installed, or SuperLU,
    and return a function which solves ``A x = b`` for a 1-d ``b``. Not for PARDISO, which
    has its own factorization.

    This factorizes ``A^T`` and solves the transposed system ``(A^T)^T x = b``, instead of
    factorizing ``A``. Both are correct, but LU factorization picks a column ordering to
    limit fill-in, and technosphere matrices have far less fill transposed. On ecoinvent
    3.9 cutoff (21,255 x 21,255), single-threaded, factorizing ``A`` took 52 s with
    SuperLU and 3.2 s with UMFPACK, and ``A^T`` took 1.7 s and 0.3 s, with 8.6x fewer
    nonzeros in the SuperLU factors, so each solve is faster too; see
    https://github.com/brightway-lca/brightway2-calc/issues/176. ``scipy``'s ``spsolve``
    does the same with a CSR matrix, so this is also what ``LCA`` does without
    ``factorize=True``. Don't use ``scipy.sparse.linalg.factorized(A.tocsc())``, which
    factorizes ``A``.

    Matrix formats: SuperLU and UMFPACK both factorize CSC matrices, and UMFPACK fails on
    CSR; see https://github.com/brightway-lca/brightway2-calc/issues/132. The CSC form of
    ``A^T`` has the same arrays as the CSR form of ``A``, which is what ``LCA`` builds, so
    ``matrix.tocsr().T`` is CSC without any conversion. Other formats are converted to CSR
    first."""
    # CSC, and a view on `matrix`'s arrays if it is CSR.
    transposed = matrix.tocsr().T

    if not UMFPACK:
        # `splu` copies the values into its factors, and `solve` doesn't read the matrix
        # again, so changes to `matrix` afterwards don't matter.
        factorization = splu(transposed)
        return lambda b: factorization.solve(b, trans="T")

    import scikits.umfpack as umfpack  # noqa: PLC0415

    # UMFPACK reads the matrix again in each solve, for iterative refinement, so it needs
    # a copy: Monte Carlo and `IterativeLCA` edit `technosphere_matrix` in place.
    # UMFPACK needs float64 values.
    transposed = transposed.astype(np.float64, copy=True)
    # The UMFPACK family must match the index type: "di" for int32 and "dl" for int64.
    # Pick it from the indices, as `scipy` does; scipy sparse constructors downcast int64
    # indices to int32 when they fit, so converting the indices wouldn't stick.
    family = "di" if transposed.indices.dtype == np.int32 else "dl"
    context = umfpack.UmfpackContext(family)
    context.numeric(transposed)

    def solve(b: np.ndarray) -> np.ndarray:
        # `UMFPACK_At` solves with the transpose of the factorized `A^T`, i.e. `A`.
        # Ignoring warnings as `scipy` does; see scipy/scipy#16523.
        with np.errstate(divide="ignore", invalid="ignore"):
            return context.solve(umfpack.UMFPACK_At, transposed, b, autoTranspose=False)

    return solve


def reference_entries(mapped_matrix) -> tuple:
    """Return the matrix ``(rows, cols)`` of the entries flagged as reference (production)
    exchanges in the ``reference_array`` resources of ``mapped_matrix``'s groups.

    Uses ``row_masked`` and ``col_masked``, which are aligned with the flags, and not
    ``row_matrix`` and ``col_matrix``, which aren't once duplicates are aggregated. The
    same entry can appear more than once."""
    rows, cols = [], []
    for group in mapped_matrix.groups:
        if group.empty or not group.has_reference:
            continue
        flags = np.asarray(group.reference_current, dtype=bool)
        rows.append(group.row_masked[flags])
        cols.append(group.col_masked[flags])
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64)


def production_pivot_map(rows: np.ndarray, cols: np.ndarray, shape: tuple) -> np.ndarray:
    """Return the pivot map ``p``, where ``p[j]`` is the row of the reference product of
    column ``j``, from the reference exchange entries ``(rows, cols)``.

    Raises ``InvalidReferenceFlags`` unless ``p`` is a permutation: every column has
    reference entries in exactly one row, and no two columns share a row."""
    n_rows, n_cols = shape
    if n_rows != n_cols:
        raise InvalidReferenceFlags(f"Technosphere matrix is not square: {shape}")
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    # The flags are read from `technosphere_mm`, but `technosphere_matrix` can have been
    # replaced by a smaller matrix since.
    if len(rows) and (
        rows.min() < 0 or cols.min() < 0 or rows.max() >= n_rows or cols.max() >= n_cols
    ):
        raise InvalidReferenceFlags(f"Reference exchange entries don't fit a {shape} matrix")
    # Unique (col, row) pairs as integer keys; `np.unique(..., axis=1)` sorts void
    # records, which is much slower.
    pair_cols, pair_rows = np.divmod(np.unique(cols * n_rows + rows), n_rows)
    rows_per_col = np.bincount(pair_cols, minlength=n_cols)
    unique = rows_per_col[pair_cols] == 1
    pivot_rows = np.full(n_cols, -1, dtype=np.int64)
    pivot_rows[pair_cols[unique]] = pair_rows[unique]

    missing = int((rows_per_col == 0).sum())
    conflicting = int((rows_per_col > 1).sum())
    paired = pivot_rows[pivot_rows >= 0]
    shared = int((np.unique(paired, return_counts=True)[1] > 1).sum())
    problems = [
        message.format(count)
        for count, message in [
            (missing, "{:,} columns have no reference exchange"),
            (conflicting, "{:,} columns have reference exchanges in several rows"),
            (shared, "{:,} rows are the reference product of several columns"),
        ]
        if count
    ]
    if problems:
        raise InvalidReferenceFlags(
            "Reference exchange flags cover {:,} of {:,} columns: {}".format(
                len(paired), n_cols, "; ".join(problems)
            )
        )
    return pivot_rows
