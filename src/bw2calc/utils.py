import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import bw_processing as bwp
import numpy as np
from bw_processing.io_helpers import generic_directory_filesystem
from fsspec import AbstractFileSystem
from fsspec.implementations.zip import ZipFileSystem
from scipy import sparse
from scipy.sparse.linalg import splu

from bw2calc import UMFPACK
from bw2calc.errors import InconsistentGlobalIndex, InvalidReferenceFlags, UnknownIndicesArray


def get_seed(seed=None):
    """Get valid Numpy random seed value"""
    # https://groups.google.com/forum/#!topic/briansupport/9ErDidIBBFM
    random = np.random.RandomState(seed)
    return random.randint(0, 2147483647)


def consistent_global_index(packages, matrix="characterization_matrix"):
    """The ``global_index`` of all resource groups of ``matrix``, or ``None``; raises
    ``InconsistentGlobalIndex`` if they have different values. No longer used by bw2calc,
    which filters each group with its own value (``GlobalIndexFilter``)."""
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


class GlobalIndexFilter:
    """A ``custom_filter`` for ``matrix_utils`` which keeps, in each resource group of
    ``matrix``, only the rows whose column is that group's own ``global_index`` (the global
    location of the project which processed it). Groups without a ``global_index`` are not
    filtered.

    Each group is filtered with its own value, so packages processed in projects whose global
    location has different ids can be used together; ``consistent_global_index`` required one
    value for all of them.

    ``matrix_utils`` calls a filter with a group's indices array only, so the filter finds the
    group by that array, and must have seen the group's package first. Give the packages to
    ``matrix_utils`` through ``through``, which registers each one as it passes::

        fltr = GlobalIndexFilter()
        MappedMatrix(packages=fltr.through(packages), custom_filter=fltr, ...)

    The packages are then iterated only once, so ``packages`` can be a generator.

    Finding the group by its array works because ``filter_by_attribute`` makes no copies, and
    because the arrays are loaded here, so that a proxy isn't loaded again (as another object)
    by a filtered package. An array which was not registered raises ``UnknownIndicesArray``,
    instead of being used unfiltered, which would add regionalized factors to the global ones.
    This happens to a group whose indices ``matrix_utils`` copies first (with ``transpose``,
    which characterization matrices don't use)."""

    def __init__(self, matrix: str = "characterization_matrix"):
        self.matrix = matrix
        # ``id`` of indices array: ``global_index`` of its group, or ``None``
        self.by_array: dict[int, int | None] = {}
        # Kept alive, so that their ids can't be given to other arrays
        self.arrays: list[np.ndarray] = []

    def register(self, package: bwp.DatapackageBase) -> None:
        """Load the indices arrays of ``matrix`` in ``package``, and store the ``global_index``
        of each. ``package`` must be the object given to ``matrix_utils``, or one that it was
        filtered from."""
        for resource in (
            package.filter_by_attribute("matrix", self.matrix)
            .filter_by_attribute("kind", "indices")
            .resources
        ):
            array, _ = package.get_resource(resource["name"])
            if id(array) not in self.by_array:
                self.arrays.append(array)
            self.by_array[id(array)] = resource.get("global_index")

    def through(self, packages: Iterable[bwp.DatapackageBase]) -> Iterator[bwp.DatapackageBase]:
        """Yield ``packages`` unchanged, after registering each one."""
        for package in packages:
            self.register(package)
            yield package

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        try:
            index = self.by_array[id(indices)]
        except KeyError:
            raise UnknownIndicesArray(
                f"Indices array for `{self.matrix}` ({len(indices)} rows) isn't an array of the "
                "registered datapackages, so its `global_index` is unknown, and it can't be "
                "filtered to the global location. Give the datapackages to `matrix_utils` "
                "through `GlobalIndexFilter.through`; indices which `matrix_utils` copies "
                "(with `transpose`) are not supported."
            ) from None
        if index is None:
            return np.ones(len(indices), dtype=bool)
        return indices["col"] == index


def global_index_filter(packages, matrix="characterization_matrix") -> GlobalIndexFilter:
    """A ``GlobalIndexFilter`` for ``matrix`` with all ``packages`` already registered."""
    fltr = GlobalIndexFilter(matrix)
    for package in packages:
        fltr.register(package)
    return fltr


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


def lu_solver(
    matrix: Any, copy: bool = True, refine: bool = True
) -> Callable[[np.ndarray], np.ndarray]:
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
    first.

    :param copy: UMFPACK reads the matrix again in each solve for iterative refinement, so
        by default it gets a copy of ``matrix``. Pass ``False`` for a matrix which is never
        changed after this call, to skip that copy. No copy is needed, or made, if ``refine``
        is ``False``, or with SuperLU.
    :param refine: UMFPACK's iterative refinement (two steps by default) is most of each
        solve. ``False`` skips it, which is about 3x faster with reference BLAS or OpenBLAS
        but can lose accuracy on badly scaled matrices; see
        https://github.com/brightway-lca/brightway2-calc/issues/184. Ignored by SuperLU."""
    # CSC, and a view on `matrix`'s arrays if it is CSR.
    transposed = matrix.tocsr().T
    if transposed.dtype != np.float64 or not transposed.has_canonical_format:
        # Both solvers need float64 values and sorted indices without duplicates, and
        # `splu` sorts in place, which through the view would reorder the arrays of
        # `matrix` too. `IterativeLCA` compares its reference copy to
        # `technosphere_matrix` array by array, so a reordered copy would never match.
        # `LCA` builds canonical float64 matrices, so this copy is for matrices from
        # elsewhere.
        transposed = transposed.astype(np.float64, copy=True)
        transposed.sum_duplicates()
        transposed.sort_indices()

    if not UMFPACK:
        # `splu` copies the values into its factors, and `solve` doesn't read the matrix
        # again, so changes to `matrix` afterwards don't matter.
        factorization = splu(transposed)
        return lambda b: factorization.solve(b, trans="T")

    import scikits.umfpack as umfpack  # noqa: PLC0415

    # UMFPACK reads the matrix again in each solve, for iterative refinement, so it needs
    # a copy unless the caller promises not to change `matrix`: Monte Carlo and
    # `IterativeLCA` edit `technosphere_matrix` in place. The conversions above, or
    # `tocsr()` for a non-CSR input, may already have made one. Without refinement, `solve`
    # only reads the factors, so there is nothing to protect.
    if copy and refine and np.may_share_memory(transposed.data, matrix.data):
        transposed = transposed.copy()
    # Always the 64-bit index family "dl", as `scipy` does, never "di" for int32 indices:
    # the 32-bit variant sizes its workspace in int32 and ran out of memory on a
    # 13k x 13k matrix; see scipy/scipy#8278. Convert the indices by assignment: the
    # sparse constructors downcast int64 indices to int32 when they fit.
    transposed.indices = transposed.indices.astype(np.int64)
    transposed.indptr = transposed.indptr.astype(np.int64)
    context = umfpack.UmfpackContext("dl")
    if not refine:
        context.control[umfpack.UMFPACK_IRSTEP] = 0
    context.numeric(transposed)

    def solve(b: np.ndarray) -> np.ndarray:
        # `UMFPACK_At` solves with the transpose of the factorized `A^T`, i.e. `A`.
        # Ignoring warnings as `scipy` does; see scipy/scipy#16523.
        with np.errstate(divide="ignore", invalid="ignore"):
            return context.solve(umfpack.UMFPACK_At, transposed, b, autoTranspose=False)

    return solve


def pardiso_refinement(solver: Any, refine: bool) -> None:
    """Turn PARDISO's iterative refinement on or off for later solves with ``solver``, a
    ``pypardiso.PyPardisoSolver`` which has already factorized a matrix.

    pypardiso leaves ``iparm`` at 0, so MKL uses its defaults, up to two refinement steps
    among them, and writes them back into ``iparm`` on the first call. ``refine=False`` keeps
    those values (``iparm[0] = 1``) but sets the maximum number of steps, ``iparm[7]``, to 0,
    which skips refinement even when pivots were perturbed (``iparm[13]``).
    ``refine=True`` undoes that: with ``iparm[0] = 0``, MKL fills in its defaults again."""
    if refine:
        solver.iparm[0] = 0
    else:
        solver.iparm[0] = 1
        solver.iparm[7] = 0


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
