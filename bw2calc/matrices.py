# -*- coding: utf-8 -*-
from .indexing import index_with_searchsorted, index_with_arrays
from bw_processing import MAX_SIGNED_32BIT_INT
from scipy import sparse
import numpy as np


def build_labelled_matrix(
    array, row_dict=None, col_dict=None, one_d=False, drop_missing=True,
):
    """
Build a `SciPy sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`__ from `NumPy structured array <https://numpy.org/doc/stable/user/basics.rec.html?highlight=record%20array>`__.

This function does the following:

* If there are no row and/or column dictionaries, construct these dictionaries from the columns ``row_value`` and ``col_value`` using the function ``bw2calc.indexing.index_with_searchsorted``. After this function is applied, the columns ``row_index`` and/or ``col_index`` are populated. Note that the values in ``row|col_index`` are overwritten, regardless of their current value (i.e. there is no check for ``bw_processing.MAX_SIGNED_32BIT_INT``).
* Otherwise, populate the columns ``row_index`` and/or ``col_index`` using ``row_dict`` and/or ``col_dict`` using the function ``bw2calc.indexing.index_with_arrays``.
* If ``drop_missing``, filter ``array`` for row or column values which were not present in the provided row or column dictionary. Only applies if such dictionaries are given as inputs. ``drop_missing`` is used, for example, in LCIA methods which cover many flows, only some of which are present in the given database.
* If ``one_d``, build a diagonal matrix using ``bw2calc.matrices.build_diagonal_matrix``. Otherwise, build a full matrix using ``bw2calc.matrices.build_matrix``.

Note that, while ``row|col_dict`` values are not strictly required to start from 0, this is highly encouraged, and is the default behaviour.

Args:
    array: numpy structured array with columns ``row_value``, ``row_index``, ``col_value``, ``col_index``, and ``amount``.
    row_dict: Dictionary mapping sorted unique integers in ``row_value`` to positive sequential integers.
    col_dict: Dictionary mapping sorted unique integers in ``col_value`` to positive sequential integers.
    one_d: Boolean on whether to only use ``row_value`` and ``row_index`` to build diagonal matrix.
    drop_missing: Boolean on whether to remove elements which are present is either ``row_dict`` or ``col_dict``. See above.

Returns:
    Modified ``array``, ``row_dict`` (constructed or given), ``col_dict`` (constructed or given; same as ``row_dict`` if ``one_d``), Scipy sparse matrix

Raises:
    ValueError: If ``drop_missing`` is ``False``, but there are missing elements (i.e. a value in ``row_value`` doesn't have a key in ``row_dict``). Either the missing elements will cause a mismatch between the number of array rows and the matrix dimensions, or the missing elements will still have the value ``MAX_SIGNED_32BIT_INT``, which will be a row or column index greater than the matrix dimensions.

    """
    if not row_dict:
        row_dict = index_with_searchsorted(array["row_value"], array["row_index"])
    else:
        index_with_arrays(array["row_value"], array["row_index"], row_dict)

    if one_d:
        # Eliminate references to row data which isn't used.
        # Unused data remains MAX_SIGNED_32BIT_INT values
        # Useful e.g. for characterization matrices where
        # not every flow in the LCIA method is present in
        # the biosphere.
        if drop_missing:
            array = array[np.where(array["row_index"] != MAX_SIGNED_32BIT_INT)]
        matrix = build_diagonal_matrix(array, max(row_dict.values()) + 1)
    else:
        if not col_dict:
            col_dict = index_with_searchsorted(array["col_value"], array["col_index"])
        else:
            index_with_arrays(array["col_value"], array["col_index"], col_dict)

        if drop_missing:
            array = array[np.where(array["row_index"] != MAX_SIGNED_32BIT_INT)]
            array = array[np.where(array["col_index"] != MAX_SIGNED_32BIT_INT)]
        nrows = max(row_dict.values()) + 1 if row_dict else 0
        ncols = max(col_dict.values()) + 1 if col_dict else 0
        matrix = build_matrix(array, nrows, ncols)
    return array, row_dict, col_dict, matrix


def build_matrix(
    array,
    nrows,
    ncols,
    row_index_label="row_index",
    col_index_label="col_index",
    new_data=None,
    flip=True,
):
    """Build sparse matrix from the indices and data in ``array``..

    Be careful with ``new_data`` - the data in this array will still have their signs flipped unless ``flip`` is false.

    Although row and column index labels can be changed, it is strongly suggested to use the default labels, as experience has shown that it is much simpler and less error-prone to always have a consistent common input format.

    Args:
        array: numpy structured array with columns ``row_index``, ``col_index``, and ``amount``.
        nrows: Integer number of rows in matrix.
        ncols: Integer number of cols in matrix.
        row_index_label (str): Label of the column in ``array`` which gives row indices.
        col_index_label (str): Label of the column in ``array`` which gives column indices.
        new_data: Numpy vector to be used instead of ``array['data']``.
        flip: Boolean on whether to flip the signs of the data using the column ``array['flip']``.

    Returns:
        A `scipy.sparse.csrmatrix` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>__.

    Raises:
        ValueError: If the number of rows or columns in ``array`` are large than ``nrows`` or ``ncols``.
        ValueError: If the shape of ``new_data`` (if provided) is different than ``array``.

    """
    vector = array["amount"] if new_data is None else new_data
    if not vector.shape[0] == array.shape[0]:
        raise ValueError(
            "Incompatible data & indices: {} / {}".format(vector.shape, array.shape)
        )
    if flip:
        vector = flip_amounts(array, vector)
    # coo_matrix construction is coo_matrix((values, (rows, cols)),
    # (row_count, col_count))
    return sparse.coo_matrix(
        (vector.astype(np.float64), (array[row_index_label], array[col_index_label]),),
        (nrows, ncols),
    ).tocsr()


def flip_amounts(params, vector=None):
    """Flip the sign of values where the column ``params['flip']`` is ``True``.

    Uses ``vector`` or ``array['amount']`` (modifying in place).

    Used primarily in the technosphere matrix for consumed inputs."""
    mask = params["flip"]
    if vector is None:
        vector = params["amount"]
    vector[mask] *= -1
    return vector


def build_diagonal_matrix(array, nrows, new_data=None):
    """Convenience wrapper for building diagonal sparse matrix."""
    return build_matrix(
        array, nrows, nrows, col_index_label="row_index", new_data=new_data
    )
