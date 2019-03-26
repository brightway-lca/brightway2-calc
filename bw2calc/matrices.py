# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from eight import *

from .indexing import index_with_searchsorted, index_with_arrays
from .utils import load_arrays, MAX_INT_32, TYPE_DICTIONARY
from scipy import sparse
import numpy as np


class MatrixBuilder(object):
    """
The class, and its subclasses, load structured arrays, manipulate them, and generate `SciPy sparse matrices <http://docs.scipy.org/doc/scipy/reference/sparse.html>`_.

Matrix builders use an array of row indices, an array of column indices, and an array of values to create a `coordinate (coo) matrix <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html>`_, which is then converted to a `compressed sparse row (csr) matrix <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_.

See the following for more information on structured arrays:

* `NumPy structured arrays <http://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html#numpy.recarray>`_
* `Intermediate and processed data <https://docs.brightwaylca.org/intro.html#intermediate-and-processed-data>`_

These classes are not instantiated, and have only `classmethods <https://docs.python.org/2/library/functions.html#classmethod>`__. They are not really true classes, but more organizational. In other words, you should use:

.. code-block:: python

    MatrixBuilder.build(args)

and not:

.. code-block:: python

    mb = MatrixBuilder()
    mb.build(args)

    """

    @classmethod
    def build(cls, paths, data_label,
              row_id_label, row_index_label,
              col_id_label=None, col_index_label=None,
              row_dict=None, col_dict=None, one_d=False, drop_missing=True):
        """
Build a sparse matrix from NumPy structured array(s).

See more detailed documentation at :ref:`building-matrices`.

This method does the following:

TODO: Update

#. Load and concatenate the :ref:`structured arrays files <building-matrices>` in filepaths ``paths`` using the function :func:`.utils.load_arrays` into a parameter array.
#. If not ``row_dict``, use :meth:`.build_dictionary` to build ``row_dict`` from the parameter array column ``row_id_label``.
#. Using the ``row_id_label`` and the ``row_dict``, use the method :meth:`.add_matrix_indices` to add matrix indices to the ``row_index_label`` column.
#. If not ``one_d``, do the same to ``col_dict`` and ``col_index_label``, using ``col_id_label``.
#. If not ``one_d``, use :meth:`.build_matrix` to build a sparse matrix using ``data_label`` for the matrix data values, and ``row_index_label`` and ``col_index_label`` for row and column indices.
#. Else if ``one_d``, use :meth:`.build_diagonal_matrix` to build a diagonal matrix using ``data_label`` for diagonal matrix data values and ``row_index_label`` as row/column indices.
#. Return the loaded parameter arrays from step 1, row and column dicts from steps 2 & 4, and matrix from step 5 or 6.

Args:
    * *paths* (list): List of array filepaths to load.
    * *data_label* (str): Label of column in parameter arrays with matrix data values.
    * *row_id_label* (str): Label of column in parameter arrays with row ID values, i.e. the integer values returned from ``mapping``.
    * *row_index_label* (str): Label of column in parameter arrays where matrix row indices will be stored.
    * *col_id_label* (str, optional): Label of column in parameter arrays with column ID values, i.e. the integer values returned from ``mapping``. Not needed for diagonal matrices.
    * *col_index_label* (str, optional): Label of column in parameter arrays where matrix column indices will be stored. Not needed for diagonal matrices.
    * *row_dict* (dict, optional): Mapping dictionary linking ``row_id_label`` values to ``row_index_label`` values. Will be built if not given.
    * *col_dict* (dict, optional): Mapping dictionary linking ``col_id_label`` values to ``col_index_label`` values. Will be built if not given.
    * *one_d* (bool): Build diagonal matrix.
    * *drop_missing* (bool): Remove rows from the parameter array which aren't mapped by ``row_dict`` or ``col_dict``. Default is ``True``. Advanced use only.

Returns:
    A :ref:`numpy parameter array <building-matrices>`, the row mapping dictionary, the column mapping dictionary, and a COO sparse matrix.

        """
        assert isinstance(paths, (tuple, list, set)), "``paths`` must be a list"
        array = load_arrays(paths)

        if not row_dict:
            row_dict = index_with_searchsorted(
                array[row_id_label],
                array[row_index_label]
            )
        else:
            index_with_arrays(
                array[row_id_label],
                array[row_index_label],
                row_dict
            )

        if one_d:
            # Eliminate references to row data which isn't used;
            # Unused data remains MAX_INT_32 values
            if drop_missing:
                array = array[np.where(array[row_index_label] != MAX_INT_32)]
            matrix = cls.build_diagonal_matrix(array, row_dict, row_index_label, data_label)
        else:
            if not col_dict:
                col_dict = index_with_searchsorted(
                    array[col_id_label],
                    array[col_index_label]
                )
            else:
                index_with_arrays(
                    array[col_id_label],
                    array[col_index_label],
                    col_dict
                )

            if drop_missing:
                array = array[np.where(array[row_index_label] != MAX_INT_32)]
                array = array[np.where(array[col_index_label] != MAX_INT_32)]
            matrix = cls.build_matrix(
                array, row_dict, col_dict, row_index_label, col_index_label,
                data_label)
        return array, row_dict, col_dict, matrix

    @classmethod
    def build_matrix(cls, array, row_dict, col_dict, row_index_label,
                     col_index_label, data_label=None, new_data=None):
        """Build sparse matrix."""
        vector = array[data_label] if new_data is None else new_data
        assert vector.shape[0] == array.shape[0], "Incompatible data & indices"
        # coo_matrix construction is coo_matrix((values, (rows, cols)),
        # (row_count, col_count))
        return sparse.coo_matrix((
            vector.astype(np.float64),
            (array[row_index_label], array[col_index_label])),
            (len(row_dict), len(col_dict))).tocsr()

    @classmethod
    def build_diagonal_matrix(cls, array, row_dict, index_label,
                              data_label=None, new_data=None):
        """Build diagonal sparse matrix."""
        return cls.build_matrix(array, row_dict, row_dict, index_label, index_label, data_label, new_data)


class TechnosphereBiosphereMatrixBuilder(MatrixBuilder):
    """Subclass of ``MatrixBuilder`` that separates technosphere and biosphere parameters"""
    @classmethod
    def build(cls, paths):
        """Build the technosphere and biosphere sparse matrices."""
        assert isinstance(paths, (tuple, list, set)), "paths must be a list"
        array = load_arrays(paths)
        # take ~10 times faster than fancy indexing
        # http://wesmckinney.com/blog/?p=215
        tech_array = cls.select_technosphere_array(array)
        bio_array = cls.select_biosphere_array(array)

        activity_dict = index_with_searchsorted(
            tech_array['output'],
            tech_array['col']
        )
        product_dict = index_with_searchsorted(
            tech_array['input'],
            tech_array['row']
        )
        bio_dict = index_with_searchsorted(
            bio_array['input'],
            bio_array['row']
        )
        index_with_arrays(
            bio_array['output'],
            bio_array['col'],
            activity_dict
        )

        technosphere = cls.build_technosphere_matrix(tech_array, activity_dict, product_dict)
        biosphere = cls.build_matrix(bio_array, bio_dict, activity_dict, "row", "col", "amount")
        return (bio_array, tech_array, bio_dict, activity_dict, product_dict,
                biosphere, technosphere)

    @classmethod
    def select_technosphere_array(cls, array):
        """Create a new array with technosphere matrix exchanges"""
        return array.take(
            np.hstack((
                np.where(array['type'] == TYPE_DICTIONARY["production"])[0],
                np.where(array['type'] == TYPE_DICTIONARY.get("substitution", -999))[0],
                np.where(array['type'] == TYPE_DICTIONARY["technosphere"])[0],
            ))
        )

    @classmethod
    def select_biosphere_array(cls, array):
        """Create a new array with biosphere matrix exchanges"""
        return array.take(np.where(
            array['type'] == TYPE_DICTIONARY["biosphere"]
        )[0])

    @classmethod
    def get_technosphere_inputs_mask(cls, array):
        """Get boolean mask of technosphere inputs from ``array`` (i.e. the ones to include when building the technosphere matrix)."""
        return np.where(
            array["type"] == TYPE_DICTIONARY["technosphere"]
        )

    @classmethod
    def get_biosphere_inputs_mask(cls, array):
        """Get boolean mask of biosphere flows from ``array`` (i.e. the ones to include when building the biosphere matrix)."""
        return np.where(
            array["type"] == TYPE_DICTIONARY["biosphere"]
        )

    @classmethod
    def fix_supply_use(cls, array, vector):
        """Make technosphere inputs negative."""
        # Inputs are consumed, so are negative
        mask = cls.get_technosphere_inputs_mask(array)
        vector[mask] = -1 * vector[mask]
        return vector

    @classmethod
    def build_technosphere_matrix(cls, array, activity_dict, product_dict, new_data=None):
        vector = array["amount"] if new_data is None else new_data
        vector = cls.fix_supply_use(array, vector.copy())
        return cls.build_matrix(array, product_dict, activity_dict, "row",
                                "col", "amount", vector)


class SingleMatrixBuilder(MatrixBuilder):
    """Subclass of ``MatrixBuilder`` that supports consumption (i.e. multiply by -1)."""
    @classmethod
    def build(cls, path):
        """Build the technosphere and biosphere sparse matrices."""
        array = load_arrays([path])
        col_dict = index_with_searchsorted(
            array['output'],
            array['col']
        )
        row_dict = index_with_searchsorted(
            array['input'],
            array['row']
        )
        matrix = cls.build_single_matrix(array, row_dict, col_dict)
        return (array, row_dict, col_dict, matrix)

    @classmethod
    def fix_supply_use(cls, array, vector):
        """Make technosphere inputs negative."""
        # Inputs are consumed, so are negative
        mask = np.where(
            array["type"] == TYPE_DICTIONARY["generic consumption"]
        )
        vector[mask] = -1 * vector[mask]
        return vector

    @classmethod
    def build_single_matrix(cls, array, row_dict, col_dict, new_data=None):
        vector = array["amount"] if new_data is None else new_data
        vector = cls.fix_supply_use(array, vector.copy())
        return cls.build_matrix(array, row_dict, col_dict, "row",
                                "col", "amount", vector)
