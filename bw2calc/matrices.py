# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from eight import *

from .fallbacks import dicter
from .utils import load_arrays, MAX_INT_32, TYPE_DICTIONARY
from scipy import sparse
import numpy as np
try:
    from bw2speedups import indexer
except ImportError:
    from .fallbacks import indexer


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
    def add_matrix_indices(cls, array_from, array_to, mapping):
        """
Map ``array_from`` keys to ``array_to`` values using the dictionary ``mapping``.

This is needed to take the ``flow``, ``input``, and ``output`` columns, which can be arbitrarily large integers, and transform them to matrix indices, which start from zero.

Here is an example:

.. code-block:: python

    import numpy as np
    a_f = np.array((1, 2, 3, 4))
    a_t = np.zeros(4)
    mapping = {1: 5, 2: 6, 3: 7, 4: 8}
    MatrixBuilder.add_matrix_indices(a_f, a_t, mapping)
    # => a_t is now [5, 6, 7, 8]

This is a relatively computationally expensive method, and therefore a stand-alone library - `bw2speedups <https://pypi.python.org/pypi/bw2speedups>`_ - is used if installed. Install it with ``pip install bw2speedups``.

Args:
    * *array_from* (array): 1-dimensional integer numpy array.
    * *array_to* (array): 1-dimensional integer numpy array.
    * *mapping* (dict): Dictionary that links ``mapping`` indices to ``row`` or ``col`` indices, e.g. ``{34: 3}``.

Operates in place. Doesn't return anything.

        """
        indexer(array_from, array_to, mapping)

    @classmethod
    def build_dictionary(cls, array):
        """
Build a dictionary from the sorted, unique elements of an array.

Here is an example:

.. code-block:: python

    import numpy as np
    array = np.array((4, 8, 6, 2, 4))
    MatrixBuilder.build_dictionary(array)
    # => returns {2: 0, 4: 1, 6: 2, 8: 3}

Args:
    * *array* (array): A numpy array of integers

Returns:
    A dictionary that maps the sorted, unique elements of ``array`` to integers starting with zero.

        """
        return dicter(array)

    @classmethod
    def build(cls, paths, data_label,
              row_id_label, row_index_label,
              col_id_label=None, col_index_label=None,
              row_dict=None, col_dict=None, one_d=False, drop_missing=True):
        """
Build a sparse matrix from NumPy structured array(s).

See more detailed documentation at :ref:`building-matrices`.

This method does the following:

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
            row_dict = cls.build_dictionary(array[row_id_label])
        cls.add_matrix_indices(array[row_id_label], array[row_index_label],
                               row_dict)
        if one_d:
            # Eliminate references to row data which isn't used;
            # Unused data remains MAX_INT_32 values because it isn't mapped
            # by ``add_matrix_indices``.
            if drop_missing:
                array = array[np.where(array[row_index_label] != MAX_INT_32)]
            matrix = cls.build_diagonal_matrix(array, row_dict, row_index_label, data_label)
        else:
            if not col_dict:
                col_dict = cls.build_dictionary(array[col_id_label])
            cls.add_matrix_indices(array[col_id_label],
                                   array[col_index_label], col_dict)
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
        tech_array = array.take(
            np.hstack((
                np.where(array['type'] == TYPE_DICTIONARY["production"])[0],
                np.where(array['type'] == TYPE_DICTIONARY.get("substitution", -999))[0],
                np.where(array['type'] == TYPE_DICTIONARY["technosphere"])[0],
            ))
        )
        bio_array = array.take(np.where(
            array['type'] == TYPE_DICTIONARY["biosphere"]
        )[0])
        activity_dict = cls.build_dictionary(np.hstack((
            tech_array['output'],
            bio_array['output']
        )))
        product_dict = cls.build_dictionary(tech_array["input"])
        bio_dict = cls.build_dictionary(bio_array["input"])
        cls.add_matrix_indices(tech_array['input'], tech_array['row'],
                               product_dict)
        cls.add_matrix_indices(tech_array['output'], tech_array['col'],
                               activity_dict)
        cls.add_matrix_indices(bio_array['input'], bio_array['row'], bio_dict)
        cls.add_matrix_indices(bio_array['output'], bio_array['col'],
                               activity_dict)
        technosphere = cls.build_technosphere_matrix(tech_array, activity_dict, product_dict)
        biosphere = cls.build_matrix(bio_array, bio_dict, activity_dict, "row", "col", "amount")
        return (bio_array, tech_array, bio_dict, activity_dict, product_dict,
                biosphere, technosphere)

    @classmethod
    def get_technosphere_inputs_mask(cls, array):
        """Get boolean mask of technosphere inputs from ``array`` (i.e. the ones to include when building the technosphere matrix)."""
        return np.where(
            array["type"] == TYPE_DICTIONARY["technosphere"]
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
