# -*- coding: utf-8 -*
from __future__ import division
from .fallbacks import dicter
from .utils import load_arrays
from bw2data.utils import MAX_INT_32, TYPE_DICTIONARY
from scipy import sparse
import numpy as np
try:
    from bw2speedups import indexer
except ImportError:
    from .fallbacks import indexer


class MatrixBuilder(object):
    """Load a structured array from a file, manipulate it. Generates a sparse matrix."""

    @classmethod
    def load(self, dirpath, names):
        """Load a structured array from a file."""
        return load_arrays(dirpath, names)

    @classmethod
    def add_matrix_indices(self, array_from, array_to, mapping):
        """Map ``array_from`` keys to ``array_to`` values using ``mapping``.

        Operates in place. Doesn't return anything."""
        indexer(array_from, array_to, mapping)

    @classmethod
    def build_dictionary(self, array):
        """Build a dictionary from the sorted, unique elements of an array"""
        return dicter(array)

    @classmethod
    def build(cls, dirpath, names, data_label,
              row_id_label, row_index_label,
              col_id_label=None, col_index_label=None,
              row_dict=None, col_dict=None, one_d=False):
        """Build a sparse matrix from NumPy structured array(s).

        This method does the following:

        #. Load and concatenate some structured arrays.
        #. Using the ``row_id_label``, and the ``row_dict`` if available, add matrix indices to the ``row_index_label`` column.
        #. If not ``ond_d``, do the same to ``col_index_label`` using ``col_id_label`` and ``col_dict``.
        #. If not ``ond_d``, build a sparse matrix using ``data_label`` for the matrix data, and ``row_index_label`` and ``col_index_label`` as indices.
        #. Else if ``ond_d``, build a diagonal matrix using only ``data_label`` for values and ``row_index_label`` as indices.
        #. Return the loaded parameter arrays, row and column dicts, and matrix."""
        assert isinstance(names, (tuple, list, set)), "names must be a list"
        array = load_arrays(dirpath, names)
        if not row_dict:
            row_dict = cls.build_dictionary(array[row_id_label])
        cls.add_matrix_indices(array[row_id_label], array[row_index_label],
                               row_dict)
        if one_d:
            # Eliminate references to row data which isn't used;
            # Unused data remains MAX_INT_32 values because it isn't mapped
            # by ``add_matrix_indices``.
            array = array[np.where(array[row_index_label] != MAX_INT_32)]
            matrix = cls.build_diagonal_matrix(array, row_dict, row_index_label, data_label)
        else:
            if not col_dict:
                col_dict = cls.build_dictionary(array[col_id_label])
            cls.add_matrix_indices(array[col_id_label],
                                   array[col_index_label], col_dict)
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
    """Subclass of ``MatrixBuilder`` that separates technosphere and biosphere parameters."""
    @classmethod
    def build(cls, dirpath, names):
        """Build the technosphere and biosphere sparse matrices."""
        assert isinstance(names, (tuple, list, set)), "names must be a list"
        array = load_arrays(dirpath, names)
        tech_array = array[
            np.hstack((
                np.where(array['type'] == TYPE_DICTIONARY["technosphere"])[0],
                np.where(array['type'] == TYPE_DICTIONARY["production"])[0]
            ))
        ]
        bio_array = array[np.where(array['type'] == TYPE_DICTIONARY["biosphere"])]
        tech_dict = cls.build_dictionary(np.hstack((
            tech_array['input'],
            tech_array['output'],
            bio_array['output']
        )))
        bio_dict = cls.build_dictionary(bio_array["input"])
        cls.add_matrix_indices(tech_array['input'], tech_array['row'],
                               tech_dict)
        cls.add_matrix_indices(tech_array['output'], tech_array['col'],
                               tech_dict)
        cls.add_matrix_indices(bio_array['input'], bio_array['row'], bio_dict)
        cls.add_matrix_indices(bio_array['output'], bio_array['col'], tech_dict)
        technosphere = cls.build_technosphere_matrix(tech_array, tech_dict)
        biosphere = cls.build_matrix(bio_array, bio_dict, tech_dict, "row", "col", "amount")
        return bio_array, tech_array, bio_dict, tech_dict, biosphere, \
            technosphere

    @classmethod
    def get_technosphere_inputs_mask(cls, array):
        """Get mask of technosphere inputs from ``array``"""
        return np.where(array["type"] ==
                        TYPE_DICTIONARY["technosphere"])

    @classmethod
    def fix_supply_use(cls, array, vector):
        """Make technosphere inputs negative."""
        # Inputs are consumed, so are negative
        mask = cls.get_technosphere_inputs_mask(array)
        vector[mask] = -1 * vector[mask]
        return vector

    @classmethod
    def build_technosphere_matrix(cls, array, tech_dict, new_data=None):
        vector = array["amount"] if new_data is None else new_data
        vector = cls.fix_supply_use(array, vector.copy())
        return cls.build_matrix(array, tech_dict, tech_dict, "row", "col", "amount", vector)
