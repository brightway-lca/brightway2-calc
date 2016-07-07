# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

from bw2calc import *
from bw2data import config, Database
from bw2data.tests import BW2DataTest
from bw2data.utils import MAX_INT_32, numpy_string
import numpy as np

TBM = TechnosphereBiosphereMatrixBuilder


class MatrixBuilderTestCase(BW2DataTest):
    def test_build_one_d(self):
        database = Database("sour")
        database.register()
        dtype = [
            (numpy_string('a'), np.uint32),
            (numpy_string('row'), np.uint32),
            (numpy_string('values'), np.float32),
        ]
        array = np.array([
            (1, MAX_INT_32, 99),
            (2, MAX_INT_32, 100),
            ], dtype=dtype
        )
        row_dict = {1: 0, 2: 1}
        np.save(database.filepath_processed(), array, allow_pickle=False)
        matrix = MatrixBuilder.build([database.filepath_processed()],
            "values", "a", "row",
            row_dict=row_dict, one_d=True)[3]
        self.assertTrue(np.allclose(
            matrix.todense(),
            np.array(((99, 0), (0, 100)))
        ))

    def test_build_one_d_drop_missing(self):
        database = Database("ghost")
        database.register()
        dtype = [
            (numpy_string('a'), np.uint32),
            (numpy_string('row'), np.uint32),
            (numpy_string('values'), np.float32),
        ]
        array = np.array([
            (1, MAX_INT_32, 99),
            (2, MAX_INT_32, 99),
            (3, MAX_INT_32, 99),
            ], dtype=dtype
        )
        row_dict = {1: 0, 2: 1}
        np.save(database.filepath_processed(), array, allow_pickle=False)
        values = MatrixBuilder.build([database.filepath_processed()],
            "values", "a", "row",
            row_dict=row_dict, one_d=True)[0]
        self.assertEqual(values.shape, (2,))

    def test_one_d_missing_in_row_dict_raise_valueerror(self):
        database = Database("ghost")
        database.register()
        dtype = [
            (numpy_string('a'), np.uint32),
            (numpy_string('row'), np.uint32),
            (numpy_string('values'), np.float32),
        ]
        array = np.array([
            (1, MAX_INT_32, 99),
            (2, MAX_INT_32, 99),
            ], dtype=dtype
        )
        row_dict = {1: 0}
        np.save(database.filepath_processed(), array, allow_pickle=False)
        with self.assertRaises(ValueError):
            MatrixBuilder.build([database.filepath_processed()],
                "values", "a", "row",
                row_dict=row_dict, one_d=True, drop_missing=False)

    def test_build_drop_missing(self):
        database = Database("boo")
        database.register()
        dtype = [
            (numpy_string('a'), np.uint32),
            (numpy_string('b'), np.uint32),
            (numpy_string('row'), np.uint32),
            (numpy_string('col'), np.uint32),
            (numpy_string('values'), np.float32),
        ]
        array = np.array([
            (1, 2, MAX_INT_32, MAX_INT_32, 99),
            (3, 4, MAX_INT_32, MAX_INT_32, 99),
            (3, 2, MAX_INT_32, MAX_INT_32, 99),
            (5, 6, MAX_INT_32, MAX_INT_32, 99),
            ], dtype=dtype
        )
        row_dict = {1: 0, 3: 1}
        col_dict = {2: 0, 6: 1}
        np.save(database.filepath_processed(), array, allow_pickle=False)
        values = MatrixBuilder.build([database.filepath_processed()],
            "values", "a", "row", "b", "col", row_dict, col_dict)[0]
        self.assertEqual(values.shape, (2,))

    def test_missing_in_row_dict_raise_valueerror(self):
        database = Database("whoah")
        database.register()
        dtype = [
            (numpy_string('a'), np.uint32),
            (numpy_string('b'), np.uint32),
            (numpy_string('row'), np.uint32),
            (numpy_string('col'), np.uint32),
            (numpy_string('values'), np.float32),
        ]
        array = np.array([
            (1, 2, MAX_INT_32, MAX_INT_32, 99),
            (1, 4, MAX_INT_32, MAX_INT_32, 99),
            ], dtype=dtype
        )
        row_dict = {1: 0}
        col_dict = {2: 0}
        np.save(database.filepath_processed(), array, allow_pickle=False)
        with self.assertRaises(ValueError):
            MatrixBuilder.build([database.filepath_processed()],
                "values", "a", "row", "b", "col", row_dict, col_dict,
                drop_missing=False)

    def test_add_matrix_indices(self):
        a = np.arange(10).astype(np.uint32)
        b = np.zeros(10).astype(np.uint32)
        m = {x: x + 100 for x in range(10)}
        MatrixBuilder.add_matrix_indices(a, b, m)
        self.assertTrue(np.allclose(
            b,
            np.arange(100, 110)
        ))
        # Test multiple inputs mapping to same output
        b = np.zeros(10).astype(np.uint32)
        m = {x: 42 for x in range(10)}
        MatrixBuilder.add_matrix_indices(a, b, m)
        self.assertTrue(np.allclose(
            b,
            np.ones(10) * 42
        ))

    def test_build_dictionary(self):
        a = np.arange(10, 30)
        a[10:] = 19
        np.random.shuffle(a)
        d = MatrixBuilder.build_dictionary(a)
        self.assertEqual(d, {x+10:x for x in range(10)})

    def test_build_matrix(self):
        a = np.zeros((4,), dtype=[
            (numpy_string('values'), np.float64),
            (numpy_string('rows'), np.uint32),
            (numpy_string('cols'), np.uint32)
        ])
        a[0] = (4.2, 0, 2)
        a[1] = (6.6, 1, 1)
        a[2] = (1.3, 2, 1)
        a[3] = (10, 2, 2)
        r = [0,0,0]  # Just need right length
        matrix = MatrixBuilder.build_matrix(
            array=a,
            row_dict=r,
            col_dict=r,
            row_index_label='rows',
            col_index_label='cols',
            data_label='values'
        )
        answer = np.array((
            (0, 0, 4.2),
            (0, 6.6, 0),
            (0, 1.3, 10)
        ))
        self.assertTrue(np.allclose(answer, matrix.todense()))

    def test_multiple_values_same_exchange(self):
        """Values for same (row, col) should add together"""
        a = np.zeros((2,), dtype=[
            (numpy_string('values'), np.float64),
            (numpy_string('rows'), np.uint32),
            (numpy_string('cols'), np.uint32)
        ])
        a[0] = (9, 1, 1)
        a[1] = (33, 1, 1)
        r = [0,0,0]  # Just need right length
        matrix = MatrixBuilder.build_matrix(
            array=a,
            row_dict=r,
            col_dict=r,
            row_index_label='rows',
            col_index_label='cols',
            data_label='values'
        )
        answer = np.array((
            (0, 0, 0),
            (0, 42, 0),
            (0, 0, 0)
        ))
        self.assertTrue(np.allclose(answer, matrix.todense()))
