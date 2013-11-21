from .. import *
from bw2data import config
from bw2data.tests import BW2DataTest
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

TBM = TechnosphereBiosphereMatrixBuilder


class MatrixBuilderTestCase(BW2DataTest):
    def test_load(self):
        a = np.ones(10)
        b = np.arange(10)
        c = np.hstack((a, b))
        with open(os.path.join(config.dir, "processed", "a.pickle"), "wb") as f:
            pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(config.dir, "processed", "b.pickle"), "wb") as f:
            pickle.dump(b, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.assertTrue(np.allclose(
            a,
            MatrixBuilder.load(config.dir, ["a"])
        ))
        self.assertTrue(np.allclose(
            b,
            MatrixBuilder.load(config.dir, ["b"])
        ))
        self.assertTrue(np.allclose(
            c,
            MatrixBuilder.load(config.dir, ["a", "b"])
        ))

    def test_add_matrix_indices(self):
        a = np.arange(10)
        b = np.zeros(10)
        m = {x: x + 100 for x in range(10)}
        MatrixBuilder.add_matrix_indices(a, b, m)
        self.assertTrue(np.allclose(
            b,
            np.arange(100, 110)
        ))
        # Test multiple inputs mapping to same output
        b = np.zeros(10)
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
            ('values', np.float64),
            ('rows', np.uint32),
            ('cols', np.uint32)
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
            ('values', np.float64),
            ('rows', np.uint32),
            ('cols', np.uint32)
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
