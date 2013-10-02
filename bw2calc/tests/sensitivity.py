# -*- coding: utf-8 -*
from __future__ import division
from stats_arrays.distributions import *
from ..sensitivity import SobolSensitivity
from bw2data import *
from bw2data.tests import BW2DataTest
import numpy as np


class SobolSensitivityTestCase(BW2DataTest):
    def test_generate_matrix(self):
        params = UncertaintyBase.from_dicts(
            {'loc': 2, 'scale': 0.2, 'uncertainty_type': NormalUncertainty.id},
            {'loc': 1.5, 'minimum': 0, 'maximum': 10,
                'uncertainty_type': TriangularUncertainty.id}
        )
        ss = SobolSensitivity()
        matrix = ss.generate_matrix(params, 1000)
        self.assertEqual(matrix.shape, (2, 1000))
        self.assertTrue(1.98 < np.median(matrix[0, :]) < 2.02)

    def test_seed(self):
        params = UncertaintyBase.from_dicts(
            {'loc': 2, 'scale': 0.2, 'uncertainty_type': NormalUncertainty.id},
        )
        ss = SobolSensitivity(seed=17)
        matrix_1 = ss.generate_matrix(params, 1000)
        ss = SobolSensitivity(seed=17)
        matrix_2 = ss.generate_matrix(params, 1000)
        self.assertTrue(np.allclose(matrix_1, matrix_2))

    def test_c_matrix(self):
        a = np.random.random(size=20000).reshape((100, 200))
        b = np.random.random(size=20000).reshape((100, 200))
        ss = SobolSensitivity()
        c = ss.generate_c(a, b, 14)
        self.assertTrue(np.allclose(
            b[:, 13], c[:, 13]
        ))
        self.assertTrue(np.allclose(
            a[:, 14], c[:, 14]
        ))
        self.assertTrue(np.allclose(
            b[:, 15], c[:, 15]
        ))

    def test_evaluate_matrix(self):
        matrix = np.arange(15).reshape((3, 5))
        model = lambda x: x[1]
        ss = SobolSensitivity()
        results = ss.evaluate_matrix(matrix, model)
        self.assertTrue(np.allclose(results, [5, 6, 7, 8, 9]))
