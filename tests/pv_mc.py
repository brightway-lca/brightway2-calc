# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

from bw2calc import *
from bw2data import Database, Method
from bw2data.tests import bw2test
import numpy as np
import pytest


def build_databases():
    Database("biosphere").write({
        ("biosphere", "1"): {'type': 'emission'},
        ("biosphere", "2"): {'type': 'emission'},
    })
    Database("test").write({
        ("test", "1"): {
            'exchanges': [{
                'amount': 0.5,
                'minimum': 0.2,
                'maximum': 0.8,
                'input': ('test', "2"),
                'type': 'technosphere',
                'uncertainty type': 4  # Uniform
            }, {
                'amount': 100,
                'minimum': 50,
                'input': ('biosphere', "1"),
                'type': 'biosphere',
                'loc': 100,
                'scale': 20,
                'uncertainty type': 3  # Normal
            }],
            'type': 'process',
        },
        ("test", "2"): {
            'exchanges': [{
                'amount': -0.42,
                'input': ('biosphere', "2"),
                'type': 'biosphere',
            }],
            'type': 'process',
            'unit': 'kg'
        },
    })
    method = Method(("a", "method"))
    method.register()
    method.write([
        (("biosphere", "1"), 1),
        (("biosphere", "2"), {
            'amount': 10,
            'uncertainty type': 5,  # Triangular
            'loc': 10,
            'minimum': 8,
            'maximum': 15
        }),
    ])


@pytest.fixture
@bw2test
def background():
    build_databases()


def test_pv_no_need_load_data(background):
    assert next(ParameterVectorLCA({("test", "1"): 1}, ("a", "method")))


def test_pv_ordering_correct(background):
    pv = ParameterVectorLCA({("test", "1"): 1}, ("a", "method"))
    pv.load_data()

    b1 = pv.biosphere_dict[("biosphere", "1")]
    b2 = pv.biosphere_dict[("biosphere", "2")]
    p1 = pv.product_dict[("test", "1")]
    p2 = pv.product_dict[("test", "2")]
    a1 = pv.activity_dict[("test", "1")]
    a2 = pv.activity_dict[("test", "2")]

    for _ in range(10):
        next(pv)
        assert pv.characterization_matrix[b1, b1] == 1
        assert 8 <= pv.characterization_matrix[b2, b2] <= 15
        assert pv.characterization_matrix[b2, b1] == 0

        assert np.allclose(pv.biosphere_matrix[b2, a2], -0.42)
        assert pv.biosphere_matrix[b1, a1] >= 50

        assert pv.technosphere_matrix[p1, a1] == 1
        assert pv.technosphere_matrix[p2, a2] == 1
        assert pv.technosphere_matrix[p1, a2] == 0
        # Inputs are negative
        assert -0.8 <= pv.technosphere_matrix[p2, a1] <= -0.2

        assert 8 <= pv.characterization_matrix[b2, b2] <= 15
        assert pv.characterization_matrix[b2, b1] == 0


def test_raise_assertion_error_with_wrong_size_vector(background):
    pv = ParameterVectorLCA({("test", "1"): 1}, ("a", "method"))
    with pytest.raises(AssertionError):
        pv.rebuild_all(np.ones(100,))


def test_no_error_with_right_size_vector(background):
    pv = ParameterVectorLCA({("test", "1"): 1}, ("a", "method"))
    pv.rebuild_all(np.ones(7,))


def test_stored_samples_correct(background):
    pv = ParameterVectorLCA({("test", "1"): 1}, ("a", "method"))
    pv.load_data()

    samples = []

    for x in range(10):
        next(pv)
        samples.append(pv.sample.copy())

    # Sample has length 7
    # 0, 1: Production exchanges, no uncertainty
    # 2: Tech input: 0.2-0.8, uniform uncertainty
    # 3: Biosphere: 50+, normal uncertainty
    # 4: Biosphere: -0.42, no uncertainty
    # 5: CF: 1, no uncertainty
    # 6: CF: 8-15, triangular uncertainty

    for sample in samples:
        assert sample[0] == 1
        assert sample[1] == 1
        assert sample[2] > 0
        assert sample[3] >= 50
        assert np.allclose(sample[4], -0.42)
        assert sample[5] == 1
        assert 8 <= sample[6] <= 15
