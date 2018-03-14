# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

from .fixtures.presamples_basic import write_database, basedir
from bw2calc.errors import OutsideTechnosphere, NonsquareTechnosphere
from bw2calc import (
    ComparativeMonteCarlo,
    LCA,
    MonteCarloLCA,
    ParameterVectorLCA,
)
from bw2calc.lca import PackagesDataLoader
from bw2data import *
from bw2data.tests import bw2test
from bw2data.utils import TYPE_DICTIONARY
import numpy as np
import pytest
import os


@pytest.fixture
@bw2test
def basic():
    write_database()


# Tests:
# - Test for single sample
# - Test for multiple samples
# - Test for solver cache being invalidated

def test_writing_test_fixture(basic):
    assert len(databases) == 2
    assert len(methods) == 1
    lca  = LCA({("test", "1"): 1})
    lca.lci()
    expected = [
        (("test", "1"), ("test", "1"), 1),
        (("test", "3"), ("test", "1"), -0.1),
        (("test", "2"), ("test", "2"), 0.5),
        (("test", "1"), ("test", "2"), 2),
        (("test", "3"), ("test", "3"), 1),
        (("test", "1"), ("test", "3"), -3),
        (("test", "2"), ("test", "3"), -2),

    ]
    for x, y, z in expected:
        assert np.allclose(lca.technosphere_matrix[
            lca.product_dict[x], lca.activity_dict[y]
        ], z)
    expected = [
        (("bio", "b"), ("test", "1"), 7),
        (("bio", "a"), ("test", "2"), 1),
        (("bio", "b"), ("test", "2"), 5),
        (("bio", "a"), ("test", "3"), 2),
    ]
    for x, y, z in expected:
        assert np.allclose(lca.biosphere_matrix[
            lca.biosphere_dict[x] , lca.activity_dict[y]
        ], z)

@pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
def test_single_sample_presamples(basic):
    ss = os.path.join(basedir, "single-sample")

    lca = LCA({("test", "2"): 1}, method=("m",))
    lca.lci()
    assert np.allclose(
        lca.supply_array,
        np.array([-(2 + 2/3),      14/15, -4/15])
    )
    lca = LCA({("test", "2"): 1}, method=("m",), presamples=[ss])
    lca.lci()
    assert np.allclose(
        lca.supply_array,
        np.array([2, 1.4, 0.2])
    )

    mc = MonteCarloLCA({("test", "2"): 1}, method=("m",))
    next(mc)
    assert np.allclose(
        mc.supply_array,
        np.array([-(2 + 2/3),      14/15, -4/15])
    )
    mc = MonteCarloLCA({("test", "2"): 1}, method=("m",), presamples=[ss])
    next(mc)
    assert np.allclose(
        mc.supply_array,
        np.array([2, 1.4, 0.2])
    )

    mc = ParameterVectorLCA({("test", "2"): 1}, method=("m",))
    next(mc)
    assert mc.technosphere_matrix[
        mc.product_dict[("test", "2")],
        mc.activity_dict[("test", "2")],
    ] == 0.5
    mc = ParameterVectorLCA({("test", "2"): 1}, method=("m",), presamples=[ss])
    next(mc)
    assert mc.technosphere_matrix[
        mc.product_dict[("test", "2")],
        mc.activity_dict[("test", "2")],
    ] == 1

    mc = ComparativeMonteCarlo([{("test", "2"): 1}], method=("m",))
    next(mc)
    assert mc.technosphere_matrix[
        mc.product_dict[("test", "2")],
        mc.activity_dict[("test", "2")],
    ] == 0.5
    mc = ComparativeMonteCarlo([{("test", "2"): 1}], method=("m",),
                               presamples=[ss])
    next(mc)
    assert mc.technosphere_matrix[
        mc.product_dict[("test", "2")],
        mc.activity_dict[("test", "2")],
    ] == 1

@pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
def test_solver_cache_invalidated(basic):
    ss = os.path.join(basedir, "single-sample")
    lca = LCA({("test", "2"): 1}, method=("m",), presamples=[ss])
    lca.lci(factorize=True)
    assert hasattr(lca, "solver")
    lca.presamples.update_matrices()
    assert not hasattr(lca, "solver")

@pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
def test_multi_sample_presamples(basic):
    ss = os.path.join(basedir, "multi")

    lca = LCA({("test", "2"): 1}, method=("m",))
    lca.lci()
    static = lca.technosphere_matrix.data

    multi = []
    for _ in range(10):
        lca = LCA({("test", "2"): 1}, method=("m",), presamples=[ss],
                  seed=42, override_presamples_seed=True)
        lca.lci()
        multi.append(lca.technosphere_matrix.data)

    assert all(np.allclose(multi[i], multi[i + 1]) for i in range(9))
    assert not np.allclose(multi[0], static)

    multi = []
    for _ in range(10):
        lca = LCA({("test", "2"): 1}, method=("m",), presamples=[ss])
        lca.lci()
        multi.append(lca.technosphere_matrix.data)

    assert not all(np.allclose(multi[i], multi[i + 1]) for i in range(9))
    assert not np.allclose(multi[0], static)
