# # -*- coding: utf-8 -*-
# from .fixtures.presamples_basic import write_database, basedir
# from bw2calc import (
#     ComparativeMonteCarlo,
#     LCA,
#     MonteCarloLCA,
#     ParameterVectorLCA,
# )
# from bw2calc.lca import PackagesDataLoader
# from bw2data import *
# from bw2data.tests import bw2test
# from pathlib import Path
# import numpy as np
# import os
# import pytest


# basedir = Path(__file__).resolve() / "fixtures"

# @pytest.fixture
# @bw2test
# def basic():
#     write_database()


# def test_writing_test_fixture(basic):
#     assert len(databases) == 2
#     assert len(methods) == 1
#     lca = LCA({("test", "1"): 1})
#     lca.lci()
#     expected = [
#         (("test", "1"), ("test", "1"), 1),
#         (("test", "3"), ("test", "1"), -0.1),
#         (("test", "2"), ("test", "2"), 0.5),
#         (("test", "1"), ("test", "2"), 2),
#         (("test", "3"), ("test", "3"), 1),
#         (("test", "1"), ("test", "3"), -3),
#         (("test", "2"), ("test", "3"), -2),
#     ]
#     for x, y, z in expected:
#         assert np.allclose(
#             lca.technosphere_matrix[lca.dicts.product[x], lca.dicts.activity[y]], z
#         )
#     expected = [
#         (("bio", "b"), ("test", "1"), 7),
#         (("bio", "a"), ("test", "2"), 1),
#         (("bio", "b"), ("test", "2"), 5),
#         (("bio", "a"), ("test", "3"), 2),
#     ]
#     for x, y, z in expected:
#         assert np.allclose(
#             lca.biosphere_matrix[lca.dicts.biosphere[x], lca.dicts.activity[y]], z
#         )


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_accept_pathlib(basic):
#     ss = Path(basedir) / "single-sample"
#     lca = LCA({("test", "2"): 1}, method=("m",), presamples=[ss])
#     lca.lci()


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_single_sample_presamples(basic):
#     ss = os.path.join(basedir, "single-sample")

#     lca = LCA({("test", "2"): 1}, method=("m",))
#     lca.lci()
#     assert np.allclose(lca.supply_array, np.array([-(2 + 2 / 3), 14 / 15, -4 / 15]))
#     lca = LCA({("test", "2"): 1}, method=("m",), presamples=[ss])
#     lca.lci()
#     assert np.allclose(lca.supply_array, np.array([2, 1.4, 0.2]))

#     mc = MonteCarloLCA({("test", "2"): 1}, method=("m",))
#     next(mc)
#     assert np.allclose(mc.supply_array, np.array([-(2 + 2 / 3), 14 / 15, -4 / 15]))
#     mc = MonteCarloLCA({("test", "2"): 1}, method=("m",), presamples=[ss])
#     next(mc)
#     assert np.allclose(mc.supply_array, np.array([2, 1.4, 0.2]))

#     mc = ParameterVectorLCA({("test", "2"): 1}, method=("m",))
#     next(mc)
#     assert (
#         mc.technosphere_matrix[
#             mc.dicts.product[("test", "2")], mc.dicts.activity[("test", "2")],
#         ]
#         == 0.5
#     )
#     mc = ParameterVectorLCA({("test", "2"): 1}, method=("m",), presamples=[ss])
#     next(mc)
#     assert (
#         mc.technosphere_matrix[
#             mc.dicts.product[("test", "2")], mc.dicts.activity[("test", "2")],
#         ]
#         == 1
#     )

#     mc = ComparativeMonteCarlo([{("test", "2"): 1}], method=("m",))
#     next(mc)
#     assert (
#         mc.technosphere_matrix[
#             mc.dicts.product[("test", "2")], mc.dicts.activity[("test", "2")],
#         ]
#         == 0.5
#     )
#     mc = ComparativeMonteCarlo([{("test", "2"): 1}], method=("m",), presamples=[ss])
#     next(mc)
#     assert (
#         mc.technosphere_matrix[
#             mc.dicts.product[("test", "2")], mc.dicts.activity[("test", "2")],
#         ]
#         == 1
#     )


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_solver_cache_invalidated(basic):
#     ss = os.path.join(basedir, "single-sample")
#     lca = LCA({("test", "2"): 1}, method=("m",), presamples=[ss])
#     lca.lci(factorize=True)
#     assert hasattr(lca, "solver")
#     lca.presamples.update_matrices()
#     assert not hasattr(lca, "solver")


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_multi_sample_presamples(basic):
#     path = os.path.join(basedir, "multi")

#     lca = LCA({("test", "2"): 1}, method=("m",))
#     lca.lci()
#     static = lca.technosphere_matrix.data

#     multi = []
#     for _ in range(10):
#         lca = LCA({("test", "2"): 1}, method=("m",), presamples=[path])
#         lca.lci()
#         multi.append(lca.technosphere_matrix.data)

#     assert all(np.allclose(multi[i], multi[i + 1]) for i in range(9))
#     for x in range(9):
#         assert not np.allclose(multi[x], static)


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_multi_sample_presamples_no_seed_different(basic):
#     path = os.path.join(basedir, "unseeded")

#     multi = []
#     for _ in range(10):
#         lca = LCA({("test", "2"): 1}, method=("m",), presamples=[path])
#         lca.lci()
#         multi.append(lca.technosphere_matrix.data)

#     assert not all(np.allclose(multi[i], multi[i + 1]) for i in range(9))


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_keep_presamples_seed(basic):
#     path = os.path.join(basedir, "multi")

#     mc = MonteCarloLCA({("test", "2"): 1}, method=("m",), presamples=[path], seed=6)
#     first = [next(mc) for _ in range(10)]
#     mc = MonteCarloLCA({("test", "2"): 1}, method=("m",), presamples=[path], seed=7)
#     second = [next(mc) for _ in range(10)]
#     assert first == second


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_override_presamples_seed(basic):
#     path = os.path.join(basedir, "multi")

#     mc = MonteCarloLCA(
#         {("test", "2"): 1},
#         method=("m",),
#         presamples=[path],
#         seed=6,
#         override_presamples_seed=True,
#     )
#     first = [next(mc) for _ in range(10)]
#     mc = MonteCarloLCA(
#         {("test", "2"): 1},
#         method=("m",),
#         presamples=[path],
#         seed=7,
#         override_presamples_seed=True,
#     )
#     second = [next(mc) for _ in range(10)]
#     assert first != second


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_sequential_seed():
#     path = os.path.join(basedir, "seq")
#     lca = LCA({("test", "2"): 1}, method=("m",), presamples=[path])
#     lca.lci()
#     assert (
#         lca.technosphere_matrix[
#             lca.dicts.product[("test", "1")], lca.dicts.activity[("test", "2")],
#         ]
#         == -1
#     )
#     lca.presamples.update_matrices()
#     assert (
#         lca.technosphere_matrix[
#             lca.dicts.product[("test", "1")], lca.dicts.activity[("test", "2")],
#         ]
#         == -2
#     )
#     lca.presamples.update_matrices()
#     assert (
#         lca.technosphere_matrix[
#             lca.dicts.product[("test", "1")], lca.dicts.activity[("test", "2")],
#         ]
#         == -3
#     )
#     lca.presamples.update_matrices()
#     assert (
#         lca.technosphere_matrix[
#             lca.dicts.product[("test", "1")], lca.dicts.activity[("test", "2")],
#         ]
#         == -1
#     )


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_sequential_seed_monte_carlo():
#     path = os.path.join(basedir, "seq")
#     mc = MonteCarloLCA({("test", "2"): 1}, method=("m",), presamples=[path])
#     next(mc)
#     assert (
#         mc.technosphere_matrix[
#             mc.dicts.product[("test", "1")], mc.dicts.activity[("test", "2")],
#         ]
#         == -1
#     )
#     next(mc)
#     assert (
#         mc.technosphere_matrix[
#             mc.dicts.product[("test", "1")], mc.dicts.activity[("test", "2")],
#         ]
#         == -2
#     )
#     next(mc)
#     assert (
#         mc.technosphere_matrix[
#             mc.dicts.product[("test", "1")], mc.dicts.activity[("test", "2")],
#         ]
#         == -3
#     )
#     next(mc)
#     assert (
#         mc.technosphere_matrix[
#             mc.dicts.product[("test", "1")], mc.dicts.activity[("test", "2")],
#         ]
#         == -1
#     )


# @pytest.mark.skipif(not PackagesDataLoader, reason="presamples not installed")
# def test_call_update_matrices_manually(basic):
#     path = os.path.join(basedir, "multi")

#     lca = LCA({("test", "2"): 1}, method=("m",), presamples=[path])
#     lca.lci()
#     lca.lcia()

#     results = set()
#     for _ in range(100):
#         lca.presamples.update_matrices()
#         lca.redo_lci()
#         lca.redo_lcia()
#         results.add(lca.score)

#     assert len(results) > 1
