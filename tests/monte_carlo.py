# # -*- coding: utf-8 -*-
# from bw2calc import (
#     MonteCarloLCA,
#     DirectSolvingMonteCarloLCA,
#     MultiMonteCarlo,
#     LCA,
#     ParallelMonteCarlo,
# )
# from bw2data import config, Database, Method, projects
# from bw2data.utils import random_string
# from numbers import Number
# import pytest
# import wrapt
# from pathlib import Path
# import json


# fixture_dir = Path(__file__).resolve().parent / "fixtures"

# no_pool = pytest.mark.skipif(
#     config._windows, reason="fork() on Windows doesn't pass temp directory"
# )


# def mc_fixture():
#     fd = fixture_dir / "mc_basic"
#     mapping = {tuple(x): y for x, y in json.load(open(fd / "mapping.json"))}
#     packages = [
#         fd / "biosphere.zip",
#         fd / "test_db.zip",
#         fd / "method.zip",
#     ]
#     return mapping[("test", "1")], mapping[("test", "2")], packages




# # def test_parameter_vector_monte_carlo(background):
# #     mc = ParameterVectorLCA(*get_args())
# #     assert next(mc) > 0


# @no_pool
# def test_multi_mc():
#     k1, k2, packages = mc_fixture()
#     mc = MultiMonteCarlo(
#         [{k1: 1}, {k2: 1}, {k1: 1, k2: 1}], data_objs=packages, iterations=10,
#     )
#     results = mc.calculate()
#     assert results


# @no_pool
# def test_multi_mc_not_same_answer():
#     k1, k2, packages = mc_fixture()
#     mc = MonteCarloLCA({k1: 1}, data_objs=packages)
#     activity_list = [
#         {k1: 1},
#         {k2: 1},
#     ]
#     mc = MultiMonteCarlo(activity_list, data_objs=packages, iterations=10)
#     results = mc.calculate()
#     assert len(results) == 2
#     for _, lst in results:
#         assert len(set(lst)) == len(lst)

#     lca = LCA(activity_list[0], data_objs=packages)
#     lca.lci()
#     lca.lcia()

#     def score(lca, func_unit):
#         lca.redo_lcia(func_unit)
#         return lca.score

#     static = [score(lca, func_unit) for func_unit in activity_list]
#     for a, b in zip(static, results):
#         assert a not in b[1]


# @no_pool
# def test_multi_mc_compound_func_units():
#     k1, k2, packages = mc_fixture()
#     activity_list = [
#         {k1: 1},
#         {k2: 1},
#         {k1: 1, k2: 1},
#     ]
#     mc = MultiMonteCarlo(activity_list, data_objs=packages, iterations=10)
#     results = mc.calculate()
#     assert len(results) == 3
#     assert activity_list == mc.demands


# @no_pool
# def test_parallel_monte_carlo():
#     k1, k2, packages = mc_fixture()
#     mc = ParallelMonteCarlo({k1: 1}, data_objs=packages, iterations=200)
#     results = mc.calculate()
#     assert results


# def test_single_activity_only_production():
#     # https://github.com/brightway-lca/brightway2-calc/issues/28
#     fd = fixture_dir / "mc_saop"
#     mapping = {tuple(x): y for x, y in json.load(open(fd / "mapping.json"))}
#     packages = [
#         fd / "biosphere.zip",
#         fd / "saop.zip",
#     ]
#     k1 = mapping[("saop", "1")]

#     mc = MonteCarloLCA({k1: 1}, data_objs=packages)
#     next(mc)
#     next(mc)
