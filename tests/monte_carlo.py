# -*- coding: utf-8 -*-
from bw2calc import MonteCarloLCA, DirectSolvingMonteCarloLCA, MultiMonteCarlo, LCA, ParallelMonteCarlo
from bw2data import config, Database, Method, projects
from bw2data.utils import random_string
from numbers import Number
import pytest
import wrapt
from pathlib import Path
import json


fixture_dir = Path(__file__).resolve().parent / "fixtures"

no_pool = pytest.mark.skipif(
    config._windows, reason="fork() on Windows doesn't pass temp directory"
)


def mc_fixture():
    fd = fixture_dir / "mc_basic"
    mapping = {tuple(x): y for x, y in json.load(open(fd / "mapping.json"))}
    packages = [
        fd / "biosphere.zip",
        fd / "test_db.zip",
        fd / "method.zip",
    ]
    return mapping[("test", "1")], mapping[("test", "2")], packages


def test_plain_monte_carlo():
    k1, k2, packages = mc_fixture()
    mc = MonteCarloLCA({k1: 1}, data_objs=packages)
    assert mc.__next__() > 0


def test_monte_carlo_next():
    k1, k2, packages = mc_fixture()
    mc = MonteCarloLCA({k1: 1}, data_objs=packages)
    assert next(mc) > 0


def test_monte_carlo_as_iterator():
    k1, k2, packages = mc_fixture()
    mc = MonteCarloLCA({k1: 1}, data_objs=packages)
    for x, _ in zip(mc, range(10)):
        assert x > 0
        break


def test_direct_solving():
    k1, k2, packages = mc_fixture()
    mc = DirectSolvingMonteCarloLCA({k1: 1}, data_objs=packages)
    assert next(mc)


# def test_parameter_vector_monte_carlo(background):
#     mc = ParameterVectorLCA(*get_args())
#     assert next(mc) > 0


@no_pool
def test_multi_mc():
    k1, k2, packages = mc_fixture()
    mc = MultiMonteCarlo(
        [{k1: 1}, {k2: 1}, {k1: 1, k2: 1}],
        data_objs=packages,
        iterations=10,
    )
    results = mc.calculate()
    assert results


@no_pool
def test_multi_mc_not_same_answer():
    k1, k2, packages = mc_fixture()
    mc = MonteCarloLCA({k1: 1}, data_objs=packages)
    activity_list = [
        {k1: 1},
        {k2: 1},
    ]
    mc = MultiMonteCarlo(activity_list, data_objs=packages, iterations=10)
    results = mc.calculate()
    assert len(results) == 2
    for _, lst in results:
        assert len(set(lst)) == len(lst)

    lca = LCA(activity_list[0], data_objs=packages)
    lca.lci()
    lca.lcia()

    def score(lca, func_unit):
        lca.redo_lcia(func_unit)
        return lca.score

    static = [score(lca, func_unit) for func_unit in activity_list]
    for a, b in zip(static, results):
        assert a not in b[1]


@no_pool
def test_multi_mc_compound_func_units():
    k1, k2, packages = mc_fixture()
    activity_list = [
        {k1: 1},
        {k2: 1},
        {k1: 1, k2: 1},
    ]
    mc = MultiMonteCarlo(activity_list, data_objs=packages, iterations=10)
    results = mc.calculate()
    assert len(results) == 3
    assert activity_list == mc.demands


@no_pool
def test_parallel_monte_carlo():
    k1, k2, packages = mc_fixture()
    mc = ParallelMonteCarlo({k1: 1}, data_objs=packages, iterations=200)
    results = mc.calculate()
    assert results


# https://github.com/brightway-lca/brightway2-calc/issues/28
# pm25_key=('biosphere3', '051aaf7a-6c1a-4e86-999f-85d5f0830df6')

# act1_key=('test_1_act','activity_1')

# biosphere_exchange_1={'amount':1,
#                     'input':pm25_key,
#                     'output':act1_key,
#                     'type':'biosphere',
#                     'uncertainty type': 0}

# production_exchange_1={'amount':1,
#                      'input':act1_key,
#                      'output':act1_key,
#                      'type':'production',
#                      'uncertainty type':0}

# act_1_dict={'name':'activity_1',
#              'unit':'megajoule',
#              'exchanges':[production_exchange_1,biosphere_exchange_1]}

# database_dict={act1_key:act_1_dict}

# db=bw.Database('test_1_act')

# db.write(database_dict)

# a1=bw.get_activity(act1_key)

# # montecarlo, problem after first iteration
# mc1a=bw.MonteCarloLCA({a1:1},bw.methods.random())

# next(mc1a)

# next(mc1a)
