# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

from bw2calc import *
from bw2data import config, Database, Method, projects, databases
from bw2data.utils import random_string
from bw2data.tests import bw2test
from numbers import Number
import pytest
import wrapt


no_pool = pytest.mark.skipif(config._windows,
                             reason="fork() on Windows doesn't pass temp directory")


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
                'uncertainty type': 4
            }, {
                'amount': 1,
                'minimum': 0.5,
                'maximum': 1.5,
                'input': ('biosphere', "1"),
                'type': 'biosphere',
                'uncertainty type': 4
            }],
            'type': 'process',
        },
        ("test", "2"): {
            'exchanges': [{
                'amount': 0.1,
                'minimum': 0,
                'maximum': 0.2,
                'input': ('biosphere', "2"),
                'type': 'biosphere',
                'uncertainty type': 4
            }],
            'type': 'process',
            'unit': 'kg'
        },
    })
    method = Method(("a", "method"))
    method.register()
    method.write([
        (("biosphere", "1"), 1),
        (("biosphere", "2"), 2),
    ])


@pytest.fixture
@bw2test
def background():
    build_databases()


@wrapt.decorator
def random_project(wrapped, instance, args, kwargs):
    config.is_test = True
    projects._restore_orig_directory()
    string = random_string()
    while string in projects:
        string = random_string()
    projects.set_current(string)
    build_databases()
    result = wrapped(*args, **kwargs)
    projects.set_current("default", writable=False)
    projects.delete_project(name=string, delete_dir=True)
    return result


def get_args():
    return {("test", "1"): 1}, ("a", "method")


@random_project
def test_random_project():
    assert "Brightway" in projects.dir

@bw2test
def test_temp_dir_again():
    assert "Brightway" not in projects.dir

def test_plain_monte_carlo(background):
    mc = MonteCarloLCA(*get_args())
    if hasattr(mc, "__next__"):
        assert mc.__next__() > 0
    else:
        assert mc.next() > 0

def test_monte_carlo_next(background):
    mc = MonteCarloLCA(*get_args())
    assert next(mc) > 0

def test_monte_carlo_as_iterator(background):
    mc = MonteCarloLCA(*get_args())
    for x in mc:
        assert x > 0
        break

def test_direct_solving(background):
    mc = DirectSolvingMonteCarloLCA(*get_args())
    assert next(mc)

def test_parameter_vector_monte_carlo(background):
    mc = ParameterVectorLCA(*get_args())
    assert next(mc) > 0

@no_pool
def test_multi_mc(background):
    mc = MultiMonteCarlo(
        [
            {("test", "1"): 1},
            {("test", "2"): 1},
            {("test", "1"): 1, ("test", "2"): 1}
        ],
        ("a", "method"),
        iterations=10
    )
    results = mc.calculate()
    print(results)
    assert results

@random_project
def test_multi_mc_no_temp_dir():
    mc = MultiMonteCarlo(
        [
            {("test", "1"): 1},
            {("test", "2"): 1},
            {("test", "1"): 1, ("test", "2"): 1}
        ],
        ("a", "method"),
        iterations=10
    )
    results = mc.calculate()
    print(results)
    assert results
    assert isinstance(results, list)
    assert len(results)

@no_pool
def test_parallel_monte_carlo(background):
    fu, method = get_args()
    mc = ParallelMonteCarlo(fu, method, iterations=200)
    results = mc.calculate()
    print(results)
    assert results

@random_project
def test_parallel_monte_carlo_no_temp_dir():
    fu, method = get_args()
    mc = ParallelMonteCarlo(fu, method, iterations=200)
    results = mc.calculate()
    print(results)
    assert results
    assert isinstance(results, list)
    assert isinstance(results[0], Number)
    assert results[0] > 0
