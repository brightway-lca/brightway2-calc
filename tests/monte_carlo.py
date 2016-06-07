# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

from bw2calc import *
from bw2data import config, Database, Method
from bw2data.tests import bw2test
import pytest


@pytest.fixture
def background():
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


def get_args():
    return {Database("test").random(): 1}, ("a", "method")


@bw2test
def test_plain_monte_carlo(background):
    print(projects.current)
    print(projects.dir)
    print(str(databases))
    mc = MonteCarloLCA(*get_args())
    assert next(mc)
