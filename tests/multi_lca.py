from pathlib import Path

import numpy as np
import pytest

from bw2calc import MultiLCA
from bw2calc.utils import get_datapackage

# Technosphere
#           α: 1    β: 2    γ: 3    δ: 4    ε: 5    ζ: 6
#   L: 100  -0.2    -0.5    1       -0.1            -0.1
#   M: 101  1                       -0.2    -0.1
#   N: 102  -0.5    1                               -0.2
#   O: 103                          -0.4    1
#   P: 104                          1
#   Q: 105                                          1

# Biosphere
#           α: 1    β: 2    γ: 3    δ: 4    ε: 5    ζ: 6
#   200     2       4       8       1       2       1
#   201     1       2       3
#   202                                     1       2
#   203                                             3

fixture_dir = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def dps():
    return [
        get_datapackage(fixture_dir / "multi_lca_simple_1.zip"),
        get_datapackage(fixture_dir / "multi_lca_simple_2.zip"),
        get_datapackage(fixture_dir / "multi_lca_simple_3.zip"),
        get_datapackage(fixture_dir / "multi_lca_simple_4.zip"),
        get_datapackage(fixture_dir / "multi_lca_simple_5.zip"),
    ]


@pytest.fixture
def config():
    return {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ]
    }


@pytest.fixture
def func_units():
    return {
        "γ": {100: 1},
        "ε": {103: 2},
        "ζ": {105: 3},
    }


def test_inventory_matrix_construction(dps, config, func_units):
    mlca = MultiLCA(demands=func_units, method_config=config, data_objs=dps)
    mlca.lci()
    mlca.lcia()

    print(mlca.scores)
    print(mlca.technosphere_matrix.todense())
    print(mlca.biosphere_matrix.todense())

    for name, mat in mlca.characterization_matrices.items():
        print(name)
        print(mat.todense())

    for name, arr in mlca.supply_arrays.items():
        print(name)
        print(arr)

    tm = [
        (100, 1, -0.2),
        (100, 2, -0.5),
        (100, 3, 1),
        (100, 4, -0.1),
        (100, 6, -0.1),
    ]
    for a, b, c in tm:
        assert mlca.technosphere_matrix[mlca.dicts.product[a], mlca.dicts.activity[b]] == c

    assert (
        mlca.technosphere_matrix.sum()
        == np.array([-0.2, -0.5, 1, -0.1, -0.1, 1, -0.2, -0.1, -0.2, -0.5, 1, -0.4, 1, 1, 1]).sum()
    )

    bm = [
        (200, 1, 2),
        (200, 2, 4),
        (200, 3, 8),
        (200, 4, 1),
        (200, 5, 2),
        (200, 6, 1),
    ]
    for a, b, c in bm:
        assert mlca.biosphere_matrix[mlca.dicts.biosphere[a], mlca.dicts.activity[b]] == c

    assert mlca.biosphere_matrix.sum() == np.array([2, 4, 8, 1, 2, 1, 1, 2, 3, 1, 2, 3]).sum()

    for a in range(200, 204):
        assert (
            mlca.characterization_matrices[("first", "category")][
                mlca.dicts.biosphere[a], mlca.dicts.biosphere[a]
            ]
            == 1
        )

    assert mlca.characterization_matrices[("first", "category")].sum() == 4

    assert (
        mlca.characterization_matrices[("second", "category")][
            mlca.dicts.biosphere[201], mlca.dicts.biosphere[201]
        ]
        == 10
    )
    assert (
        mlca.characterization_matrices[("second", "category")][
            mlca.dicts.biosphere[203], mlca.dicts.biosphere[203]
        ]
        == 10
    )
    assert mlca.characterization_matrices[("second", "category")].sum() == 20

    assert mlca.scores[(("second", "category"), "ζ")] == 3 * (3 * 10 + 1 * 10)
    assert mlca.scores[(("first", "category"), "γ")] == 8 + 3


def test_consistent_indexing():
    pass
