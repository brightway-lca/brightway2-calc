from collections import defaultdict
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


def test_single_demand(dps, config):
    single_func_unit = {"γ": {100: 1}}
    mlca = MultiLCA(demands=single_func_unit, method_config=config, data_objs=dps)
    mlca.lci()
    mlca.lcia()

    assert mlca.scores[(("first", "category"), "γ")] == 8 + 3
    assert mlca.scores[(("second", "category"), "γ")] == 3 * 10


def test_normalization(dps, func_units):
    config = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "normalizations": {
            ("n", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
    }

    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
    )

    mlca = MultiLCA(demands=func_units, method_config=config, data_objs=dps)
    mlca.lci()
    mlca.lcia()
    mlca.normalize()

    assert len(mlca.normalization_matrices) == 1
    assert len(mlca.normalized_inventories) == len(mlca.characterized_inventories)

    rows = np.zeros(mlca.biosphere_matrix.shape[0])
    rows = np.array([mlca.dicts.biosphere[201], mlca.dicts.biosphere[203]])

    for key, mat in mlca.normalized_inventories.items():
        expected = mlca.characterized_inventories[key[1:]][rows, :].sum()
        assert mat.sum() == expected


def test_normalization_with_weighting(dps, func_units):
    config = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "normalizations": {
            ("n", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
        "weightings": {("w", "1"): [("n", "1")]},
    }

    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
    )
    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
    )

    mlca = MultiLCA(demands=func_units, method_config=config, data_objs=dps)
    mlca.lci()
    mlca.lcia()
    mlca.normalize()
    mlca.weight()

    assert len(mlca.weighting_matrices) == 1
    assert len(mlca.normalized_inventories) == len(mlca.characterized_inventories)
    assert len(mlca.weighted_inventories) == len(mlca.characterized_inventories)

    rows = np.zeros(mlca.biosphere_matrix.shape[0])
    rows = np.array([mlca.dicts.biosphere[201], mlca.dicts.biosphere[203]])

    for k, v in mlca.weighting_matrices.items():
        print(k)
        print(v.todense())

    for key, mat in mlca.weighted_inventories.items():
        expected = mlca.characterized_inventories[key[2:]][rows, :].sum()
        assert np.allclose(mat.sum(), expected * 42)


def test_normalization_without_weighting(dps, func_units):
    config = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "weightings": {
            ("w", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
    }

    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
    )

    mlca = MultiLCA(demands=func_units, method_config=config, data_objs=dps)
    mlca.lci()
    mlca.lcia()
    mlca.weight()

    assert len(mlca.weighting_matrices) == 1
    assert len(mlca.weighted_inventories) == len(mlca.characterized_inventories)

    for key, mat in mlca.weighted_inventories.items():
        expected = mlca.characterized_inventories[key[1:]].sum()
        assert np.allclose(mat.sum(), expected * 42)


def test_selective_use(dps, func_units):
    config = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "normalizations": {
            ("n", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
        "weightings": {("w", "1"): [("n", "1")]},
    }

    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
    )
    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
    )

    su = {
        "characterization_matrix": {"use_distributions": True},
        "weighting_matrix": {"use_distributions": True},
    }

    mlca = MultiLCA(demands=func_units, method_config=config, data_objs=dps, selective_use=su)
    mlca.lci()
    mlca.lcia()
    mlca.normalize()
    mlca.weight()

    results = {key: mat.sum() for key, mat in mlca.weighted_inventories.items()}
    w = {key: mat.sum() for key, mat in mlca.weighting_matrices.items()}
    n = {key: mat.sum() for key, mat in mlca.normalization_matrices.items()}
    c = {key: mat.sum() for key, mat in mlca.characterization_matrices.items()}
    t = mlca.technosphere_matrix.sum()
    b = mlca.biosphere_matrix.sum()

    next(mlca)

    assert not any(
        np.allclose(mat.sum(), results[key]) for key, mat in mlca.weighted_inventories.items()
    )
    assert not any(np.allclose(mat.sum(), w[key]) for key, mat in mlca.weighting_matrices.items())
    assert all(np.allclose(mat.sum(), n[key]) for key, mat in mlca.normalization_matrices.items())
    assert not any(
        np.allclose(mat.sum(), c[key]) for key, mat in mlca.characterization_matrices.items()
    )
    assert np.allclose(b, mlca.biosphere_matrix.sum())
    assert np.allclose(t, mlca.technosphere_matrix.sum())


def test_selective_use_keep_first(dps, func_units):
    config = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "normalizations": {
            ("n", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
        "weightings": {("w", "1"): [("n", "1")]},
    }

    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
    )
    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
    )

    su = {
        "characterization_matrix": {"use_distributions": True},
        "weighting_matrix": {"use_distributions": True},
    }

    mlca = MultiLCA(demands=func_units, method_config=config, data_objs=dps, selective_use=su)
    mlca.lci()
    mlca.lcia()
    mlca.normalize()
    mlca.weight()
    mlca.keep_first_iteration()

    results = {key: mat.sum() for key, mat in mlca.weighted_inventories.items()}
    w = {key: mat.sum() for key, mat in mlca.weighting_matrices.items()}
    n = {key: mat.sum() for key, mat in mlca.normalization_matrices.items()}
    c = {key: mat.sum() for key, mat in mlca.characterization_matrices.items()}
    t = mlca.technosphere_matrix.sum()
    b = mlca.biosphere_matrix.sum()

    next(mlca)

    assert all(
        np.allclose(mat.sum(), results[key]) for key, mat in mlca.weighted_inventories.items()
    )
    assert all(np.allclose(mat.sum(), w[key]) for key, mat in mlca.weighting_matrices.items())
    assert all(np.allclose(mat.sum(), n[key]) for key, mat in mlca.normalization_matrices.items())
    assert all(
        np.allclose(mat.sum(), c[key]) for key, mat in mlca.characterization_matrices.items()
    )
    assert np.allclose(b, mlca.biosphere_matrix.sum())
    assert np.allclose(t, mlca.technosphere_matrix.sum())


def test_monte_carlo_multiple_iterations_use_distributions(dps, func_units):
    config = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "normalizations": {
            ("n", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
        "weightings": {("w", "1"): [("n", "1")]},
    }

    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
    )
    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
    )

    mlca = MultiLCA(demands=func_units, method_config=config, data_objs=dps, use_distributions=True)
    mlca.lci()
    mlca.lcia()
    mlca.normalize()
    mlca.weight()

    results_manual = {key: [mat.sum()] for key, mat in mlca.weighted_inventories.items()}
    results_scores = {k: [v] for k, v in mlca.scores.items()}

    for _ in range(9):
        next(mlca)
        for key, mat in mlca.weighted_inventories.items():
            results_manual[key].append(mat.sum())
        for key, val in mlca.scores.items():
            results_scores[key].append(val)

    for key, lst in results_manual.items():
        assert np.unique(lst).shape == (10,)
    for key, lst in results_scores.items():
        assert np.unique(lst).shape == (10,)


def test_monte_carlo_multiple_iterations_selective_use(dps, func_units):
    config = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "normalizations": {
            ("n", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
        "weightings": {("w", "1"): [("n", "1")]},
    }

    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
    )
    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
    )

    su = {
        "characterization_matrix": {"use_distributions": True},
        "weighting_matrix": {"use_distributions": True},
    }

    mlca = MultiLCA(demands=func_units, method_config=config, data_objs=dps, selective_use=su)
    mlca.lci()
    mlca.lcia()
    mlca.normalize()
    mlca.weight()

    results_manual = {key: [mat.sum()] for key, mat in mlca.weighted_inventories.items()}
    results_scores = {k: [v] for k, v in mlca.scores.items()}

    for _ in range(9):
        next(mlca)
        for key, mat in mlca.weighted_inventories.items():
            results_manual[key].append(mat.sum())
        for key, val in mlca.scores.items():
            results_scores[key].append(val)

    for key, lst in results_manual.items():
        assert np.unique(lst).shape == (10,)
    for key, lst in results_scores.items():
        assert np.unique(lst).shape == (10,)


def test_monte_carlo_multiple_iterations_selective_use_in_list_comprehension(dps, func_units):
    config = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "normalizations": {
            ("n", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
        "weightings": {("w", "1"): [("n", "1")]},
    }

    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
    )
    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
    )

    su = {
        "characterization_matrix": {"use_distributions": True},
        "weighting_matrix": {"use_distributions": True},
    }

    mlca = MultiLCA(demands=func_units, method_config=config, data_objs=dps, selective_use=su)
    mlca.lci()
    mlca.lcia()
    mlca.normalize()
    mlca.weight()

    results = [mlca.scores for _ in zip(range(10), mlca)]

    aggregated = defaultdict(list)
    for line in results:
        for k, v in line.items():
            aggregated[k].append(v)

    for key, lst in aggregated.items():
        assert np.unique(lst).shape == (10,)
