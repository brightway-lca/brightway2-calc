import bw2calc as bc
from bw_processing import create_datapackage, INDICES_DTYPE
import numpy as np
import pytest


def test_multifunctional_x_shape_one_path():
    dp = create_datapackage()

    data_array = np.array([10, 100])
    indices_array = np.array([(20, 0), (21, 0)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="c",
        indices_array=indices_array,
    )

    data_array = np.array([1, 1])
    indices_array = np.array([(20, 10), (21, 11)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="b",
        indices_array=indices_array,
    )

    data_array = np.array([-1, 1, 1, 1, -1, -1, -1, 1])
    indices_array = np.array([
        (1, 13),
        (3, 10),
        (3, 11),
        (4, 11),
        (4, 13),
        (3, 13),
        (3, 12),
        (2, 12),
    ], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="t",
        indices_array=indices_array,
    )

    lca = bc.LCA({2: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()

    assert lca.score == 10

    results = bc.MultifunctionalGraphTraversal.calculate(lca=lca)
    assert results['products'] == {2: {'amount': 1.0, 'supply_chain_score': 10.0}, 3: {'amount': 1.0, 'supply_chain_score': 10.0}}
    assert results['activities'] == {
        -1: {'amount': 1, 'direct_score': 0},
        12: {'amount': 1.0, 'direct_score': 0.0},
        10: {'amount': 1.0, 'direct_score': 10.0}
    }
    assert results['edges'] == [
        {'target': 2, 'source': -1, 'type': 'product', 'amount': 1.0, 'exc_amount': 1.0, 'supply_chain_score': 10.0},
        {'target': 12, 'source': 2, 'type': 'activity', 'amount': 1.0, 'exc_amount': 1.0, 'direct_score': 0.0},
        {'target': 3, 'source': 12, 'type': 'product', 'amount': 1.0, 'exc_amount': -1.0, 'supply_chain_score': 10.0},
        {'target': 10, 'source': 3, 'type': 'activity', 'amount': 1.0, 'exc_amount': 1.0, 'direct_score': 10.0}
    ]
    assert results['counter'] == 2


def test_multifunctional_coproduction():
    dp = create_datapackage()

    data_array = np.array([10, 100])
    indices_array = np.array([(20, 0), (21, 0)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="c",
        indices_array=indices_array,
    )

    data_array = np.array([1, 1])
    indices_array = np.array([(20, 10), (21, 11)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="b",
        indices_array=indices_array,
    )

    data_array = np.array([1, 1, 1])
    indices_array = np.array([
        (1, 10),
        (2, 10),
        (2, 11),
    ], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="t",
        indices_array=indices_array,
    )

    lca = bc.LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()

    assert lca.score == -90

    results = bc.MultifunctionalGraphTraversal.calculate(lca=lca)
    assert results['products'] == {1: {'amount': 1.0, 'supply_chain_score': -90.0}, 2: {'amount': -1.0, 'supply_chain_score': -100.0}}
    assert results['activities'] == {
        -1: {'amount': 1, 'direct_score': 0},
        10: {'amount': 1.0, 'direct_score': 10.0},
        11: {'amount': -1.0, 'direct_score': -100.0}
    }
    assert results['edges'] == [
        {'target': 1, 'source': -1, 'type': 'product', 'amount': 1.0, 'exc_amount': 1.0, 'supply_chain_score': -90.0},
        {'target': 10, 'source': 1, 'type': 'activity', 'amount': 1.0, 'exc_amount': 1.0, 'direct_score': 10.0},
        {'target': 2, 'source': 10, 'type': 'product', 'amount': -1.0, 'exc_amount': 1.0, 'supply_chain_score': -100.0},
        {'target': 11, 'source': 2, 'type': 'activity', 'amount': -1.0, 'exc_amount': 1.0, 'direct_score': -100.0}
    ]
    assert results['counter'] == 2
