import bw2calc as bc
from bw_processing import create_datapackage, INDICES_DTYPE
import numpy as np
from numbers import Number


def compare_dict(one, two):
    assert set(one) == set(two)
    for key, value in one.items():
        if isinstance(value, Number):
            assert np.allclose(value, two[key])
        elif isinstance(value, dict):
            compare_dict(one[key], two[key])
        else:
            assert value == two[key]


def ordered(edges):
    return sorted(edges, key=lambda x: (x['source'], x['target'], x['type']))


def compare_list_of_dicts(one, two):
    for a, b in zip(ordered(one), ordered(two)):
        compare_dict(a, b)


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
    EXPECTED = {
        -1: {'amount': 1, 'direct_score': 0},
        12: {'amount': 1.0, 'direct_score': 0.0},
        10: {'amount': 1.0, 'direct_score': 10.0}
    }
    compare_dict(results['activities'], EXPECTED)
    EXPECTED = [
        {'target': 2, 'source': -1, 'type': 'product', 'amount': 1.0, 'exc_amount': 1.0, 'supply_chain_score': 10.0},
        {'target': 12, 'source': 2, 'type': 'activity', 'amount': 1.0, 'exc_amount': 1.0, 'direct_score': 0.0},
        {'target': 3, 'source': 12, 'type': 'product', 'amount': 1.0, 'exc_amount': -1.0, 'supply_chain_score': 10.0},
        {'target': 10, 'source': 3, 'type': 'activity', 'amount': 1.0, 'exc_amount': 1.0, 'direct_score': 10.0}
    ]
    compare_list_of_dicts(results['edges'], EXPECTED)
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
    compare_dict(results['products'], {1: {'amount': 1.0, 'supply_chain_score': -90.0}, 2: {'amount': -1.0, 'supply_chain_score': -100.0}})
    EXPECTED = {
        -1: {'amount': 1, 'direct_score': 0},
        10: {'amount': 1.0, 'direct_score': 10.0},
        11: {'amount': -1.0, 'direct_score': -100.0}
    }
    compare_dict(results['activities'], EXPECTED)
    EXPECTED = [
        {'target': 1, 'source': -1, 'type': 'product', 'amount': 1.0, 'exc_amount': 1.0, 'supply_chain_score': -90.0},
        {'target': 10, 'source': 1, 'type': 'activity', 'amount': 1.0, 'exc_amount': 1.0, 'direct_score': 10.0},
        {'target': 2, 'source': 10, 'type': 'product', 'amount': -1.0, 'exc_amount': 1.0, 'supply_chain_score': -100.0},
        {'target': 11, 'source': 2, 'type': 'activity', 'amount': -1.0, 'exc_amount': 1.0, 'direct_score': -100.0}
    ]
    compare_list_of_dicts(results['edges'], EXPECTED)
    assert results['counter'] == 2


def test_multifunctional_x_path_two_paths():
    dp = create_datapackage()

    data_array = np.array([10, 100])
    indices_array = np.array([(21, 0), (22, 0)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="c",
        indices_array=indices_array,
    )

    data_array = np.array([1, 1])
    indices_array = np.array([(21, 11), (22, 12)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="b",
        indices_array=indices_array,
    )

    data_array = np.array([1, -1, 1, 1, 0.5, -0.1])
    indices_array = np.array([
        (1, 10),
        (2, 10),
        (2, 11),
        (2, 12),
        (3, 11),
        (3, 12),
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

    assert np.allclose(lca.score, 85)

    results = bc.MultifunctionalGraphTraversal.calculate(lca=lca, max_calc=10)

    assert np.allclose(results['products'][1]['supply_chain_score'], 85)
    assert np.allclose(results['products'][2]['amount'], 1)
    assert np.allclose(results['products'][2]['supply_chain_score'], 85)

    # Hit product 3 twice
    assert results['counter'] == 4

    assert sorted(results['activities']) == [-1, 10, 11, 12]
    assert np.allclose(results['activities'][10]['direct_score'], 0)
    assert np.allclose(results['activities'][11]['direct_score'], 1 / 6 * 10)
    assert np.allclose(results['activities'][11]['amount'], 1 / 6)
    assert np.allclose(results['activities'][12]['direct_score'], 5 / 6 * 100)
    assert np.allclose(results['activities'][12]['amount'], 5 / 6)

    EXPECTED = [
        {'source': -1, 'target': 1, 'type': 'product', 'amount': 1.0, 'exc_amount': 1.0, 'supply_chain_score': 84.99999999999999},
        {'source': 1, 'target': 10, 'type': 'activity', 'amount': 1.0, 'exc_amount': 1.0, 'direct_score': 0.0},
        {'source': 2, 'target': 11, 'type': 'activity', 'amount': 1.0, 'exc_amount': 1.0, 'direct_score': 1.6666666666666665},
        {'source': 2, 'target': 12, 'type': 'activity', 'amount': 1.0, 'exc_amount': 1.0, 'direct_score': 83.33333333333331},
        {'source': 10, 'target': 2, 'type': 'product', 'amount': 1.0, 'exc_amount': -1.0, 'supply_chain_score': 84.99999999999999},
        {'amount': -0.08333333333333333, 'exc_amount': 0.5, 'source': 11, 'supply_chain_score': 12.499999999999995, 'target': 3, 'type': 'product'},
        {'source': 12, 'target': 3, 'type': 'product', 'amount': 0.08333333333333331, 'exc_amount': -0.1, 'supply_chain_score': -12.499999999999996}
    ]
    compare_list_of_dicts(results['edges'], EXPECTED)


def test_multifunctional_scaling():
    dp = create_datapackage()

    data_array = np.array([10, 100])
    indices_array = np.array([(20, 0), (21, 0)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="c",
        indices_array=indices_array,
    )

    data_array = np.array([1, 1, 1])
    indices_array = np.array([(20, 10), (21, 11), (20, 12)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="b",
        indices_array=indices_array,
    )

    data_array = np.array([2, 4, 6, -2, 10])
    indices_array = np.array([
        (1, 10),
        (2, 10),
        (2, 11),
        (3, 10),
        (3, 12),
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

    assert np.allclose(lca.score, 1 + 5 - 100 / 3)

    results = bc.MultifunctionalGraphTraversal.calculate(lca=lca)
    EXPECTED = {
        1: {'amount': 1.0, 'supply_chain_score': -27.33333333333333},
        2: {'amount': -2.0, 'supply_chain_score': -33.33333333333333},
        3: {'amount': 1.0, 'supply_chain_score': 1.0}
    }
    compare_dict(results['products'], EXPECTED)
    EXPECTED = {
        -1: {'amount': 1, 'direct_score': 0},
        10: {'amount': 0.5, 'direct_score': 5.0},
        11: {'amount': -0.3333333333333333, 'direct_score': -33.33333333333333},
        12: {'amount': 0.1, 'direct_score': 1.0}
    }
    compare_dict(results['activities'], EXPECTED)
    EXPECTED = [
        {'source': -1, 'target': 1, 'type': 'product', 'amount': 1.0, 'exc_amount': 1.0, 'supply_chain_score': -27.33333333333333},
        {'source': 1, 'target': 10, 'type': 'activity', 'amount': 1.0, 'exc_amount': 2.0, 'direct_score': 5.0},
        {'source': 2, 'target': 11, 'type': 'activity', 'amount': -2.0, 'exc_amount': 6.0, 'direct_score': -33.33333333333333},
        {'source': 3, 'target': 12, 'type': 'activity', 'amount': 1.0, 'exc_amount': 10.0, 'direct_score': 1.0},
        {'source': 10, 'target': 2, 'type': 'product', 'amount': -2.0, 'exc_amount': 4.0, 'supply_chain_score': -33.33333333333333},
        {'source': 10, 'target': 3, 'type': 'product', 'amount': 1.0, 'exc_amount': -2.0, 'supply_chain_score': 1.0}
    ]
    compare_list_of_dicts(results['edges'], EXPECTED)
    assert results['counter'] == 3


# This was an attempt to test forcing multiple paths to see if there could be possible double counting
# if we would have to go up and down the same edge. But I can't get a system designed where
# I can force this successfully.
def test_multifunctional_multiple_paths():
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
    indices_array = np.array([(20, 11), (21, 12)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="b",
        indices_array=indices_array,
    )

    data_array = np.array([-1, -10, -3, 1, 1, -2, 1, -1, -1, -4, 1, 3, -2])
    indices_array = np.array([
        (1, 14),
        (2, 14),
        (3, 14),
        (1, 10),
        (1, 13),
        (3, 10),
        (3, 11),
        (5, 11),
        (5, 12),
        (2, 13),
        (2, 12),
        (4, 11),
        (4, 12),
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

    print(lca.supply_array)
    print(lca.technosphere_matrix.toarray())
    print(lca.characterized_inventory.toarray())
