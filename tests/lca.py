from bw2calc.errors import (
    OutsideTechnosphere,
    NonsquareTechnosphere,
    EmptyBiosphere,
    InconsistentGlobalIndex,
)
from bw2calc.lca import LCA
from pathlib import Path
import bw_processing as bwp
import json
import numpy as np
import pytest
from collections.abc import Mapping

fixture_dir = Path(__file__).resolve().parent / "fixtures"


######
### Basic functionality
######


def test_example_db_basic():
    mapping = dict(json.load(open(fixture_dir / "bw2io_example_db_mapping.json")))
    print(mapping)
    packages = [
        fixture_dir / "bw2io_example_db.zip",
        fixture_dir / "ipcc_simple.zip",
    ]

    lca = LCA(
        {mapping["Driving an electric car"]: 1},
        data_objs=packages,
    )
    lca.lci()
    lca.lcia()
    assert lca.supply_array.sum()
    assert lca.technosphere_matrix.sum()
    assert lca.score


def test_basic():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    answer = np.zeros((2,))
    answer[lca.dicts.activity[101]] = 1
    answer[lca.dicts.activity[102]] = 0.5
    assert np.allclose(answer, lca.supply_array)


def test_basic_negative_production():
    pass


def test_basic_substitution():
    pass


def test_basic_nonunitary_production():
    pass


def test_circular_inputs():
    pass


######
### __init__
######


def test_invalid_datapackage():
    packages = ["basic_fixture.zip"]
    with pytest.raises(TypeError):
        LCA({1: 1}, data_objs=packages)


def test_demand_not_mapping():
    packages = [fixture_dir / "basic_fixture.zip"]
    with pytest.raises(ValueError):
        LCA((1, 1), data_objs=packages)


def test_demand_mapping_but_not_dict():
    class M(Mapping):
        def __getitem__(self, key):
            return 1

        def __iter__(self):
            return iter((1,))

        def __len__(self):
            return 1

    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA(M(), data_objs=packages)
    lca.lci()
    answer = np.zeros((2,))
    answer[lca.dicts.activity[101]] = 1
    answer[lca.dicts.activity[102]] = 0.5
    assert np.allclose(answer, lca.supply_array)


######
### __next__
######


def test_next_data_array():
    packages = [fixture_dir / "array_sequential.zip"]
    lca = LCA({1: 1}, data_objs=packages, use_arrays=True)
    lca.lci()
    lca.lcia()

    for x in range(1, 5):
        assert lca.biosphere_matrix.sum() == x
        next(lca)


def test_data_array_keep_first():
    packages = [fixture_dir / "array_sequential.zip"]
    lca = LCA({1: 1}, data_objs=packages, use_arrays=True)
    lca.lci()
    lca.lcia()
    lca.keep_first_iteration()

    for x in range(1, 5):
        next(lca)
        assert lca.biosphere_matrix.sum() == x


def test_next_only_vectors():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()
    current = lca.characterized_inventory.sum()

    next(lca)
    assert lca.characterized_inventory.sum() == current


def test_next_plain_monte_carlo():
    packages = [
        fixture_dir / "mc_basic.zip",
    ]
    mc = LCA({3: 1}, data_objs=packages, use_distributions=True)
    mc.lci()
    mc.lcia()
    first = mc.score
    next(mc)
    assert first != mc.score


def test_next_monte_carlo_as_iterator():
    packages = [
        fixture_dir / "mc_basic.zip",
    ]
    mc = LCA({3: 1}, data_objs=packages, use_distributions=True)
    mc.lci()
    mc.lcia()
    for _, _ in zip(mc, range(10)):
        assert mc.score > 0


def test_next_monte_carlo_all_matrices_change():
    packages = [
        fixture_dir / "mc_basic.zip",
    ]
    mc = LCA({3: 1}, data_objs=packages, use_distributions=True)
    mc.lci()
    mc.lcia()
    a = [
        mc.technosphere_matrix.sum(),
        mc.biosphere_matrix.sum(),
        mc.characterization_matrix.sum(),
    ]
    next(mc)
    b = [
        mc.technosphere_matrix.sum(),
        mc.biosphere_matrix.sum(),
        mc.characterization_matrix.sum(),
    ]
    print(a, b)
    for x, y in zip(a, b):
        assert x != y


######
### build_demand_array
######


def test_build_demand_array():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()

    assert lca.demand_array.shape == (2,)
    assert lca.demand_array.sum() == 1
    assert lca.demand_array[lca.dicts.product[1]] == 1


def test_build_demand_array_pass_dict():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()

    lca.build_demand_array({2: 5})

    assert lca.demand_array.shape == (2,)
    assert lca.demand_array.sum() == 5
    assert lca.demand_array[lca.dicts.product[2]] == 5


def test_build_demand_array_outside_technosphere():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({100: 1}, data_objs=packages)
    with pytest.raises(OutsideTechnosphere):
        lca.lci()


def test_build_demand_array_activity_not_product():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({101: 1}, data_objs=packages)
    with pytest.raises(ValueError):
        lca.lci()


def test_build_demand_array_pass_object():
    packages = [fixture_dir / "basic_fixture.zip"]

    class Foo:
        pass

    obj = Foo()
    with pytest.raises(ValueError):
        LCA(obj, data_objs=packages)


######
### load_lci_data
######


def test_load_lci_data():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    tm = np.array([[1, 0], [-0.5, 1]])
    assert np.allclose(lca.technosphere_matrix.toarray(), tm)
    assert lca.dicts.product[1] == 0
    assert lca.dicts.product[2] == 1
    assert lca.dicts.activity[101] == 0
    assert lca.dicts.activity[102] == 1
    assert lca.dicts.biosphere[1] == 0


def test_load_lci_data_nonsquare_technosphere():
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5, 2, 3])
    indices_array = np.array(
        [(1, 101), (2, 102), (2, 101), (3, 101), (3, 102)], dtype=bwp.INDICES_DTYPE
    )
    flip_array = np.array([0, 0, 1, 1, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    with pytest.raises(NonsquareTechnosphere):
        lca.lci()
    # lca.lci()
    # tm = np.array([
    #     [1, 0],
    #     [-0.5, 1],
    #     [-2, -3]
    # ])
    # assert np.allclose(lca.technosphere_matrix.toarray(), tm)
    # assert lca.dicts.product[1] == 0
    # assert lca.dicts.product[2] == 1
    # assert lca.dicts.product[3] == 2
    # assert lca.dicts.activity[101] == 0
    # assert lca.dicts.activity[102] == 1


def test_load_lci_data_empty_biosphere_warning():
    lca = LCA({1: 1}, data_objs=[fixture_dir / "empty_biosphere.zip"])
    with pytest.warns(UserWarning):
        lca.lci()


######
### remap_inventory_dicts
######


def test_remap_inventory_dicts():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA(
        {1: 1},
        data_objs=packages,
        remapping_dicts={"product": {1: ("foo", "bar")}, "biosphere": {1: "z"}},
    )
    lca.lci()
    lca.remap_inventory_dicts()
    tm = np.array([[1, 0], [-0.5, 1]])
    assert np.allclose(lca.technosphere_matrix.toarray(), tm)
    assert lca.dicts.product[("foo", "bar")] == 0
    assert lca.dicts.product[2] == 1
    assert lca.dicts.activity[101] == 0
    assert lca.dicts.activity[102] == 1
    assert lca.dicts.biosphere["z"] == 0


######
### load_lcia_data
######


def test_load_lcia_data():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()
    cm = np.array([[1]])
    assert np.allclose(lca.characterization_matrix.toarray(), cm)


def test_load_lcia_data_multiple_characterization_packages():
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2, 3])
    indices_array = np.array([(1, 101), (2, 102), (3, 101)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1])
    indices_array = np.array([(1, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
        nrows=1,
    )
    data_array = np.array([2])
    indices_array = np.array([(3, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="second-characterization",
        indices_array=indices_array,
        global_index=0,
        nrows=1,
    )
    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    cm = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]])
    assert np.allclose(lca.characterization_matrix.toarray(), cm)
    assert lca.dicts.biosphere[1] == 0
    assert lca.dicts.biosphere[2] == 1
    assert lca.dicts.biosphere[3] == 2


def test_load_lcia_data_inconsistent_globals():
    # Activities: 101, 102
    # Products: 1, 2
    # Biosphere flows: 201, 202
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(201, 101), (202, 102)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1])
    indices_array = np.array([(201, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
        nrows=1,
    )

    data_array = np.array([10])
    indices_array = np.array([(202, 1)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="second-characterization",
        indices_array=indices_array,
        global_index=1,
        nrows=1,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    with pytest.raises(InconsistentGlobalIndex):
        lca.lcia()


def test_load_lcia_data_none_global_value():
    # Should include all because no filter
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(201, 101), (202, 102)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1])
    indices_array = np.array([(201, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=None,
        nrows=1,
    )

    data_array = np.array([10])
    indices_array = np.array([(202, 1)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="second-characterization",
        indices_array=indices_array,
        global_index=None,
        nrows=1,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    assert lca.characterization_matrix.sum() == 11


def test_load_lcia_data_nonglobal_filtered():
    # Activities: 101, 102
    # Products: 1, 2
    # Biosphere flows: 201, 202
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(201, 101), (202, 102)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1])
    indices_array = np.array([(201, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
        nrows=1,
    )

    data_array = np.array([10])
    indices_array = np.array([(202, 1)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="second-characterization",
        indices_array=indices_array,
        global_index=0,
        nrows=1,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    assert lca.characterization_matrix.sum() == 1


######
### Warnings on uncommon inputs
######


@pytest.mark.filterwarnings("ignore:no biosphere")
def test_empty_biosphere_lcia():
    lca = LCA({1: 1}, data_objs=[fixture_dir / "empty_biosphere.zip"])
    lca.lci()
    assert lca.biosphere_matrix.shape[0] == 0
    with pytest.raises(EmptyBiosphere):
        lca.lcia()


def test_lca_has():
    mapping = dict(json.load(open(fixture_dir / "bw2io_example_db_mapping.json")))
    packages = [
        fixture_dir / "bw2io_example_db.zip",
        fixture_dir / "ipcc_simple.zip",
    ]

    lca = LCA(
        {mapping["Driving an electric car"]: 1},
        data_objs=packages,
    )
    lca.lci()
    lca.lcia()
    assert lca.has("technosphere")
    assert lca.has("characterization")
    assert not lca.has("foo")


######
### normalize
######


def test_lca_with_normalization():
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(201, 101), (202, 102)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1, 10])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
    )

    data_array = np.array([10, 4])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="normalization_matrix",
        data_array=data_array,
        name="nm",
        indices_array=indices_array,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    assert lca.score == 11
    lca.normalize()
    assert lca.score == 1 * 10 + 10 * 4
    assert lca.normalization_matrix.shape == (2, 2)
    assert lca.normalization_matrix.sum() == 14


######
### weighting
######


def test_lca_with_weighting():
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(201, 101), (202, 102)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1, 10])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
    )

    data_array = np.array([4])
    indices_array = np.array([(0, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="weighting_matrix",
        data_array=data_array,
        name="wm",
        indices_array=indices_array,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    assert lca.score == 11
    lca.weight()
    assert lca.score == 11 * 4
    assert lca.weighting_matrix.shape == (2, 2)
    assert lca.weighting_matrix.sum() == 8


def test_lca_with_weighting_deprecation():
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(201, 101), (202, 102)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1, 10])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
    )

    data_array = np.array([4])
    indices_array = np.array([(0, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="weighting_matrix",
        data_array=data_array,
        name="wm",
        indices_array=indices_array,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    with pytest.deprecated_call():
        lca.weighting()


def test_lca_with_weighting_and_normalization():
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(201, 101), (202, 102)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1, 10])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
    )

    data_array = np.array([10, 4])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="normalization_matrix",
        data_array=data_array,
        name="nm",
        indices_array=indices_array,
    )

    data_array = np.array([8])
    indices_array = np.array([(0, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="weighting_matrix",
        data_array=data_array,
        name="wm",
        indices_array=indices_array,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    assert lca.score == 11
    lca.normalize()
    assert lca.score == (1 * 10 + 10 * 4)
    lca.weight()
    assert lca.score == (1 * 10 + 10 * 4) * 8


######
### switch_method
######


def test_switch_method():
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2, 3])
    indices_array = np.array([(1, 101), (2, 102), (3, 101)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1])
    indices_array = np.array([(1, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    cm = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert np.allclose(lca.characterization_matrix.toarray(), cm)
    assert len(lca.packages) == 1

    ndp = bwp.create_datapackage()

    data_array = np.array([10])
    indices_array = np.array([(3, 0)], dtype=bwp.INDICES_DTYPE)
    ndp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
    )

    lca.switch_method([ndp])
    cm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 10]])
    assert np.allclose(lca.characterization_matrix.toarray(), cm)
    assert len(lca.packages) == 2
    assert not any(
        res["matrix"] == "characterization_matrix" for res in lca.packages[0].resources
    )


######
### switch_normalization
######


def test_switch_normalization():
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(201, 101), (202, 102)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1, 10])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
    )

    data_array = np.array([10, 4])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="normalization_matrix",
        data_array=data_array,
        name="nm",
        indices_array=indices_array,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    assert lca.score == 11
    lca.normalize()
    assert lca.score == 1 * 10 + 10 * 4
    assert lca.normalization_matrix.shape == (2, 2)
    assert lca.normalization_matrix.sum() == 14
    assert len(lca.packages) == 1

    ndp = bwp.create_datapackage()

    data_array = np.array([102, 142])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    ndp.add_persistent_vector(
        matrix="normalization_matrix",
        data_array=data_array,
        name="nm2",
        indices_array=indices_array,
    )

    lca.switch_normalization([ndp])
    assert lca.score == 1 * 10 + 10 * 4
    lca.normalize()
    assert lca.score == 1 * 102 + 10 * 142
    assert lca.normalization_matrix.shape == (2, 2)
    assert lca.normalization_matrix.sum() == 102 + 142
    assert len(lca.packages) == 2
    assert not any(
        res["matrix"] == "normalization_matrix" for res in lca.packages[0].resources
    )


######
### switch_weighting
######


def test_switch_weighting():
    dp = bwp.create_datapackage()

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=bwp.INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(201, 101), (202, 102)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1, 10])
    indices_array = np.array([(201, 0), (202, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="first-characterization",
        indices_array=indices_array,
        global_index=0,
    )

    data_array = np.array([4])
    indices_array = np.array([(0, 0)], dtype=bwp.INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="weighting_matrix",
        data_array=data_array,
        name="wm",
        indices_array=indices_array,
    )

    lca = LCA({1: 1}, data_objs=[dp])
    lca.lci()
    lca.lcia()
    assert lca.score == 11
    lca.weight()
    assert lca.score == 11 * 4
    assert lca.weighting_matrix.shape == (2, 2)
    assert lca.weighting_matrix.sum() == 8

    ndp = bwp.create_datapackage()

    data_array = np.array([42])
    indices_array = np.array([(0, 0)], dtype=bwp.INDICES_DTYPE)
    ndp.add_persistent_vector(
        matrix="weighting_matrix",
        data_array=data_array,
        name="wm2",
        indices_array=indices_array,
    )

    lca.switch_weighting([ndp])
    assert lca.weighting_matrix.shape == (2, 2)
    assert lca.weighting_matrix.sum() == 42 * 2
    assert lca.score == 11 * 4
    assert not any(
        res["matrix"] == "weighting_matrix" for res in lca.packages[0].resources
    )
    lca.weight()
    assert lca.score == 11 * 42


######
### invert_technosphere
######


def test_invert_technosphere():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()

    assert not hasattr(lca, "inverted_technosphere_matrix")
    lca.invert_technosphere_matrix()
    assert hasattr(lca, "inverted_technosphere_matrix")

    itm = lca.inverted_technosphere_matrix
    assert itm.shape == (2, 2)
    expected = np.array([[1, 0], [0.5, 1]])
    assert np.allclose(itm, expected)


######
### redo_lci
######


def test_redo_lci():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()

    ref_sup = lca.supply_array.sum()
    lca.lci({2: 1})
    assert lca.supply_array.sum() != ref_sup


def test_redo_lci_same_fu():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()

    ref_sup = lca.supply_array.sum()
    lca.lci()
    assert lca.supply_array.sum() == ref_sup


def test_redo_lci_deprecated():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()

    ref_sup = lca.supply_array.sum()
    with pytest.deprecated_call():
        lca.redo_lci({2: 1})
    assert lca.supply_array.sum() != ref_sup


def test_redo_lci_fails_if_activity_outside_technosphere():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    with pytest.deprecated_call():
        with pytest.raises(OutsideTechnosphere):
            lca.redo_lci({10: 1})


def test_redo_lci_fails_if_passed_bw2_key_tuple():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()

    with pytest.deprecated_call():
        with pytest.raises(KeyError) as excinfo:
            lca.redo_lci({('foo', 'bar'): 1})
            assert 'make sure to pass the integer id' in str(excinfo)


def test_redo_lci_with_no_new_demand_no_error():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    with pytest.deprecated_call():
        lca.redo_lci()


######
### redo_lcia
######


def test_redo_lcia():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()

    ref = lca.score
    lca.lcia({2: 1})
    assert lca.score != ref


def test_redo_lcia_keyerror_bw2_key():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()

    with pytest.deprecated_call():
        with pytest.raises(KeyError) as excinfo:
            lca.redo_lcia(('foo', 'bar'))
            assert 'make sure to pass the integer id' in str(excinfo)


def test_redo_lcia_outside_technosphere():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()

    with pytest.deprecated_call():
        with pytest.raises(OutsideTechnosphere):
            lca.redo_lcia({10: 1})


def test_redo_lcia_same_fu():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()

    ref = lca.score
    lca.lcia()
    assert lca.score == ref


def test_redo_lcia_deprecated():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    lca.lcia()

    ref = lca.score
    with pytest.deprecated_call():
        lca.redo_lcia({2: 1})
    assert lca.score != ref


######
### has
######


def test_has():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    assert lca.has("technosphere")
    assert lca.has("biosphere")
    assert lca.has("characterization")


#     def test_circular_chains(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {
#                         "amount": 0.5,
#                         "input": ("t", "2"),
#                         "type": "technosphere",
#                         "uncertainty type": 0,
#                     },
#                     {
#                         "amount": 1,
#                         "input": ("biosphere", "1"),
#                         "type": "biosphere",
#                         "uncertainty type": 0,
#                     },
#                 ],
#                 "type": "process",
#                 "unit": "kg",
#             },
#             ("t", "2"): {
#                 "exchanges": [
#                     {
#                         "amount": 0.1,
#                         "input": ("t", "1"),
#                         "type": "technosphere",
#                         "uncertainty type": 0,
#                     }
#                 ],
#                 "type": "process",
#                 "unit": "kg",
#             },
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.register()
#         test_db.write(test_data)
#         lca = LCA({("t", "1"): 1})
#         lca.lci()
#         answer = np.zeros((2,))
#         answer[lca.dicts.activity[("t", "1")]] = 20 / 19.0
#         answer[lca.dicts.activity[("t", "2")]] = 10 / 19.0
#         self.assertTrue(np.allclose(answer, lca.supply_array))

#     def test_only_products(self):
#         test_data = {
#             ("t", "p1"): {"type": "product"},
#             ("t", "p2"): {"type": "product"},
#             ("t", "a1"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "p1"), "type": "production",},
#                     {"amount": 1, "input": ("t", "p2"), "type": "production",},
#                 ]
#             },
#             ("t", "a2"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "p2"), "type": "production",}
#                 ]
#             },
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.register()
#         test_db.write(test_data)
#         lca = LCA({("t", "p1"): 1})
#         lca.lci()
#         answer = np.zeros((2,))
#         answer[lca.dicts.activity[("t", "a1")]] = 1
#         answer[lca.dicts.activity[("t", "a2")]] = -1
#         self.assertTrue(np.allclose(answer, lca.supply_array))

#     def test_activity_product_dict(self):
#         test_data = {
#             ("t", "activity 1"): {"type": "process"},
#             ("t", "activity 2"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "activity 2"), "type": "production",}
#                 ]
#             },
#             ("t", "activity 3"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "product 4"), "type": "production",}
#                 ]
#             },
#             ("t", "product 4"): {"type": "product"},
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.write(test_data)
#         lca = LCA({("t", "activity 1"): 1})
#         lca.lci()
#         self.assertEqual(
#             [("t", "activity 1"), ("t", "activity 2"), ("t", "activity 3")],
#             sorted(lca.dicts.activity),
#         )
#         self.assertEqual(
#             [("t", "activity 1"), ("t", "activity 2"), ("t", "product 4")],
#             sorted(lca.dicts.product),
#         )

#         self.assertEqual(
#             [("t", "activity 1"), ("t", "activity 2"), ("t", "activity 3")],
#             sorted(lca.dicts.activity.reversed.values()),
#         )
#         self.assertEqual(
#             [("t", "activity 1"), ("t", "activity 2"), ("t", "product 4")],
#             sorted(lca.dicts.product.reversed.values()),
#         )

#     def test_process_product_split(self):
#         test_data = {
#             ("t", "p1"): {"type": "product"},
#             ("t", "a1"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "p1"), "type": "production",},
#                     {"amount": 1, "input": ("t", "a1"), "type": "production",},
#                 ]
#             },
#             ("t", "a2"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "p1"), "type": "production",}
#                 ]
#             },
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.write(test_data)
#         lca = LCA({("t", "a1"): 1})
#         lca.lci()
#         answer = np.zeros((2,))
#         answer[lca.dicts.activity[("t", "a1")]] = 1
#         answer[lca.dicts.activity[("t", "a2")]] = -1
#         self.assertTrue(np.allclose(answer, lca.supply_array))

#     def test_activity_as_fu_raises_error(self):
#         test_data = {
#             ("t", "p1"): {"type": "product"},
#             ("t", "a1"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "p1"), "type": "production",},
#                     {"amount": 1, "input": ("t", "a1"), "type": "production",},
#                 ]
#             },
#             ("t", "a2"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "p1"), "type": "production",}
#                 ]
#             },
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.write(test_data)
#         with self.assertRaises(ValueError):
#             lca = LCA({("t", "a2"): 1})
#             lca.lci()


#     def test_multiple_lci_calculations(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {
#                         "amount": 1,
#                         "input": ("biosphere", "1"),
#                         "type": "biosphere",
#                         "uncertainty type": 0,
#                     }
#                 ],
#                 "type": "process",
#                 "unit": "kg",
#             },
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.write(test_data)
#         lca = LCA({test_db.random(): 1})
#         lca.lci()
#         lca.lci()

#     def test_dependent_databases(self):
#         databases["one"] = {"depends": ["two", "three"]}
#         databases["two"] = {"depends": ["four", "five"]}
#         databases["three"] = {"depends": ["four"]}
#         databases["four"] = {"depends": ["six"]}
#         databases["five"] = {"depends": ["two"]}
#         databases["six"] = {"depends": []}
#         lca = LCA({("one", None): 1})
#         self.assertEqual(
#             sorted(lca.database_filepath),
#             sorted(
#                 {
#                     Database(name).filepath_processed()
#                     for name in ("one", "two", "three", "four", "five", "six")
#                 }
#             ),
#         )

#     def test_filepaths_full(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {
#                         "amount": 1,
#                         "input": ("biosphere", "1"),
#                         "type": "biosphere",
#                         "uncertainty type": 0,
#                     }
#                 ],
#                 "type": "process",
#                 "unit": "kg",
#             },
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.write(test_data)
#         method = Method(("M",))
#         method.write([[("biosphere", "1"), 1.0]])
#         normalization = Normalization(("N",))
#         normalization.write([[("biosphere", "1"), 1.0]])
#         weighting = Weighting("W")
#         weighting.write([1])
#         lca = LCA({("t", "1"): 1}, method.name, weighting.name, normalization.name,)
#         self.assertEqual(
#             sorted(Database(x).filepath_processed() for x in ("biosphere", "t")),
#             sorted(lca.database_filepath),
#         )
#         self.assertEqual(lca.method_filepath, [method.filepath_processed()])
#         self.assertEqual(lca.weighting_filepath, [weighting.filepath_processed()])
#         self.assertEqual(
#             lca.normalization_filepath, [normalization.filepath_processed()]
#         )

#     def test_filepaths_empty(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {
#                         "amount": 1,
#                         "input": ("biosphere", "1"),
#                         "type": "biosphere",
#                         "uncertainty type": 0,
#                     }
#                 ],
#                 "type": "process",
#                 "unit": "kg",
#             },
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.write(test_data)
#         lca = LCA({("t", "1"): 1})
#         self.assertEqual(
#             sorted(Database(x).filepath_processed() for x in ("biosphere", "t")),
#             sorted(lca.database_filepath),
#         )
#         self.assertTrue(lca.method_filepath is None)
#         self.assertTrue(lca.normalization_filepath is None)
#         self.assertTrue(lca.weighting_filepath is None)

#     def test_demand_type(self):
#         with self.assertRaises(ValueError):
#             LCA(("foo", "1"))
#         with self.assertRaises(ValueError):
#             LCA("foo")
#         with self.assertRaises(ValueError):
#             LCA([{"foo": "1"}])

#     def test_decomposed_uses_solver(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {
#                         "amount": 0.5,
#                         "input": ("t", "2"),
#                         "type": "technosphere",
#                         "uncertainty type": 0,
#                     },
#                     {
#                         "amount": 1,
#                         "input": ("biosphere", "1"),
#                         "type": "biosphere",
#                         "uncertainty type": 0,
#                     },
#                 ],
#                 "type": "process",
#                 "unit": "kg",
#             },
#             ("t", "2"): {"exchanges": [], "type": "process", "unit": "kg"},
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.register()
#         test_db.write(test_data)
#         lca = LCA({("t", "1"): 1})
#         lca.lci(factorize=True)
#         # Indirect test because no easy way to test a function is called
#         lca.technosphere_matrix = None
#         self.assertEqual(float(lca.solve_linear_system().sum()), 1.5)

#     def test_fix_dictionaries(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {
#                         "amount": 0.5,
#                         "input": ("t", "2"),
#                         "type": "technosphere",
#                         "uncertainty type": 0,
#                     },
#                     {
#                         "amount": 1,
#                         "input": ("biosphere", "1"),
#                         "type": "biosphere",
#                         "uncertainty type": 0,
#                     },
#                 ],
#                 "type": "process",
#                 "unit": "kg",
#             },
#             ("t", "2"): {"exchanges": [], "type": "process", "unit": "kg"},
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.register()
#         test_db.write(test_data)
#         lca = LCA({("t", "1"): 1})
#         lca.lci()

#         supply = lca.supply_array.sum()

#         self.assertTrue(lca._fixed)
#         self.assertFalse(lca.fix_dictionaries())
#         # Second time doesn't do anything
#         self.assertFalse(lca.fix_dictionaries())
#         self.assertTrue(lca._fixed)
#         lca.redo_lci({("t", "1"): 2})
#         self.assertEqual(lca.supply_array.sum(), supply * 2)

#     def test_redo_lci_switches_demand(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {
#                         "amount": 0.5,
#                         "input": ("t", "2"),
#                         "type": "technosphere",
#                         "uncertainty type": 0,
#                     },
#                     {
#                         "amount": 1,
#                         "input": ("biosphere", "1"),
#                         "type": "biosphere",
#                         "uncertainty type": 0,
#                     },
#                 ],
#                 "type": "process",
#                 "unit": "kg",
#             },
#             ("t", "2"): {"exchanges": [], "type": "process", "unit": "kg"},
#         }
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.write(test_data)

#         lca = LCA({("t", "1"): 1})
#         lca.lci()
#         self.assertEqual(lca.demand, {("t", "1"): 1})

#         lca.redo_lci({("t", "1"): 2})
#         self.assertEqual(lca.demand, {("t", "1"): 2})

#     def test_basic_lcia(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "2"), "type": "technosphere",}
#                 ],
#             },
#             ("t", "2"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("biosphere", "1"), "type": "biosphere",}
#                 ],
#             },
#         }
#         method_data = [(("biosphere", "1"), 42)]
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.register()
#         test_db.write(test_data)

#         method = Method(("a method",))
#         method.register()
#         method.write(method_data)

#         lca = LCA({("t", "1"): 1}, ("a method",))
#         lca.lci()
#         lca.lcia()

#         self.assertTrue(np.allclose(42, lca.score))

#     def test_redo_lcia_switches_demand(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "2"), "type": "technosphere",}
#                 ],
#             },
#             ("t", "2"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("biosphere", "1"), "type": "biosphere",}
#                 ],
#             },
#         }
#         method_data = [(("biosphere", "1"), 42)]
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.register()
#         test_db.write(test_data)

#         method = Method(("a method",))
#         method.register()
#         method.write(method_data)

#         lca = LCA({("t", "1"): 1}, ("a method",))
#         lca.lci()
#         lca.lcia()
#         self.assertEqual(lca.demand, {("t", "1"): 1})

#         lca.redo_lcia({("t", "2"): 2})
#         self.assertEqual(lca.demand, {("t", "2"): 2})

#     def test_lcia_regionalized_ignored(self):
#         test_data = {
#             ("t", "1"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("t", "2"), "type": "technosphere",}
#                 ],
#             },
#             ("t", "2"): {
#                 "exchanges": [
#                     {"amount": 1, "input": ("biosphere", "1"), "type": "biosphere",}
#                 ],
#             },
#         }
#         method_data = [
#             (("biosphere", "1"), 21),
#             (("biosphere", "1"), 21, config.global_location),
#             (("biosphere", "1"), 100, "somewhere else"),
#         ]
#         self.add_basic_biosphere()
#         test_db = Database("t")
#         test_db.register()
#         test_db.write(test_data)

#         method = Method(("a method",))
#         method.register()
#         method.write(method_data)

#         lca = LCA({("t", "1"): 1}, ("a method",))
#         lca.lci()
#         lca.lcia()

#         self.assertTrue(np.allclose(42, lca.score))
