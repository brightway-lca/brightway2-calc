"""Characterization factors are filtered to the global location of each resource group, with
that group's own ``global_index``: packages processed in projects in which the global location
has different ids can be used together."""

from functools import partial

import bw_processing as bwp
import matrix_utils as mu
import numpy as np
import pytest
from bw_processing.proxies import Proxy
from fsspec.implementations.zip import ZipFileSystem

from bw2calc import LCA, MultiLCA
from bw2calc.errors import UnknownIndicesArray
from bw2calc.method_config import MethodConfig
from bw2calc.utils import GlobalIndexFilter, global_index_filter


def inventory():
    """Product 1 from activity 101, which emits 1 of flow 201 and 2 of flow 202."""
    dp = bwp.create_datapackage()
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=np.array([1.0]),
        name="technosphere",
        indices_array=np.array([(1, 101)], dtype=bwp.INDICES_DTYPE),
    )
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=np.array([1.0, 2.0]),
        name="biosphere",
        indices_array=np.array([(201, 101), (202, 101)], dtype=bwp.INDICES_DTYPE),
    )
    return dp


def method(rows, global_index, name="method", identifier=None, fs=None):
    """``rows`` are ``(flow, location id, amount)``."""
    dp = bwp.create_datapackage(name=name, fs=fs)
    kwargs = {"identifier": identifier} if identifier is not None else {}
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=np.array([amount for _, _, amount in rows], dtype=float),
        name=name,
        indices_array=np.array(
            [(flow, location) for flow, location, _ in rows], dtype=bwp.INDICES_DTYPE
        ),
        global_index=global_index,
        **kwargs,
    )
    return dp


def test_each_package_with_its_own_global_index():
    # Location 7 is global in the first package, and a region in the second: its factor there
    # must not be used
    first = method([(201, 7, 10)], global_index=7, name="first")
    second = method([(202, 3, 100), (202, 7, 1000)], global_index=3, name="second")
    lca = LCA({1: 1}, data_objs=[inventory(), first, second])
    lca.lci()
    lca.lcia()
    assert lca.score == 1 * 10 + 2 * 100


def test_groups_without_a_global_index_are_not_filtered():
    first = method([(201, 7, 10), (201, 8, 1)], global_index=7, name="first")
    second = method([(202, 3, 100)], global_index=None, name="second")
    lca = LCA({1: 1}, data_objs=[inventory(), first, second])
    lca.lci()
    lca.lcia()
    assert lca.score == 10 + 200


def test_packages_loaded_with_proxies(tmp_path):
    for name, rows, global_index in (
        ("first", [(201, 7, 10), (201, 3, 5)], 7),
        ("second", [(202, 3, 100), (202, 7, 1000)], 3),
    ):
        fs = ZipFileSystem(tmp_path / f"{name}.zip", mode="w")
        method(rows, global_index, name, fs=fs).finalize_serialization()
    first, second = (
        bwp.load_datapackage(ZipFileSystem(tmp_path / f"{name}.zip"), proxy=True)
        for name in ("first", "second")
    )
    lca = LCA({1: 1}, data_objs=[inventory(), first, second])
    lca.lci()
    lca.lcia()
    assert lca.score == 10 + 200


def test_multi_lca_with_different_global_indices():
    first = method([(201, 7, 10), (202, 3, 1)], 7, "first", identifier=["first"])
    second = method([(202, 3, 100), (202, 7, 1000)], 3, "second", identifier=["second"])
    config = MethodConfig(impact_categories=[("first",), ("second",)])
    mlca = MultiLCA(
        demands={"fu": {1: 1}},
        method_config=config,
        data_objs=[inventory(), first, second],
    )
    mlca.lci()
    mlca.lcia()
    assert mlca.scores[(("first",), "fu")] == 10
    assert mlca.scores[(("second",), "fu")] == 200


def proxied(tmp_path, name, rows, global_index, **kwargs):
    fs = ZipFileSystem(tmp_path / f"{name}.zip", mode="w")
    method(rows, global_index, name, fs=fs, **kwargs).finalize_serialization()
    return bwp.load_datapackage(ZipFileSystem(tmp_path / f"{name}.zip"), proxy=True)


def indices_of(dp, name):
    return dp.get_resource(f"{name}.indices")[0]


def score(*methods):
    lca = LCA({1: 1}, data_objs=[inventory(), *methods])
    lca.lci()
    lca.lcia()
    return lca.score


##############
# The filter #
##############


def test_filter_with_global_index():
    first = method([(201, 7, 10), (201, 3, 5)], 7, "first")
    fltr = GlobalIndexFilter()
    fltr.register(first)
    mask = fltr(indices_of(first, "first"))
    assert mask.dtype == bool
    assert mask.tolist() == [True, False]


def test_filter_group_without_global_index_is_registered_and_not_filtered():
    second = method([(202, 3, 100), (202, 4, 1)], None, "second")
    fltr = GlobalIndexFilter()
    fltr.register(second)
    indices = indices_of(second, "second")
    assert fltr.by_array == {id(indices): None}
    mask = fltr(indices)
    assert mask.dtype == bool
    assert mask.tolist() == [True, True]


def test_filter_global_index_zero():
    # Zero is a valid global index, and not the same as no global index
    first = method([(201, 0, 10), (201, 3, 5)], 0, "first")
    fltr = global_index_filter([first])
    assert fltr(indices_of(first, "first")).tolist() == [True, False]


def test_filter_empty_indices():
    first = method([], 7, "first")
    second = method([], None, "second")
    fltr = global_index_filter([first, second])
    assert fltr(indices_of(first, "first")).tolist() == []
    assert fltr(indices_of(second, "second")).tolist() == []


def test_filter_only_registers_indices_of_its_matrix():
    dp = inventory()
    fltr = global_index_filter([dp])
    assert not fltr.by_array
    assert not fltr.arrays
    with pytest.raises(UnknownIndicesArray):
        fltr(indices_of(dp, "biosphere"))

    fltr = global_index_filter([dp], matrix="biosphere_matrix")
    assert fltr(indices_of(dp, "biosphere")).tolist() == [True, True]
    with pytest.raises(UnknownIndicesArray):
        fltr(indices_of(dp, "technosphere"))


def test_filter_several_groups_in_one_package():
    dp = bwp.create_datapackage()
    for name, rows, global_index in (
        ("a", [(201, 7), (201, 3)], 7),
        ("b", [(202, 7), (202, 3)], 3),
        ("c", [(202, 7), (202, 3)], None),
    ):
        dp.add_persistent_vector(
            matrix="characterization_matrix",
            data_array=np.ones(len(rows)),
            name=name,
            indices_array=np.array(rows, dtype=bwp.INDICES_DTYPE),
            global_index=global_index,
        )
    fltr = global_index_filter([dp])
    assert fltr(indices_of(dp, "a")).tolist() == [True, False]
    assert fltr(indices_of(dp, "b")).tolist() == [False, True]
    assert fltr(indices_of(dp, "c")).tolist() == [True, True]


def test_filter_keeps_arrays_alive():
    first = method([(201, 7, 10)], 7, "first")
    second = method([(202, 3, 100)], None, "second")
    fltr = global_index_filter([first, second])
    assert [id(array) for array in fltr.arrays] == list(fltr.by_array)
    assert fltr.arrays[0] is indices_of(first, "first")
    assert fltr.arrays[1] is indices_of(second, "second")


def test_filter_registering_twice():
    first = method([(201, 7, 10), (201, 3, 5)], 7, "first")
    fltr = GlobalIndexFilter()
    fltr.register(first)
    fltr.register(first)
    fltr.register(first.filter_by_attribute("matrix", "characterization_matrix"))
    assert len(fltr.arrays) == len(fltr.by_array) == 1
    assert fltr(indices_of(first, "first")).tolist() == [True, False]


##################
# Unknown arrays #
##################


def test_unknown_array_raises_instead_of_not_filtering():
    first = method([(201, 7, 10), (201, 3, 5)], 7, "first")
    other = method([(201, 7, 10), (201, 3, 5)], 7, "other")
    fltr = global_index_filter([first])
    indices = indices_of(first, "first")
    assert fltr(indices).tolist() == [True, False]

    # Equal values are not enough: these are other objects
    for unknown in (indices.copy(), indices[:], indices[:1], indices_of(other, "other")):
        with pytest.raises(UnknownIndicesArray):
            fltr(unknown)


def test_unknown_array_error_message():
    first = method([(201, 7, 10), (201, 3, 5)], 7, "first")
    fltr = GlobalIndexFilter()
    with pytest.raises(UnknownIndicesArray, match="characterization_matrix.*2 rows"):
        fltr(indices_of(first, "first"))


def test_unknown_array_raises_with_no_registered_packages():
    # Nothing registered is not the same as nothing to filter
    fltr = global_index_filter([])
    with pytest.raises(UnknownIndicesArray):
        fltr(np.array([(201, 7)], dtype=bwp.INDICES_DTYPE))


def test_unregistered_package_raises_in_matrix_utils():
    first = method([(201, 7, 10), (201, 3, 5)], 7, "first")
    second = method([(202, 3, 100), (202, 7, 1000)], 3, "second")
    fltr = global_index_filter([first])
    with pytest.raises(UnknownIndicesArray):
        mu.MappedMatrix(
            packages=[first, second],
            matrix="characterization_matrix",
            diagonal=True,
            custom_filter=fltr,
        )


def test_transposed_indices_raise():
    # ``matrix_utils`` copies the indices to transpose them
    first = method([(201, 7, 10), (201, 3, 5)], 7, "first")
    fltr = GlobalIndexFilter()
    with pytest.raises(UnknownIndicesArray):
        mu.MappedMatrix(
            packages=fltr.through([first]),
            matrix="characterization_matrix",
            transpose=True,
            custom_filter=fltr,
        )


class Copying:
    """Gives a new array object each time, also from filtered packages."""

    def get_resource(self, name_or_index):
        data, metadata = bwp.DatapackageBase.get_resource(self, name_or_index)
        return (data.copy() if isinstance(data, np.ndarray) else data), metadata

    def filter_by_attribute(self, key, value):
        filtered = bwp.DatapackageBase.filter_by_attribute(self, key, value)
        copying = CopyingFilteredDatapackage()
        copying.__dict__.update(filtered.__dict__)
        return copying


class CopyingDatapackage(Copying, bwp.Datapackage):
    pass


class CopyingFilteredDatapackage(Copying, bwp.datapackage.FilteredDatapackage):
    pass


def copying(dp):
    obj = CopyingDatapackage()
    obj.__dict__.update(dp.__dict__)
    return obj


def test_lca_raises_if_arrays_are_copied():
    # Without the error, the regional factor (5) would be added to the global one (10)
    first = copying(method([(201, 7, 10), (201, 3, 5)], 7, "first"))
    lca = LCA({1: 1}, data_objs=[inventory(), first])
    lca.lci()
    with pytest.raises(UnknownIndicesArray):
        lca.lcia()


def test_multi_lca_raises_if_arrays_are_copied():
    first = copying(method([(201, 7, 10), (201, 3, 5)], 7, "first", identifier=["first"]))
    mlca = MultiLCA(
        demands={"fu": {1: 1}},
        method_config=MethodConfig(impact_categories=[("first",)]),
        data_objs=[inventory(), first],
    )
    mlca.lci()
    with pytest.raises(UnknownIndicesArray):
        mlca.lcia()


###########
# through #
###########


def test_through_yields_the_same_packages_and_registers_lazily():
    first = method([(201, 7, 10), (201, 3, 5)], 7, "first")
    second = method([(202, 3, 100), (202, 7, 1000)], 3, "second")
    fltr = GlobalIndexFilter()
    packages = fltr.through([first, second])
    assert not fltr.by_array

    assert next(packages) is first
    assert fltr(indices_of(first, "first")).tolist() == [True, False]
    with pytest.raises(UnknownIndicesArray):
        fltr(indices_of(second, "second"))

    assert next(packages) is second
    assert fltr(indices_of(second, "second")).tolist() == [True, False]
    with pytest.raises(StopIteration):
        next(packages)


def test_through_generator_in_matrix_utils():
    first = method([(201, 7, 10), (201, 3, 5)], 7, "first")
    second = method([(202, 3, 100), (202, 7, 1000)], 3, "second")
    fltr = GlobalIndexFilter()
    mm = mu.MappedMatrix(
        packages=fltr.through(dp for dp in (first, second)),
        matrix="characterization_matrix",
        diagonal=True,
        custom_filter=fltr,
    )
    assert mm.matrix.diagonal().tolist() == [10, 100]
    # Every group was filtered with its own index, by the array that the group has
    assert len(mm.groups) == 2
    for group, expected in zip(mm.groups, ([True, False], [True, False])):
        assert id(group.get_indices_data()) in fltr.by_array
        assert group.custom_filter_mask.tolist() == expected


def test_through_proxies_are_loaded_once(tmp_path):
    first = proxied(tmp_path, "first", [(201, 7, 10), (201, 3, 5)], 7)
    assert isinstance(first.data[0], (Proxy, partial))
    fltr = GlobalIndexFilter()
    mm = mu.MappedMatrix(
        packages=fltr.through([first]),
        matrix="characterization_matrix",
        diagonal=True,
        custom_filter=fltr,
    )
    assert mm.matrix.diagonal().tolist() == [10]
    # The array of the group is the one which was loaded in the package given
    assert mm.groups[0].get_indices_data() is indices_of(first, "first")
    assert mm.groups[0].get_indices_data() is fltr.arrays[0]


def test_register_package_filtered_by_identifier(tmp_path):
    # As in ``MultiLCA``: the package given to ``matrix_utils`` is already filtered
    dp = proxied(tmp_path, "first", [(201, 7, 10), (201, 3, 5)], 7, identifier=["first"])
    filtered = dp.filter_by_attribute("identifier", ["first"])
    fltr = GlobalIndexFilter()
    mm = mu.MappedMatrix(
        packages=fltr.through([filtered]),
        matrix="characterization_matrix",
        diagonal=True,
        custom_filter=fltr,
    )
    assert mm.matrix.diagonal().tolist() == [10]


#######
# LCA #
#######


def test_lca_data_objs_generator():
    # Iterated only once: a generator used to be exhausted before the matrix was built
    first = method([(201, 7, 10), (201, 3, 5)], global_index=7, name="first")
    second = method([(202, 3, 100), (202, 7, 1000)], global_index=3, name="second")
    lca = LCA({1: 1}, data_objs=[inventory()])
    lca.lci()
    lca.load_lcia_data(data_objs=(dp for dp in (first, second)))
    lca.lcia_calculation()
    assert lca.score == 10 + 200


def test_lca_data_objs_generator_without_characterization_data():
    lca = LCA({1: 1}, data_objs=[inventory()])
    lca.lci()
    with pytest.raises(ValueError, match="no characterization data"):
        lca.load_lcia_data(data_objs=(dp for dp in (inventory(),)))


def test_lca_all_groups_without_global_index():
    first = method([(201, 7, 10), (201, 8, 1)], global_index=None, name="first")
    second = method([(202, 3, 100)], global_index=None, name="second")
    assert score(first, second) == 11 + 200


def test_lca_global_index_zero():
    first = method([(201, 0, 10), (201, 8, 1)], global_index=0, name="first")
    second = method([(202, 3, 100), (202, 0, 1000)], global_index=3, name="second")
    assert score(first, second) == 10 + 200


def test_lca_same_flow_in_packages_with_different_global_indices():
    first = method([(201, 7, 10), (201, 3, 5)], global_index=7, name="first")
    second = method([(201, 3, 100), (201, 7, 1000)], global_index=3, name="second")
    # The global factor of a later package replaces that of an earlier one
    assert score(first, second) == 100
    assert score(second, first) == 10


def test_lca_no_global_rows_in_a_group():
    first = method([(201, 7, 10)], global_index=7, name="first")
    second = method([(202, 4, 100)], global_index=3, name="second")
    assert score(first, second) == 10


def test_lca_switch_method():
    first = method([(201, 7, 10), (201, 3, 5)], global_index=7, name="first")
    second = method([(202, 3, 100), (202, 7, 1000)], global_index=3, name="second")
    lca = LCA({1: 1}, data_objs=[inventory(), first])
    lca.lci()
    lca.lcia()
    assert lca.score == 10
    lca.switch_method([second])
    lca.lcia_calculation()
    assert lca.score == 200
    # Loaded again with the same packages
    lca.switch_method([second])
    lca.lcia_calculation()
    assert lca.score == 200


def test_lca_iteration_keeps_filter():
    first = method([(201, 7, 10), (201, 3, 5)], global_index=7, name="first")
    second = method([(202, 3, 100), (202, 7, 1000)], global_index=3, name="second")
    lca = LCA({1: 1}, data_objs=[inventory(), first, second])
    lca.lci()
    lca.lcia()
    for _ in range(3):
        next(lca)
        assert lca.score == 10 + 200


def test_lca_use_distributions():
    dp = bwp.create_datapackage(name="uncertain")
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=np.array([10.0, 5.0]),
        name="uncertain",
        indices_array=np.array([(201, 7), (201, 3)], dtype=bwp.INDICES_DTYPE),
        distributions_array=np.array(
            [(0, 10, np.nan, np.nan, np.nan, np.nan, False)] * 2, dtype=bwp.UNCERTAINTY_DTYPE
        ),
        global_index=7,
    )
    second = method([(202, 3, 100), (202, 7, 1000)], global_index=3, name="second")
    lca = LCA(
        {1: 1}, data_objs=[inventory(), dp, second], calculation_config={"use_distributions": True}
    )
    lca.lci()
    lca.lcia()
    assert lca.score == 10 + 200


def test_lca_array_resources():
    dp = bwp.create_datapackage(name="arrays", sequential=True)
    dp.add_persistent_array(
        matrix="characterization_matrix",
        data_array=np.array([[10.0, 20.0], [5.0, 6.0]]),
        name="arrays",
        indices_array=np.array([(201, 7), (201, 3)], dtype=bwp.INDICES_DTYPE),
        global_index=7,
    )
    second = method([(202, 3, 100), (202, 7, 1000)], global_index=3, name="second")
    lca = LCA({1: 1}, data_objs=[inventory(), dp, second], calculation_config={"use_arrays": True})
    lca.lci()
    lca.lcia()
    assert lca.score == 10 + 200
    next(lca)
    assert lca.score == 20 + 200


def test_lca_mixed_proxy_and_memory_packages(tmp_path):
    first = proxied(tmp_path, "first", [(201, 7, 10), (201, 3, 5)], 7)
    second = method([(202, 3, 100), (202, 7, 1000)], global_index=3, name="second")
    third = proxied(tmp_path, "third", [(202, 9, 1), (202, 4, 50000)], None)
    # Not filtered, and its sum replaces the factor of the second package
    assert score(first, second, third) == 10 + 2 * 50001
    assert score(third, first, second) == 10 + 200


############
# MultiLCA #
############


def multi(data_objs, categories=("first", "second")):
    mlca = MultiLCA(
        demands={"fu": {1: 1}},
        method_config=MethodConfig(impact_categories=[(label,) for label in categories]),
        data_objs=data_objs,
    )
    mlca.lci()
    return mlca


def test_multi_lca_groups_without_a_global_index():
    first = method([(201, 7, 10), (201, 8, 1)], 7, "first", identifier=["first"])
    second = method([(202, 3, 100), (202, 7, 1000)], None, "second", identifier=["second"])
    mlca = multi([inventory(), first, second])
    mlca.lcia()
    assert mlca.scores[(("first",), "fu")] == 10
    assert mlca.scores[(("second",), "fu")] == 2 * 1100


def test_multi_lca_categories_in_one_package():
    dp = bwp.create_datapackage()
    for name, rows, global_index in (
        ("first", [(201, 7, 10), (201, 3, 5)], 7),
        ("second", [(202, 3, 100), (202, 7, 1000)], 3),
    ):
        dp.add_persistent_vector(
            matrix="characterization_matrix",
            data_array=np.array([amount for _, _, amount in rows], dtype=float),
            name=name,
            indices_array=np.array([row[:2] for row in rows], dtype=bwp.INDICES_DTYPE),
            global_index=global_index,
            identifier=[name],
        )
    mlca = multi([inventory(), dp])
    mlca.lcia()
    assert mlca.scores[(("first",), "fu")] == 10
    assert mlca.scores[(("second",), "fu")] == 200


def test_multi_lca_packages_loaded_with_proxies(tmp_path):
    first = proxied(tmp_path, "first", [(201, 7, 10), (201, 3, 5)], 7, identifier=["first"])
    second = proxied(tmp_path, "second", [(202, 3, 100), (202, 7, 1000)], 3, identifier=["second"])
    mlca = multi([inventory(), first, second])
    mlca.lcia()
    assert mlca.scores[(("first",), "fu")] == 10
    assert mlca.scores[(("second",), "fu")] == 200


def test_multi_lca_only_loads_indices_of_selected_categories(tmp_path):
    first = proxied(tmp_path, "first", [(201, 7, 10), (201, 3, 5)], 7, identifier=["first"])
    unused = proxied(tmp_path, "unused", [(202, 3, 100)], 3, identifier=["unused"])
    mlca = multi([inventory(), first, unused], categories=("first",))
    mlca.lcia()
    assert mlca.scores[(("first",), "fu")] == 10
    assert all(isinstance(obj, (Proxy, partial)) for obj in unused.data)


def test_multi_lca_data_objs_generator():
    # Iterated once per impact category
    first = method([(201, 7, 10), (202, 3, 1)], 7, "first", identifier=["first"])
    second = method([(202, 3, 100), (202, 7, 1000)], 3, "second", identifier=["second"])
    mlca = multi([inventory(), first, second])
    mlca.load_lcia_data(data_objs=(dp for dp in (first, second)))
    mlca.lcia_calculation()
    assert mlca.scores[(("first",), "fu")] == 10
    assert mlca.scores[(("second",), "fu")] == 200


def test_multi_lca_iteration_keeps_filter():
    first = method([(201, 7, 10), (202, 3, 1)], 7, "first", identifier=["first"])
    second = method([(202, 3, 100), (202, 7, 1000)], 3, "second", identifier=["second"])
    mlca = multi([inventory(), first, second])
    mlca.lcia()
    for _ in range(3):
        next(mlca)
        assert mlca.scores[(("first",), "fu")] == 10
        assert mlca.scores[(("second",), "fu")] == 200
