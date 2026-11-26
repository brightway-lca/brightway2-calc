"""Characterization factors are filtered to the global location of each resource group, with
that group's own ``global_index``: packages processed in projects in which the global location
has different ids can be used together."""

import bw_processing as bwp
import numpy as np
from fsspec.implementations.zip import ZipFileSystem

from bw2calc import LCA, MultiLCA
from bw2calc.method_config import MethodConfig
from bw2calc.utils import global_index_filter


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


def test_filter():
    assert global_index_filter([inventory()]) is None
    first = method([(201, 7, 10), (201, 3, 5)], 7, "first")
    second = method([(202, 3, 100)], None, "second")
    fltr = global_index_filter([first, second])
    (indices, _), (others, _) = (
        dp.get_resource(f"{name}.indices") for dp, name in ((first, "first"), (second, "second"))
    )
    assert fltr(indices).tolist() == [True, False]
    # Not filtered: no global index, or an array of no package
    assert fltr(others).tolist() == [True]
    assert fltr(indices.copy()).tolist() == [True, True]
