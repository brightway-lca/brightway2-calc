from bw2calc.dictionary_manager import (
    DictionaryManager as DM,
    ReversibleRemappableDictionary as RRD,
)
import pytest


def test_dm_initiation():
    DM()


def test_dm_setting():
    dm = DM()
    dm.foo = {1: 2}


def test_dm_getting():
    dm = DM()
    with pytest.raises(ValueError):
        dm.foo[1]
    dm.foo = {1: 2}
    assert dm.foo[1] == 2
    with pytest.raises(KeyError):
        dm.foo[2]


def test_dm_str():
    dm = DM()
    str(dm)
    dm.foo = {1: 2}
    str(dm)


def test_dm_iter():
    dm = DM()
    assert list(dm) == []
    dm.foo = {1: 2}
    assert list(dm) == ["foo"]


def test_dm_len():
    dm = DM()
    assert len(dm) == 0
    dm.foo = {1: 2}
    assert len(dm) == 1


def test_rrd_input_error():
    with pytest.raises(ValueError):
        RRD(1)


def test_rrd_basic():
    r = RRD({1: 2})
    assert r[1] == 2
    with pytest.raises(KeyError):
        r[2]


def test_rrd_reversed():
    r = RRD({1: 2})
    assert r.reversed == {2: 1}
    r = RRD({1: 10, 2: 10})
    assert r.reversed == {10: 2}


def test_rrd_reversed_create_on_demand():
    r = RRD({1: 2})
    assert not hasattr(r, "_reversed")
    r.reversed
    assert hasattr(r, "_reversed")


def test_rrd_remapping_multiple():
    r = RRD({1: 2})
    r.remap({1: "foo"})
    r.remap({"foo": "bar"})
    assert r["bar"] == 2


def test_rrd_remapping():
    r = RRD({1: 2})
    r.remap({1: "foo"})
    assert r["foo"] == 2


def test_rrd_remapping_deletes_reversed():
    r = RRD({1: 2})
    r.reversed
    assert hasattr(r, "_reversed")
    r.remap({1: "foo"})
    assert not hasattr(r, "_reversed")


def test_rrd_remapping_sets_original():
    r = RRD({1: 2})
    r.remap({1: "foo"})
    assert r.original == {1: 2}
    assert hasattr(r, "_original")


def test_rrd_remapping_multiple_original():
    r = RRD({1: 2})
    r.remap({1: "foo"})
    r.remap({"foo": "bar"})
    assert r.original == {"foo": 2}


def test_rrd_str():
    assert str(RRD({1: 2}))


def test_rrd_unmap():
    r = RRD({1: 2})
    assert r[1] == 2
    r.remap({1: "foo"})
    assert r["foo"] == 2
    with pytest.raises(KeyError):
        r[1]
    r.unmap()
    assert r[1] == 2
    with pytest.raises(KeyError):
        r["foo"]


def test_rrd_unmap_reversed():
    r = RRD({1: 2})
    assert not hasattr(r, "_reversed")
    r.reversed
    assert hasattr(r, "_reversed")
    r.remap({1: "foo"})
    assert not hasattr(r, "_reversed")
    r.reversed
    assert hasattr(r, "_reversed")
    r.unmap()
    assert not hasattr(r, "_reversed")
    assert r.reversed == {2: 1}
    assert hasattr(r, "_reversed")


def test_rrd_unmap_original():
    r = RRD({1: 2})
    r.remap({1: "foo"})
    assert r["foo"] == 2
    assert r.original == {1: 2}
    r.unmap()
    assert r.original == {1: 2}


def test_rrd_iter():
    r = RRD({1: 2})
    assert list(r) == [1]


def test_rrd_len():
    assert len(RRD({1: 2})) == 1
