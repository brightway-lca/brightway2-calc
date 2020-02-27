from io import BytesIO

# import numpy as np

from bw2calc.utils import get_seed, wrap_functional_unit, os, load_arrays, np
import multiprocessing
import pytest
import sys

MAX_INT_32 = 4294967295


@pytest.mark.skipif(sys.version_info < (3,0), reason="MP pool changes")
def test_get_seeds_different_under_mp_pool():
    with multiprocessing.Pool(processes=4) as pool:
        results = list(pool.map(get_seed, [None] * 10))
    assert sorted(set(results)) == sorted(results)


def test_wrap_functional_unit():
    given = {17: 42}
    expected = {'key': 17, 'amount': 42}
    assert wrap_functional_unit(given) == [expected]

    given = {('a', 'b'): 42}
    expected = {'database': 'a', 'code': 'b', 'amount': 42}
    assert wrap_functional_unit(given) == [expected]

    class Foo:
        def __getitem__(self, index):
            if index == 0:
                return 'a'
            elif index == 1:
                return 'b'

    given = {Foo(): 42}
    expected = {'database': 'a', 'code': 'b', 'amount': 42}
    assert wrap_functional_unit(given) == [expected]


def test_load_arrays_order(monkeypatch):
    monkeypatch.setattr(np, "load", lambda x, allow_pickle: x)
    monkeypatch.setattr(np, "hstack", lambda x: x)
    given = [1, "a", np.ones(5), 2, "b", np.zeros(3)]
    result = load_arrays(given)
    assert result[0].sum() == 5
    assert result[1].sum() == 0
    assert result[2:] == ["a", "b", 1, 2]


def test_load_arrays_string():
    # use the independent lca fixture data
    fpath = os.path.join(os.path.dirname(__file__), "fixtures", "independent", "ia.npy")

    dtype = [
        ('flow', np.uint32),
        ('row', np.uint32),
        ('col', np.uint32),
        ('geo', np.uint32),
        ('amount', np.float32),
    ]
    expected_array = np.asarray(
        [(10, MAX_INT_32, MAX_INT_32, 17, 1),
         (11, MAX_INT_32, MAX_INT_32, 17, 10)],
        dtype=dtype)

    actual_array = load_arrays([fpath])

    assert np.all(actual_array == expected_array), "Failed loading array from filepath"


def test_load_arrays_bytesio():
    dtype = [
        ('flow', np.uint32),
        ('row', np.uint32),
        ('col', np.uint32),
        ('geo', np.uint32),
        ('amount', np.float32),
    ]
    expected_array = np.asarray(
        [(10, MAX_INT_32, MAX_INT_32, 17, 1),
         (11, MAX_INT_32, MAX_INT_32, 17, 10.2)],
        dtype=dtype)

    with BytesIO() as b:
        np.save(b, expected_array, allow_pickle=False)
        b.seek(0)  # return to beginning of file
        actual_array = load_arrays([b])

    assert np.all(actual_array == expected_array), "Failed loading array from bytes"

