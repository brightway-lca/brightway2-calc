import numpy as np
import pytest

from bw2calc.result_cache import ResultCache


def test_first_use():
    rc = ResultCache()
    assert not hasattr(rc, "array")
    rc.add([5], np.arange(5).reshape((-1, 1)))

    assert rc.array.shape == (5, 100)
    assert np.allclose(rc.array[:, 0], np.arange(5))
    assert rc.indices[5] == 0


def test_missing():
    rc = ResultCache()
    rc.add([5], np.arange(5).reshape((-1, 1)))

    with pytest.raises(KeyError):
        rc[10]


def test_missing_before_first_use():
    rc = ResultCache()

    with pytest.raises(KeyError):
        rc[10]


def test_getitem():
    rc = ResultCache()
    rc.add([5], np.arange(5).reshape((-1, 1)))

    assert np.allclose(rc[5], np.arange(5))


def test_contains():
    rc = ResultCache()
    rc.add([5], np.arange(5).reshape((-1, 1)))

    assert 5 in rc


def test_add_errors():
    rc = ResultCache()
    rc.add([5], np.arange(5).reshape((-1, 1)))

    with pytest.raises(ValueError):
        rc.add([5], np.arange(10).reshape((-1, 1)))
    with pytest.raises(ValueError):
        rc.add([5], np.arange(5).reshape((-1, 1, 1)))
    with pytest.raises(ValueError):
        rc.add([5, 2], np.arange(5).reshape((-1, 1)))


def test_add_2d():
    rc = ResultCache()
    rc.add([5], np.arange(5).reshape((-1, 1)))
    rc.add([7, 10], np.arange(5, 15).reshape((5, 2)))

    assert rc.array.shape == (5, 100)
    assert np.allclose(rc.array[:, 1], [5, 7, 9, 11, 13])
    assert np.allclose(rc.array[:, 2], [6, 8, 10, 12, 14])
    assert rc.indices[5] == 0
    assert rc.indices[7] == 1
    assert rc.indices[10] == 2
    assert np.allclose(rc[7], [5, 7, 9, 11, 13])
    assert np.allclose(rc[10], [6, 8, 10, 12, 14])


def test_dont_overwrite_existing():
    rc = ResultCache()
    rc.add([5], np.arange(5).reshape((-1, 1)))
    rc.add([5, 10], np.arange(5, 15).reshape((5, 2)))

    assert rc.array.shape == (5, 100)
    assert np.allclose(rc.array[:, 0], np.arange(5))
    assert np.allclose(rc.array[:, 1], [6, 8, 10, 12, 14])
    assert rc.indices[5] == 0
    assert rc.indices[10] == 1
    assert np.allclose(rc[5], np.arange(5))
    assert np.allclose(rc[10], [6, 8, 10, 12, 14])


def test_expand():
    rc = ResultCache(10)
    rc.add(list(range(25)), np.arange(100).reshape((4, -1)))

    assert rc.array.shape == (4, 30)
    assert np.allclose(rc.array[0, :25], range(25))
    assert np.allclose(rc.array[:, 0], [0, 25, 50, 75])


def test_reset():
    rc = ResultCache(10)
    rc.add(list(range(25)), np.arange(100).reshape((4, -1)))
    rc.reset()

    assert not hasattr(rc, "array")
    assert rc.indices == {}
    assert rc.next_index == 0
