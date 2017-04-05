import numpy as np
from bw2calc.indexing import index_with_searchsorted, index_with_arrays
import pytest

MAX_INT_32 = 4294967295


def test_index_with_searchsorted():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    expected = np.array([0, 1, 2, 4, 5, 6, 5, 4, 3])
    mapping = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 9: 5, 12: 6}
    output = np.zeros(inpt.size)
    result = index_with_searchsorted(inpt, output)
    assert result == mapping
    assert np.allclose(expected, output)

def test_index_with_searchsorted_preserves_dtype():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    output = np.zeros(inpt.size, dtype=np.uint32)
    index_with_searchsorted(inpt, output)
    assert output.dtype == np.uint32

def test_index_with_arrays():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    mapping = {1: 0, 3: 2, 5: 3, 6: 4, 9: 5}
    expected = np.array([0, MAX_INT_32, 2, 4, 5, MAX_INT_32, 5, 4, 3])
    output = np.zeros(inpt.size)
    index_with_arrays(inpt, output, mapping)
    assert np.allclose(output, expected)

def test_index_with_arrays_preserves_dtype():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    mapping = {1: 0, 3: 2, 5: 3, 6: 4, 9: 5}
    output = np.zeros(inpt.size, dtype=np.uint32)
    index_with_arrays(inpt, output, mapping)
    assert output.dtype == np.uint32

def test_index_with_arrays_negative_error():
    inpt = np.array([1, 2, 3, 6, 9, 12, 9, 6, 5])
    mapping = {-1: 0, 3: 2, 5: 3, 6: 4, 9: 5}
    output = np.zeros(inpt.size)
    with pytest.raises(ValueError):
        index_with_arrays(inpt, output, mapping)

