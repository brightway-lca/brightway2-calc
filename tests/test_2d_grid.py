import pytest

from bw2calc.grid import TwoDimensionalGrid


@pytest.fixture
def grid():
    keys = [("a", 1), ("a", 2), ("b", 1), ("c", 3)]
    values = [10, 11, 12, 13]
    return TwoDimensionalGrid(keys, values)


def test_2d_grid_inconsistent():
    keys = [("a", 1), ("a", 2)]
    values = [10, 11, 12, 13]
    with pytest.raises(ValueError):
        TwoDimensionalGrid(keys, values)


def test_2d_grid_empty():
    grid = TwoDimensionalGrid([], [])
    assert len(grid) == 0


def test_2d_grid_normal(grid):
    keys = [("a", 1), ("a", 2), ("b", 1), ("c", 3)]

    assert grid[("a", 2)] == 11
    assert len(grid) == 4

    for item, reference in zip(grid, keys):
        assert item == reference


def test_2d_grid_get_slice(grid):
    assert grid[("a", ...)] == {1: 10, 2: 11}
    assert grid[(..., 1)] == {"a": 10, "b": 12}


def test_2d_grid_slice_error(grid):
    with pytest.raises(KeyError):
        grid[(..., ...)]


def test_2d_grid_wrong_keylength(grid):
    with pytest.raises(KeyError):
        grid["abc"]
    with pytest.raises(KeyError):
        grid["a"]


def test_2d_grid_missing_key(grid):
    with pytest.raises(KeyError):
        grid[("d", 1)]


def test_2d_grid_read_only(grid):
    with pytest.raises(TypeError):
        grid[("d", 3)] = "foo"
