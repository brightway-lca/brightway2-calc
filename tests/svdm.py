from bw2calc.errors import MultipleValues
from matrix_utils.errors import AllArraysEmpty
from bw2calc.single_value_diagonal_matrix import SingleValueDiagonalMatrix as SVDM
from bw2calc.utils import get_datapackage
from pathlib import Path
import numpy as np
import pytest

fixture_dir = Path(__file__).resolve().parent / "fixtures"


def test_svdm_missing_dimension_kwarg():
    with pytest.raises(TypeError):
        SVDM(packages=["something"], matrix="something")


def test_svdm_no_data():
    with pytest.raises(AllArraysEmpty):
        SVDM(
            packages=[get_datapackage(fixture_dir / "basic_fixture.zip")],
            matrix="weighting_matrix",
            dimension=5,
        )


def test_svdm_multiple_dataresources():
    with pytest.raises(MultipleValues):
        SVDM(
            packages=[
                get_datapackage(fixture_dir / "svdm.zip"),
                get_datapackage(fixture_dir / "svdm2.zip"),
            ],
            matrix="weighting_matrix",
            dimension=500,
        )


def test_svdm_multiple_values():
    with pytest.raises(MultipleValues):
        SVDM(
            packages=[get_datapackage(fixture_dir / "basic_fixture.zip")],
            matrix="technosphere_matrix",
            dimension=500,
        )


def test_svdm_basic():
    obj = SVDM(
        packages=[
            get_datapackage(fixture_dir / "svdm.zip"),
        ],
        matrix="weighting_matrix",
        use_arrays=False,
        dimension=500,
    )
    assert obj.matrix.shape == (500, 500)
    assert obj.matrix.sum() == 500 * 42
    assert np.allclose(obj.matrix.tocoo().row, np.arange(500))
    assert np.allclose(obj.matrix.tocoo().col, np.arange(500))


def test_svdm_iteration():
    obj = SVDM(
        packages=[
            get_datapackage(fixture_dir / "svdm.zip"),
        ],
        matrix="weighting_matrix",
        use_arrays=False,
        dimension=500,
    )
    assert obj.matrix.shape == (500, 500)
    assert obj.matrix.sum() == 500 * 42
    next(obj)
    assert obj.matrix.shape == (500, 500)
    assert obj.matrix.sum() == 500 * 42


def test_svdm_distributions():
    obj = SVDM(
        packages=[
            get_datapackage(fixture_dir / "svdm.zip"),
        ],
        matrix="weighting_matrix",
        use_distributions=True,
        use_arrays=False,
        dimension=500,
    )
    assert obj.matrix.shape == (500, 500)
    total = obj.matrix.sum()
    assert total
    next(obj)
    assert total != obj.matrix.sum()
    next(obj)
    assert total != obj.matrix.sum()


def test_svdm_arrays():
    obj = SVDM(
        packages=[
            get_datapackage(fixture_dir / "svdm.zip"),
        ],
        matrix="weighting_matrix",
        use_vectors=False,
        use_arrays=True,
        dimension=500,
    )
    assert obj.matrix.shape == (500, 500)
    assert obj.matrix.sum() == 500 * 1
    next(obj)
    assert obj.matrix.shape == (500, 500)
    assert obj.matrix.sum() == 500 * 2
    next(obj)
    assert obj.matrix.shape == (500, 500)
    assert obj.matrix.sum() == 500 * 3
