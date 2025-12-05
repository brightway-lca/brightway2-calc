import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray

from bw2calc.fast_scores import PYPARDISO, FastScoresOnlyMultiLCA
from bw2calc.method_config import MethodConfig
from bw2calc.utils import get_datapackage

try:
    import scikits.umfpack  # noqa: F401

    UMFPACK = True
except ImportError:
    UMFPACK = False


def test_initialization_chunk_size(basic_test_data):
    """Test initialization with default chunk size."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    assert fsmlca.chunk_size == 50

    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
        chunk_size=100,
    )
    assert fsmlca.chunk_size == 100


def test_umfpack_warning(basic_test_data, only_umfpack_available):
    """Test that UMFPACK warning is issued when using FastScoresOnlyMultiLCA."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FastScoresOnlyMultiLCA(
            demands=basic_test_data["demands"],
            method_config=basic_test_data["config"],
            data_objs=basic_test_data["dps"],
        )
        assert len(w) == 1
        assert "UMFPACK" in str(w[0].message)
        assert "PARDISO" in str(w[0].message)


def test_methods_not_implemented(basic_test_data):
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    with pytest.raises(NotImplementedError, match="LCI and LCIA aren't separate"):
        fsmlca.lci()
    with pytest.raises(NotImplementedError, match="LCI and LCIA aren't separate"):
        fsmlca.lci_calculation()
    with pytest.raises(NotImplementedError, match="LCI and LCIA aren't separate"):
        fsmlca.lcia()
    with pytest.raises(NotImplementedError, match="LCI and LCIA aren't separate"):
        fsmlca.lcia_calculation()


def test_build_precalculated_basic(basic_test_data):
    """Test build_precalculated with basic setup."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()

    fsmlca.build_precalculated()

    # Check that precalculated is a dictionary
    assert isinstance(fsmlca.precalculated, dict)

    # Check that all characterization matrices are processed
    assert len(fsmlca.precalculated) == len(fsmlca.characterization_matrices)

    # Check that the correct keys are present
    expected_keys = {("first", "category"), ("second", "category")}
    assert set(fsmlca.precalculated.keys()) == expected_keys

    # Check that each precalculated matrix is a numpy array
    for key, matrix in fsmlca.precalculated.items():
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2
        assert matrix.shape[0] == 1
        assert key in expected_keys


def test_build_precalculated_with_normalization(basic_test_data):
    """Test build_precalculated with normalization matrices."""
    # Add normalization datapackage
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    dps_with_norm = basic_test_data["dps"] + [
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip")
    ]

    config_with_norm = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "normalizations": {
            ("n", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
    }

    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"], method_config=config_with_norm, data_objs=dps_with_norm
    )
    fsmlca._load_datapackages()

    fsmlca.build_precalculated()

    assert hasattr(fsmlca, "normalization_matrices")
    assert isinstance(fsmlca.precalculated, dict)

    # Check that the correct keys are present (should be the same as basic test)
    expected_keys = {(("n", "1"), ("first", "category")), (("n", "1"), ("second", "category"))}
    assert set(fsmlca.precalculated.keys()) == expected_keys

    # Check that each precalculated matrix is a numpy array
    for key, matrix in fsmlca.precalculated.items():
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2
        assert matrix.shape[0] == 1
        assert key in expected_keys


def test_build_precalculated_with_weighting(basic_test_data):
    """Test build_precalculated with weighting matrices."""
    # Add weighting datapackage
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    dps_with_weight = basic_test_data["dps"] + [
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip")
    ]

    config_with_weight = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "weightings": {
            ("w", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
    }

    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=config_with_weight,
        data_objs=dps_with_weight,
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Check that weighting matrices are applied
    assert hasattr(fsmlca, "weighting_matrices")
    assert isinstance(fsmlca.precalculated, dict)

    # Check that the correct keys are present (should be the same as basic test)
    expected_keys = {(("w", "1"), ("first", "category")), (("w", "1"), ("second", "category"))}
    assert set(fsmlca.precalculated.keys()) == expected_keys

    # Check that each precalculated matrix is a numpy array
    for key, matrix in fsmlca.precalculated.items():
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2
        assert matrix.shape[0] == 1
        assert key in expected_keys


def test_no_solver_fixture(no_solvers_available):
    with pytest.raises(ImportError):
        import pypardiso

        assert pypardiso
    with pytest.raises(ImportError):
        import scikits.umfpack  # noqa: F811

        assert scikits.umfpack


def test_calculate_no_solver_error(basic_test_data, no_solvers_available):
    """Test that calculate raises error when no suitable solver is available."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    with pytest.raises(ValueError, match="only supported with PARDISO and UMFPACK solvers"):
        fsmlca.calculate()


def test_next_no_solver_error(basic_test_data, no_solvers_available):
    """Test that __next__ raises error when no suitable solver is available."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    with pytest.raises(ValueError, match="only supported with PARDISO and UMFPACK"):
        next(fsmlca)


# Test the scores property getter and setter


def test_scores_getter_not_calculated(basic_test_data):
    """Test that scores getter raises error when scores not calculated."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )

    with pytest.raises(ValueError, match="Scores not calculated yet"):
        _ = fsmlca.scores


def test_scores_setter_and_getter(basic_test_data):
    """Test that scores setter and getter work correctly."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )

    # Create a mock DataArray
    mock_scores = xarray.DataArray(
        np.array([[1.0, 2.0, 3.0]]),
        coords=[["first||category"], ["γ", "ε", "ζ"]],
        dims=["LCIA", "processes"],
    )

    # Set scores
    fsmlca.scores = mock_scores

    # Get scores
    retrieved_scores = fsmlca.scores

    # Check that they are the same
    assert retrieved_scores is mock_scores
    assert hasattr(fsmlca, "_scores")
    assert fsmlca._scores is mock_scores


def test_calculation_with_different_chunk_sizes(
    basic_test_data, only_pypardiso_available, mock_pypardiso_solver
):
    """Test calculation with different chunk sizes."""
    chunk_sizes = [1, 2, 5, 10, 50]

    for chunk_size in chunk_sizes:
        fsmlca = FastScoresOnlyMultiLCA(
            demands=basic_test_data["demands"],
            method_config=basic_test_data["config"],
            data_objs=basic_test_data["dps"],
            chunk_size=chunk_size,
        )
        assert fsmlca.chunk_size == chunk_size

        # Mock the PyPardisoSolver and PYPARDISO flag
        with (
            patch("bw2calc.fast_supply_arrays.PYPARDISO", True),
            patch("bw2calc.fast_supply_arrays.PyPardisoSolver", mock_pypardiso_solver),
        ):
            result = fsmlca.calculate()

            # All should produce valid results
            assert isinstance(result, xarray.DataArray)
            assert result.dims == ("LCIA", "processes")
            assert result.shape[0] == 2  # Two impact categories
            assert result.shape[1] == 3  # Three functional units


def test_calculation_with_normalization_and_weighting(
    basic_test_data, only_pypardiso_available, mock_pypardiso_solver
):
    """Test calculation with normalization and weighting."""
    # Add normalization and weighting datapackages
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    dps_with_all = basic_test_data["dps"] + [
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
    ]

    config_with_all = {
        "impact_categories": [
            ("first", "category"),
            ("second", "category"),
        ],
        "normalizations": {
            ("n", "1"): [
                ("first", "category"),
                ("second", "category"),
            ]
        },
        "weightings": {
            ("w", "1"): [
                ("n", "1"),
            ]
        },
    }

    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"], method_config=config_with_all, data_objs=dps_with_all
    )

    # Mock the PyPardisoSolver
    with (
        patch("bw2calc.fast_supply_arrays.PYPARDISO", True),
        patch("bw2calc.fast_supply_arrays.PyPardisoSolver", mock_pypardiso_solver),
    ):
        result = fsmlca.calculate()

        # Should still work with normalization and weighting
        assert isinstance(result, xarray.DataArray)
        assert result.dims == ("LCIA", "processes")
        assert hasattr(fsmlca, "precalculated")

        # Check that precalculated has the correct keys
        expected_keys = {
            (("w", "1"), ("n", "1"), ("second", "category")),
            (("w", "1"), ("n", "1"), ("first", "category")),
        }
        assert set(fsmlca.precalculated.keys()) == expected_keys


@pytest.mark.skipif((not PYPARDISO and not UMFPACK), reason="Fast sparse solvers not installed")
def test_integration(basic_test_data, fixture_dir):
    method_config = MethodConfig(
        impact_categories=[
            ("first", "category"),
            ("second", "category"),
        ],
        normalizations={
            ("n", "1"): [("first", "category")],
            ("n", "2"): [("second", "category")],
        },
        weightings={
            ("w", "1"): [("n", "1")],
            ("w", "2"): [("n", "2")],
        },
    )

    func_units = basic_test_data["demands"]
    dps = basic_test_data["dps"]
    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_normalization.zip"),
    )
    dps.append(
        get_datapackage(fixture_dir / "multi_lca_simple_weighting.zip"),
    )

    mlca = FastScoresOnlyMultiLCA(demands=func_units, method_config=method_config, data_objs=dps)
    mlca.calculate()

    assert mlca.scores.shape == (2, 3)
    assert (
        mlca.scores.loc["(('w', '2'), ('n', '2'), ('second', 'category'))", "ζ"]
        == 3 * (3 * 10 + 1 * 10) * 84
    )
    assert mlca.scores.loc["(('w', '1'), ('n', '1'), ('first', 'category'))", "γ"] == 3 * 42
