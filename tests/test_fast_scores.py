import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray

from bw2calc.fast_scores import FastScoresOnlyMultiLCA
from bw2calc.utils import get_datapackage

# Test FastScoresOnlyMultiLCA initialization and basic properties


@pytest.mark.solver_agnostic
def test_initialization_with_default_chunk_size(basic_test_data):
    """Test initialization with default chunk size."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    assert fsmlca.chunk_size == 50
    assert isinstance(fsmlca, FastScoresOnlyMultiLCA)


@pytest.mark.solver_agnostic
def test_initialization_with_custom_chunk_size(basic_test_data):
    """Test initialization with custom chunk size."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
        chunk_size=100,
    )
    assert fsmlca.chunk_size == 100


@pytest.mark.solver_agnostic
def test_initialization_with_small_chunk_size(basic_test_data):
    """Test initialization with small chunk size."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
        chunk_size=5,
    )
    assert fsmlca.chunk_size == 5


@pytest.mark.umfpack
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


# Test that certain methods raise NotImplementedError as expected


@pytest.mark.solver_agnostic
def test_lci_not_implemented(basic_test_data):
    """Test that lci() raises NotImplementedError."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    with pytest.raises(NotImplementedError, match="LCI and LCIA aren't separate"):
        fsmlca.lci()


@pytest.mark.solver_agnostic
def test_lci_calculation_not_implemented(basic_test_data):
    """Test that lci_calculation() raises NotImplementedError."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    with pytest.raises(NotImplementedError, match="LCI and LCIA aren't separate"):
        fsmlca.lci_calculation()


@pytest.mark.solver_agnostic
def test_lcia_not_implemented(basic_test_data):
    """Test that lcia() raises NotImplementedError."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    with pytest.raises(NotImplementedError, match="LCI and LCIA aren't separate"):
        fsmlca.lcia()


@pytest.mark.solver_agnostic
def test_lcia_calculation_not_implemented(basic_test_data):
    """Test that lcia_calculation() raises NotImplementedError."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    with pytest.raises(NotImplementedError, match="LCI and LCIA aren't separate"):
        fsmlca.lcia_calculation()


# Test the build_precalculated method


@pytest.mark.solver_agnostic
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


@pytest.mark.solver_agnostic
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

    # Check that normalization matrices are applied
    print(list(fsmlca.__dict__.keys()))
    assert hasattr(fsmlca, "normalization_matrices")
    assert isinstance(fsmlca.precalculated, dict)

    # Check that the correct keys are present (should be the same as basic test)
    expected_keys = {("first", "category"), ("second", "category")}
    assert set(fsmlca.precalculated.keys()) == expected_keys

    # Check that each precalculated matrix is a numpy array
    for key, matrix in fsmlca.precalculated.items():
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 1  # Should be summed to 1D
        assert key in expected_keys


@pytest.mark.solver_agnostic
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
    print(list(fsmlca.__dict__.keys()))
    assert hasattr(fsmlca, "weighting_matrices")
    assert isinstance(fsmlca.precalculated, dict)

    # Check that the correct keys are present (should be the same as basic test)
    expected_keys = {("first", "category"), ("second", "category")}
    assert set(fsmlca.precalculated.keys()) == expected_keys

    # Check that each precalculated matrix is a numpy array
    for key, matrix in fsmlca.precalculated.items():
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2
        assert matrix.shape[0] == 1
        assert key in expected_keys


# Test the calculate method and solver selection


@pytest.mark.no_solvers
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


@pytest.mark.pypardiso
def test_calculate_pardiso_selection(
    basic_test_data, only_pypardiso_available, mock_pypardiso_solver
):
    """Test that PARDISO is selected when available."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Mock the PyPardisoSolver import
    with patch("bw2calc.fast_scores.PyPardisoSolver", mock_pypardiso_solver) as mock_solver_class:
        result = fsmlca.calculate()
        assert isinstance(result, xarray.DataArray)

        # Verify that PyPardisoSolver was instantiated
        mock_solver_class.assert_called_once()

        # Get the mock solver instance and verify its methods were called
        mock_solver_instance = mock_solver_class.return_value
        mock_solver_instance.factorize.assert_called_once()
        # set_phase should be called for each chunk
        assert mock_solver_instance.set_phase.call_count >= 1
        # _call_pardiso should be called for each chunk
        assert mock_solver_instance._call_pardiso.call_count >= 1


@pytest.mark.umfpack
def test_calculate_umfpack_selection(basic_test_data, only_umfpack_available):
    """Test that UMFPACK is selected when PARDISO is not available."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Mock the factorized function
    with patch("bw2calc.fast_scores.factorized") as mock_factorized:
        mock_factorized.return_value = lambda x: np.ones(fsmlca.technosphere_matrix.shape[0])
        result = fsmlca.calculate()
        assert isinstance(result, xarray.DataArray)

        # Verify that factorized was called
        mock_factorized.assert_called_once()

        # Verify that the returned solver function was called for each demand
        solver_func = mock_factorized.return_value
        # The solver function should be called once for each demand array
        assert solver_func.call_count == len(fsmlca.demand_arrays)


# Test the _calculate_pardiso method


@pytest.mark.pypardiso
def test_calculate_pardiso_basic(basic_test_data, only_pypardiso_available, mock_pypardiso_solver):
    """Test _calculate_pardiso with basic setup."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Mock the PyPardisoSolver
    with patch("bw2calc.fast_scores.PyPardisoSolver", mock_pypardiso_solver) as mock_solver_class:
        result = fsmlca._calculate_pardiso()

        # Check return type
        assert isinstance(result, xarray.DataArray)

        # Check dimensions
        assert result.dims == ("LCIA", "processes")

        # Check coordinates
        assert "LCIA" in result.coords
        assert "processes" in result.coords

        # Check that scores are set
        assert hasattr(fsmlca, "_scores")
        assert fsmlca._scores is result

        # Verify that PyPardisoSolver was instantiated
        mock_solver_class.assert_called_once()

        # Get the mock solver instance and verify its methods were called
        mock_solver_instance = mock_solver_class.return_value
        mock_solver_instance.factorize.assert_called_once_with(fsmlca.technosphere_matrix)
        # set_phase should be called for each chunk
        assert mock_solver_instance.set_phase.call_count >= 1
        # _call_pardiso should be called for each chunk
        assert mock_solver_instance._call_pardiso.call_count >= 1


@pytest.mark.pypardiso
def test_calculate_pardiso_chunking(
    basic_test_data, only_pypardiso_available, mock_pypardiso_solver
):
    """Test _calculate_pardiso with chunking functionality."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
        chunk_size=1,
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Mock the PyPardisoSolver
    with patch("bw2calc.fast_scores.PyPardisoSolver", mock_pypardiso_solver) as mock_solver_class:
        result = fsmlca._calculate_pardiso()

        # Should still work with small chunk size
        assert isinstance(result, xarray.DataArray)
        assert result.dims == ("LCIA", "processes")

        # Verify that PyPardisoSolver was instantiated
        mock_solver_class.assert_called_once()

        # Get the mock solver instance and verify its methods were called
        mock_solver_instance = mock_solver_class.return_value
        mock_solver_instance.factorize.assert_called_once()
        # With chunk_size=1, we should have more calls to set_phase and _call_pardiso
        assert mock_solver_instance.set_phase.call_count >= 1
        assert mock_solver_instance._call_pardiso.call_count >= 1


@pytest.mark.pypardiso
def test_calculate_pardiso_supply_array(
    basic_test_data, only_pypardiso_available, mock_pypardiso_solver
):
    """Test that supply_array is properly set in _calculate_pardiso."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Mock the PyPardisoSolver
    with patch("bw2calc.fast_scores.PyPardisoSolver", mock_pypardiso_solver):
        fsmlca._calculate_pardiso()

        # Check that supply_array is set
        assert hasattr(fsmlca, "supply_array")
        assert isinstance(fsmlca.supply_array, np.ndarray)
        assert fsmlca.supply_array.ndim == 2  # Should be 2D array


# Test the _calculate_umfpack method


@pytest.mark.umfpack
def test_calculate_umfpack_basic(basic_test_data, only_umfpack_available):
    """Test _calculate_umfpack with basic setup."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Mock the factorized function
    with patch("bw2calc.fast_scores.factorized") as mock_factorized:
        mock_factorized.return_value = lambda x: np.ones(fsmlca.technosphere_matrix.shape[0])
        result = fsmlca._calculate_umfpack()

        # Check return type
        assert isinstance(result, xarray.DataArray)

        # Check dimensions
        assert result.dims == ("LCIA", "processes")

        # Check coordinates
        assert "LCIA" in result.coords
        assert "processes" in result.coords

        # Check that scores are set
        assert hasattr(fsmlca, "_scores")
        assert fsmlca._scores is result

        # Verify that factorized was called with the technosphere matrix
        mock_factorized.assert_called_once()
        call_args = mock_factorized.call_args[0]
        assert call_args[0] is fsmlca.technosphere_matrix.tocsc()

        # Verify that the returned solver function was called for each demand
        solver_func = mock_factorized.return_value
        assert solver_func.call_count == len(fsmlca.demand_arrays)


@pytest.mark.umfpack
def test_calculate_umfpack_supply_array(basic_test_data, only_umfpack_available):
    """Test that supply_array is properly set in _calculate_umfpack."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Mock the factorized function
    with patch("bw2calc.fast_scores.factorized") as mock_factorized:
        mock_factorized.return_value = lambda x: np.ones(fsmlca.technosphere_matrix.shape[0])
        fsmlca._calculate_umfpack()

        # Check that supply_array is set
        assert hasattr(fsmlca, "supply_array")
        assert isinstance(fsmlca.supply_array, np.ndarray)
        assert fsmlca.supply_array.ndim == 2  # Should be 2D array


# Test the __next__ method and solver cleanup


@pytest.mark.umfpack
def test_next_umfpack_solver_cleanup(basic_test_data, only_umfpack_available):
    """Test that UMFPACK solver is cleaned up in __next__."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Add a mock solver attribute
    fsmlca.solver = "mock_solver"

    # Call __next__
    next(fsmlca)

    # Check that solver attribute is removed
    assert not hasattr(fsmlca, "solver")


@pytest.mark.pypardiso
def test_next_pardiso_memory_cleanup(basic_test_data, only_pypardiso_available):
    """Test that PARDISO memory is cleaned up in __next__."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    # Mock the pypardiso_solver.free_memory method
    with patch("bw2calc.fast_scores.PyPardisoSolver") as mock_solver_class:
        mock_solver = mock_solver_class.return_value
        mock_solver.free_memory = Mock()

        # Mock the pypardiso.scipy_aliases import
        with patch("bw2calc.fast_scores.pypardiso_solver") as mock_pypardiso_solver:
            mock_pypardiso_solver.free_memory = Mock()

            # This should not raise an error
            next(fsmlca)

            # Verify that free_memory was called
            mock_pypardiso_solver.free_memory.assert_called_once()


@pytest.mark.no_solvers
def test_next_no_solver_error(basic_test_data, no_solvers_available):
    """Test that __next__ raises error when no suitable solver is available."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )
    fsmlca._load_datapackages()
    fsmlca.build_precalculated()

    with pytest.raises(ValueError, match="No suitable solver installed"):
        next(fsmlca)


# Test the scores property getter and setter


@pytest.mark.solver_agnostic
def test_scores_getter_not_calculated(basic_test_data):
    """Test that scores getter raises error when scores not calculated."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )

    with pytest.raises(ValueError, match="Scores not calculated yet"):
        _ = fsmlca.scores


@pytest.mark.solver_agnostic
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


@pytest.mark.solver_agnostic
def test_scores_property_direct_access(basic_test_data):
    """Test direct access to _scores attribute."""
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

    # Set scores directly
    fsmlca._scores = mock_scores

    # Get scores via property
    retrieved_scores = fsmlca.scores

    # Check that they are the same
    assert retrieved_scores is mock_scores


# Integration tests for the complete calculation workflow


@pytest.mark.pypardiso
def test_complete_calculation_workflow(
    basic_test_data, only_pypardiso_available, mock_pypardiso_solver
):
    """Test the complete calculation workflow."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )

    # Mock the PyPardisoSolver
    with patch("bw2calc.fast_scores.PyPardisoSolver", mock_pypardiso_solver) as mock_solver_class:
        # Test that calculate() loads data and builds precalculated matrices
        result = fsmlca.calculate()

        # Check that all necessary attributes are set
        assert hasattr(fsmlca, "technosphere_matrix")
        assert hasattr(fsmlca, "biosphere_matrix")
        assert hasattr(fsmlca, "characterization_matrices")
        assert hasattr(fsmlca, "precalculated")
        assert hasattr(fsmlca, "supply_array")
        assert hasattr(fsmlca, "_scores")

        # Check that precalculated has the correct keys
        expected_keys = {("first", "category"), ("second", "category")}
        assert set(fsmlca.precalculated.keys()) == expected_keys

        # Check result type and structure
        assert isinstance(result, xarray.DataArray)
        assert result.dims == ("LCIA", "processes")

        # Check that scores property works
        scores = fsmlca.scores
        assert scores is result

        # Verify that PyPardisoSolver was actually used
        mock_solver_class.assert_called_once()
        mock_solver_instance = mock_solver_class.return_value
        mock_solver_instance.factorize.assert_called_once()
        assert mock_solver_instance.set_phase.call_count >= 1
        assert mock_solver_instance._call_pardiso.call_count >= 1


@pytest.mark.pypardiso
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

        # Mock the PyPardisoSolver
        with patch("bw2calc.fast_scores.PyPardisoSolver", mock_pypardiso_solver):
            result = fsmlca.calculate()

            # All should produce valid results
            assert isinstance(result, xarray.DataArray)
            assert result.dims == ("LCIA", "processes")
            assert result.shape[0] == 2  # Two impact categories
            assert result.shape[1] == 3  # Three functional units


@pytest.mark.pypardiso
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
                ("first", "category"),
                ("second", "category"),
            ]
        },
    }

    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"], method_config=config_with_all, data_objs=dps_with_all
    )

    # Mock the PyPardisoSolver
    with patch("bw2calc.fast_scores.PyPardisoSolver", mock_pypardiso_solver):
        result = fsmlca.calculate()

        # Should still work with normalization and weighting
        assert isinstance(result, xarray.DataArray)
        assert result.dims == ("LCIA", "processes")
        assert hasattr(fsmlca, "precalculated")

        # Check that precalculated has the correct keys
        expected_keys = {("first", "category"), ("second", "category")}
        assert set(fsmlca.precalculated.keys()) == expected_keys


@pytest.mark.umfpack
def test_umfpack_complete_calculation_workflow(basic_test_data, only_umfpack_available):
    """Test the complete calculation workflow with UMFPACK."""
    fsmlca = FastScoresOnlyMultiLCA(
        demands=basic_test_data["demands"],
        method_config=basic_test_data["config"],
        data_objs=basic_test_data["dps"],
    )

    # Mock the factorized function
    with patch("bw2calc.fast_scores.factorized") as mock_factorized:
        mock_factorized.return_value = lambda x: np.ones(fsmlca.technosphere_matrix.shape[0])

        # Test that calculate() loads data and builds precalculated matrices
        result = fsmlca.calculate()

        # Check that all necessary attributes are set
        assert hasattr(fsmlca, "technosphere_matrix")
        assert hasattr(fsmlca, "biosphere_matrix")
        assert hasattr(fsmlca, "characterization_matrices")
        assert hasattr(fsmlca, "precalculated")
        assert hasattr(fsmlca, "supply_array")
        assert hasattr(fsmlca, "_scores")

        # Check that precalculated has the correct keys
        expected_keys = {("first", "category"), ("second", "category")}
        assert set(fsmlca.precalculated.keys()) == expected_keys

        # Check result type and structure
        assert isinstance(result, xarray.DataArray)
        assert result.dims == ("LCIA", "processes")

        # Check that scores property works
        scores = fsmlca.scores
        assert scores is result

        # Verify that factorized was actually used
        mock_factorized.assert_called_once()
        call_args = mock_factorized.call_args[0]
        assert call_args[0] is fsmlca.technosphere_matrix.tocsc()

        # Verify that the returned solver function was called for each demand
        solver_func = mock_factorized.return_value
        assert solver_func.call_count == len(fsmlca.demand_arrays)
