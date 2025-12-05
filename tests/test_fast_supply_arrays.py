"""Unit tests for FastSupplyArraysMixin."""

from unittest.mock import patch

import numpy as np
import pytest

from bw2calc import PYPARDISO, UMFPACK
from bw2calc.fast_supply_arrays import FastSupplyArraysMixin
from bw2calc.lca import LCA


# Create a test class that mixes FastSupplyArraysMixin with LCA
# Note: Name doesn't start with "Test" to avoid pytest collection
class LCAWithFastSupplyArrays(LCA, FastSupplyArraysMixin):
    """Test class that mixes FastSupplyArraysMixin with LCA."""

    pass


@pytest.fixture
def basic_lca(fixture_dir):
    """Create a basic LCA instance with FastSupplyArraysMixin."""
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCAWithFastSupplyArrays({1: 1}, data_objs=packages)
    lca.load_lci_data()
    return lca


class TestChunkSize:
    """Test chunk_size attribute and set_chunk_size method."""

    def test_default_chunk_size(self, basic_lca):
        """Test that default chunk_size is 50."""
        assert basic_lca.chunk_size == 50

    def test_set_chunk_size_valid(self, basic_lca):
        """Test setting chunk_size with valid values."""
        basic_lca.set_chunk_size(100)
        assert basic_lca.chunk_size == 100

        basic_lca.set_chunk_size(1)
        assert basic_lca.chunk_size == 1

    def test_set_chunk_size_invalid(self, basic_lca):
        """Test that set_chunk_size raises ValueError for invalid values."""
        with pytest.raises(ValueError, match="Invalid chunk_size"):
            basic_lca.set_chunk_size(0)

        with pytest.raises(ValueError, match="Invalid chunk_size"):
            basic_lca.set_chunk_size(-1)


class TestCalculateSupplyArraysNoSolver:
    """Test calculate_supply_arrays when no solver is available."""

    def test_no_solver_error(self, basic_lca, no_solvers_available):
        """Test that calculate_supply_arrays raises error when no solver is available."""
        # Create demand array with correct size
        demand = np.zeros(len(basic_lca.dicts.product))
        demand[basic_lca.dicts.product[1]] = 1.0
        demand_arrays = [demand]

        # Patch the fast_supply_arrays module's solver flags
        with (
            patch("bw2calc.fast_supply_arrays.PYPARDISO", False),
            patch("bw2calc.fast_supply_arrays.UMFPACK", False),
        ):
            with pytest.raises(ValueError, match="only supported with PARDISO and UMFPACK solvers"):
                basic_lca.calculate_supply_arrays(demand_arrays)


class TestCalculateSupplyArraysUMFPACK:
    """Test calculate_supply_arrays with UMFPACK solver."""

    @pytest.mark.skipif(not UMFPACK, reason="UMFPACK not available")
    def test_calculate_supply_arrays_umfpack_single(self, basic_lca):
        """Test calculate_supply_arrays with UMFPACK for a single demand array."""
        # Build demand array
        basic_lca.build_demand_array()
        demand_arrays = [basic_lca.demand_array]

        # Calculate supply arrays
        result = basic_lca.calculate_supply_arrays(demand_arrays)

        # Check result shape
        assert result.shape == (basic_lca.technosphere_matrix.shape[0], 1)

        # Verify against standard solve
        expected = basic_lca.solve_linear_system(basic_lca.demand_array)
        np.testing.assert_array_almost_equal(result[:, 0], expected)

    @pytest.mark.skipif(not UMFPACK, reason="UMFPACK not available")
    def test_calculate_supply_arrays_umfpack_multiple(self, basic_lca):
        """Test calculate_supply_arrays with UMFPACK for multiple demand arrays."""
        # Create multiple demand arrays
        demand1 = np.zeros(len(basic_lca.dicts.product))
        demand1[basic_lca.dicts.product[1]] = 1.0

        demand2 = np.zeros(len(basic_lca.dicts.product))
        demand2[basic_lca.dicts.product[1]] = 2.0

        demand_arrays = [demand1, demand2]

        # Calculate supply arrays
        result = basic_lca.calculate_supply_arrays(demand_arrays)

        # Check result shape
        assert result.shape == (basic_lca.technosphere_matrix.shape[0], 2)

        # Verify against standard solve for each demand
        expected1 = basic_lca.solve_linear_system(demand1)
        expected2 = basic_lca.solve_linear_system(demand2)

        np.testing.assert_array_almost_equal(result[:, 0], expected1)
        np.testing.assert_array_almost_equal(result[:, 1], expected2)

    def test_calculate_supply_arrays_umfpack_mocked(self, basic_lca, mock_umfpack_solver):
        """Test calculate_supply_arrays with mocked UMFPACK solver."""
        # Create demand array with correct size
        demand = np.zeros(len(basic_lca.dicts.product))
        demand[basic_lca.dicts.product[1]] = 1.0
        demand_arrays = [demand]

        with (
            patch("bw2calc.fast_supply_arrays.UMFPACK", True),
            patch("bw2calc.fast_supply_arrays.PYPARDISO", False),
            patch("bw2calc.fast_supply_arrays.factorized", mock_umfpack_solver),
        ):
            result = basic_lca.calculate_supply_arrays(demand_arrays)

            # Check result shape
            assert result.shape == (basic_lca.technosphere_matrix.shape[0], 1)

            # With mocked solver returning ones, result should be ones
            expected = np.ones(basic_lca.technosphere_matrix.shape[0])
            np.testing.assert_array_equal(result[:, 0], expected)


class TestCalculateSupplyArraysPARDISO:
    """Test calculate_supply_arrays with PARDISO solver."""

    @pytest.mark.skipif(not PYPARDISO, reason="PARDISO not available")
    def test_calculate_supply_arrays_pardiso_single(self, basic_lca):
        """Test calculate_supply_arrays with PARDISO for a single demand array."""
        # Build demand array
        basic_lca.build_demand_array()
        demand_arrays = [basic_lca.demand_array]

        # Calculate supply arrays
        result = basic_lca.calculate_supply_arrays(demand_arrays)

        # Check result shape
        assert result.shape == (basic_lca.technosphere_matrix.shape[0], 1)

        # Verify against standard solve
        expected = basic_lca.solve_linear_system(basic_lca.demand_array)
        np.testing.assert_array_almost_equal(result[:, 0], expected)

    @pytest.mark.skipif(not PYPARDISO, reason="PARDISO not available")
    def test_calculate_supply_arrays_pardiso_multiple(self, basic_lca):
        """Test calculate_supply_arrays with PARDISO for multiple demand arrays."""
        # Create multiple demand arrays
        demand1 = np.zeros(len(basic_lca.dicts.product))
        demand1[basic_lca.dicts.product[1]] = 1.0

        demand2 = np.zeros(len(basic_lca.dicts.product))
        demand2[basic_lca.dicts.product[1]] = 2.0

        demand_arrays = [demand1, demand2]

        # Calculate supply arrays
        result = basic_lca.calculate_supply_arrays(demand_arrays)

        # Check result shape
        assert result.shape == (basic_lca.technosphere_matrix.shape[0], 2)

        # Verify against standard solve for each demand
        expected1 = basic_lca.solve_linear_system(demand1)
        expected2 = basic_lca.solve_linear_system(demand2)

        np.testing.assert_array_almost_equal(result[:, 0], expected1)
        np.testing.assert_array_almost_equal(result[:, 1], expected2)

    def test_calculate_supply_arrays_pardiso_chunking(
        self, basic_lca, only_pypardiso_available, mock_pypardiso_solver
    ):
        """Test calculate_supply_arrays with PARDISO and chunking."""
        # Set a small chunk size to test chunking
        basic_lca.set_chunk_size(1)

        # Create multiple demand arrays (more than chunk_size)
        demand_arrays = []
        for i in range(5):
            demand = np.zeros(len(basic_lca.dicts.product))
            demand[basic_lca.dicts.product[1]] = float(i + 1)
            demand_arrays.append(demand)

        with (
            patch("bw2calc.fast_supply_arrays.PyPardisoSolver", mock_pypardiso_solver),
            patch("bw2calc.fast_supply_arrays.PYPARDISO", True),
        ):
            result = basic_lca.calculate_supply_arrays(demand_arrays)

            # Check result shape
            assert result.shape == (basic_lca.technosphere_matrix.shape[0], 5)

    def test_calculate_supply_arrays_pardiso_mocked(
        self, basic_lca, only_pypardiso_available, mock_pypardiso_solver
    ):
        """Test calculate_supply_arrays with mocked PARDISO solver."""
        # Create demand array with correct size
        demand = np.zeros(len(basic_lca.dicts.product))
        demand[basic_lca.dicts.product[1]] = 1.0
        demand_arrays = [demand]

        with (
            patch("bw2calc.fast_supply_arrays.PyPardisoSolver", mock_pypardiso_solver),
            patch("bw2calc.fast_supply_arrays.PYPARDISO", True),
        ):
            result = basic_lca.calculate_supply_arrays(demand_arrays)

            # Check result shape
            assert result.shape == (basic_lca.technosphere_matrix.shape[0], 1)

            # Mock solver returns ones, so result should be ones
            expected = np.ones(basic_lca.technosphere_matrix.shape[0])
            np.testing.assert_array_equal(result[:, 0], expected)

    def test_calculate_supply_arrays_pardiso_large_chunk(
        self, basic_lca, only_pypardiso_available, mock_pypardiso_solver
    ):
        """Test calculate_supply_arrays with PARDISO and large chunk size."""
        # Set a large chunk size
        basic_lca.set_chunk_size(100)

        # Create multiple demand arrays
        demand_arrays = []
        for i in range(3):
            demand = np.zeros(len(basic_lca.dicts.product))
            demand[basic_lca.dicts.product[1]] = float(i + 1)
            demand_arrays.append(demand)

        with (
            patch("bw2calc.fast_supply_arrays.PyPardisoSolver", mock_pypardiso_solver),
            patch("bw2calc.fast_supply_arrays.PYPARDISO", True),
        ):
            result = basic_lca.calculate_supply_arrays(demand_arrays)

            # Check result shape
            assert result.shape == (basic_lca.technosphere_matrix.shape[0], 3)


class TestCalculateSupplyArraysEdgeCases:
    """Test edge cases for calculate_supply_arrays."""

    def test_empty_demand_arrays(self, basic_lca, only_pypardiso_available, mock_pypardiso_solver):
        """Test calculate_supply_arrays with empty demand_arrays list."""
        with (
            patch("bw2calc.fast_supply_arrays.PyPardisoSolver", mock_pypardiso_solver),
            patch("bw2calc.fast_supply_arrays.PYPARDISO", True),
        ):
            # Empty list should return array with shape (n_rows, 0)
            result = basic_lca.calculate_supply_arrays([])
            assert result.shape == (basic_lca.technosphere_matrix.shape[0], 0)

    @pytest.mark.skipif(not PYPARDISO and not UMFPACK, reason="No fast solver available")
    def test_single_element_demand_array(self, basic_lca):
        """Test calculate_supply_arrays with a single element demand array."""
        # Create a minimal demand array
        demand = np.zeros(len(basic_lca.dicts.product))
        demand[basic_lca.dicts.product[1]] = 1.0

        result = basic_lca.calculate_supply_arrays([demand])

        # Check result shape
        assert result.shape == (basic_lca.technosphere_matrix.shape[0], 1)
        assert result.shape[0] == basic_lca.technosphere_matrix.shape[0]
