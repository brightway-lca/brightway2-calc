"""
Pytest configuration and fixtures for bw2calc tests.

This module provides fixtures for handling solver availability and testing
different solver configurations.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def check_solver_availability():
    """Check which solvers are available."""
    pypardiso_available = False
    umfpack_available = False

    try:
        import pypardiso  # noqa: F401

        pypardiso_available = True
    except ImportError:
        pass

    try:
        import scikits.umfpack  # noqa: F401

        umfpack_available = True
    except ImportError:
        pass

    return pypardiso_available, umfpack_available


@pytest.fixture(scope="session")
def solver_availability():
    """Session-scoped fixture that checks solver availability."""
    return check_solver_availability()


@pytest.fixture
def pypardiso_available(solver_availability):
    """Fixture indicating if pypardiso is available."""
    return solver_availability[0]


@pytest.fixture
def umfpack_available(solver_availability):
    """Fixture indicating if scikits.umfpack is available."""
    return solver_availability[1]


@pytest.fixture
def no_solvers_available():
    """Fixture that monkey-patches both solvers to be unavailable."""
    with patch.dict(
        "sys.modules",
        {
            "pypardiso": None,
            "scikits.umfpack": None,
            "scikits": None,
        },
    ):
        # Patch the bw2calc module's solver flags in all places they're used
        # Note: Only patch attributes that actually exist in each module
        with (
            patch("bw2calc.__init__.PYPARDISO", False),
            patch("bw2calc.__init__.UMFPACK", False),
            patch("bw2calc.fast_scores.PYPARDISO", False),
            patch("bw2calc.fast_scores.UMFPACK", False),
            patch("bw2calc.fast_supply_arrays.PYPARDISO", False),
            patch("bw2calc.fast_supply_arrays.UMFPACK", False),
            patch("bw2calc.lca_base.PYPARDISO", False),
            patch("bw2calc.lca.PYPARDISO", False),
            patch("bw2calc.multi_lca.PYPARDISO", False),
        ):
            yield


@pytest.fixture
def only_pypardiso_available():
    """Fixture that makes only pypardiso available."""
    with patch.dict(
        "sys.modules",
        {
            "scikits.umfpack": None,
            "scikits": None,
        },
    ):
        # Mock pypardiso module
        mock_pypardiso = Mock()
        mock_pypardiso.pardiso_wrapper = Mock()

        with (
            patch.dict("sys.modules", {"pypardiso": mock_pypardiso}),
            patch("bw2calc.fast_scores.PYPARDISO", True),
            patch("bw2calc.fast_scores.UMFPACK", False),
        ):
            yield


@pytest.fixture
def only_umfpack_available():
    """Fixture that makes only scikits.umfpack available."""
    with patch.dict(
        "sys.modules",
        {
            "pypardiso": None,
        },
    ):
        # Mock scikits.umfpack module
        mock_scikits = Mock()
        mock_scikits.umfpack = Mock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "scikits": mock_scikits,
                    "scikits.umfpack": mock_scikits.umfpack,
                },
            ),
            patch("bw2calc.fast_scores.PYPARDISO", False),
            patch("bw2calc.fast_scores.UMFPACK", True),
        ):
            yield


@pytest.fixture
def both_solvers_available():
    """Fixture that makes both solvers available."""
    # Mock pypardiso module
    mock_pypardiso = Mock()
    mock_pypardiso.pardiso_wrapper = Mock()

    # Mock scikits.umfpack module
    mock_scikits = Mock()
    mock_scikits.umfpack = Mock()

    with (
        patch.dict(
            "sys.modules",
            {
                "pypardiso": mock_pypardiso,
                "scikits": mock_scikits,
                "scikits.umfpack": mock_scikits.umfpack,
            },
        ),
        patch("bw2calc.fast_scores.PYPARDISO", True),
        patch("bw2calc.fast_scores.UMFPACK", True),
    ):
        yield


@pytest.fixture
def mock_pypardiso_solver():
    """Fixture providing a mock PyPardisoSolver."""

    def create_mock_solver():
        mock_solver = Mock()
        mock_solver.factorized = False
        mock_solver.phase = None

        def factorize(matrix):
            mock_solver.factorized = True

        def set_phase(phase):
            mock_solver.phase = phase

        def _check_b(matrix, b):
            return b

        def _call_pardiso(matrix, b):
            # Return a mock solution
            import numpy as np

            return np.ones((matrix.shape[0], b.shape[1]))

        mock_solver.factorize = factorize
        mock_solver.set_phase = set_phase
        mock_solver._check_b = _check_b
        mock_solver._call_pardiso = _call_pardiso

        return mock_solver

    return create_mock_solver


@pytest.fixture
def mock_umfpack_solver():
    """Fixture providing a mock UMFPACK solver."""

    def mock_factorized(matrix):
        def solver(b):
            import numpy as np

            return np.ones(matrix.shape[0])

        return solver

    return mock_factorized


# Pytest markers for different solver configurations
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "pypardiso: mark test as requiring pypardiso solver")
    config.addinivalue_line("markers", "umfpack: mark test as requiring scikits.umfpack solver")
    config.addinivalue_line("markers", "no_solvers: mark test as requiring no solvers available")
    config.addinivalue_line(
        "markers", "solver_agnostic: mark test as working with any solver configuration"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on solver availability."""
    pypardiso_available, umfpack_available = check_solver_availability()

    for item in items:
        # Skip pypardiso tests if pypardiso is not available
        if item.get_closest_marker("pypardiso") and not pypardiso_available:
            skip_marker = pytest.mark.skip(reason="pypardiso not available")
            item.add_marker(skip_marker)

        # Skip umfpack tests if umfpack is not available
        if item.get_closest_marker("umfpack") and not umfpack_available:
            skip_marker = pytest.mark.skip(reason="scikits.umfpack not available")
            item.add_marker(skip_marker)

        # Skip tests that require no solvers if any solver is available
        if item.get_closest_marker("no_solvers") and (pypardiso_available or umfpack_available):
            skip_marker = pytest.mark.skip(reason="solvers are available")
            item.add_marker(skip_marker)


# Convenience fixtures for common test scenarios
@pytest.fixture
def solver_config(request):
    """Parametrized fixture for testing different solver configurations."""
    config = request.param
    if config == "no_solvers":
        return pytest.fixture(no_solvers_available)
    elif config == "pypardiso_only":
        return pytest.fixture(only_pypardiso_available)
    elif config == "umfpack_only":
        return pytest.fixture(only_umfpack_available)
    elif config == "both_solvers":
        return pytest.fixture(both_solvers_available)
    else:
        raise ValueError(f"Unknown solver config: {config}")


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def basic_test_data(fixture_dir):
    """Basic test data for FastScoresOnlyMultiLCA tests."""

    from bw2calc.utils import get_datapackage

    return {
        "dps": [
            get_datapackage(fixture_dir / "multi_lca_simple_1.zip"),
            get_datapackage(fixture_dir / "multi_lca_simple_2.zip"),
            get_datapackage(fixture_dir / "multi_lca_simple_3.zip"),
            get_datapackage(fixture_dir / "multi_lca_simple_4.zip"),
            get_datapackage(fixture_dir / "multi_lca_simple_5.zip"),
        ],
        "config": {
            "impact_categories": [
                ("first", "category"),
                ("second", "category"),
            ]
        },
        "demands": {
            "γ": {100: 1},
            "ε": {103: 2},
            "ζ": {105: 3},
        },
    }
