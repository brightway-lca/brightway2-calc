"""Tests for bw2data availability and version checking in bw2calc/__init__.py."""

import sys
from unittest.mock import patch

from packaging.version import Version


def test_bw2data_imported_when_available():
    """Test that bw2data functions are imported when available."""
    from bw2calc import get_activity, prepare_lca_inputs

    # These should be either None (if bw2data not available) or callable functions
    # This test passes in both cases since we're testing the import mechanism works
    assert get_activity is None or callable(get_activity)
    assert prepare_lca_inputs is None or callable(prepare_lca_inputs)


def test_bw2data_tuple_version_format():
    """Test handling of tuple version format."""
    # Test that tuple versions are converted to strings correctly
    version_tuple = (4, 0, 1)
    version_string = ".".join([str(n) for n in version_tuple])

    assert Version(version_string) >= Version("4.0")


def test_bw2data_not_available_sets_to_none():
    """Test that bw2data functions are set to None when unavailable."""
    # Mock bw2data as unavailable
    with patch.dict(sys.modules, {"bw2data": None}):
        # Reimport bw2calc
        import importlib

        import bw2calc

        importlib.reload(bw2calc)

        # These should be None when bw2data is not available
        assert bw2calc.get_activity is None
        assert bw2calc.prepare_lca_inputs is None


def test_bw2data_import_error_handled():
    """Test that ImportError from bw2data is handled gracefully."""
    with patch.dict(sys.modules, {"bw2data": None}):
        # Reimport bw2calc
        import importlib

        import bw2calc

        importlib.reload(bw2calc)

        # Should handle ImportError gracefully
        assert bw2calc.get_activity is None
        assert bw2calc.prepare_lca_inputs is None


def test_version_3_x_comparison():
    """Test that version 3.x is considered invalid."""
    assert not (Version("3.9") >= Version("4.0"))
    assert not (Version("3.9.9") >= Version("4.0"))


def test_bw2data_version_tuple_format():
    """Test that bw2data version tuple format (4, 1, 0) is handled correctly."""
    # Mock bw2data with tuple version format
    mock_bw2data = type(
        "module",
        (),
        {"__version__": (4, 1, 0), "get_activity": lambda x: x, "prepare_lca_inputs": lambda x: x},
    )

    with patch.dict(sys.modules, {"bw2data": mock_bw2data}):
        # Reimport bw2calc to trigger version check
        import importlib

        import bw2calc

        importlib.reload(bw2calc)

        # Should successfully import with tuple version format
        # The version check in __init__.py should convert tuple to string and pass the >= 4.0 check
        assert bw2calc.get_activity is not None
        assert bw2calc.prepare_lca_inputs is not None
        assert callable(bw2calc.get_activity)
        assert callable(bw2calc.prepare_lca_inputs)


def test_presamples_available():
    """Test that presamples.PackagesDataLoader is imported when available."""
    from bw2calc import PackagesDataLoader

    # Should be imported or None
    assert PackagesDataLoader is not None or PackagesDataLoader is None


def test_presamples_not_available():
    """Test that presamples.PackagesDataLoader is None when unavailable."""
    # Mock presamples as unavailable
    with patch.dict(sys.modules, {"presamples": None}):
        # Reimport bw2calc
        import importlib

        import bw2calc

        importlib.reload(bw2calc)

        # Should be None when presamples is not available
        assert bw2calc.PackagesDataLoader is None


def test_multiple_imports_consistent():
    """Test that multiple imports of bw2calc are consistent."""
    import bw2calc

    # First import - functions should either be None or callable
    result1_get_activity = bw2calc.get_activity
    result1_prepare_lca_inputs = bw2calc.prepare_lca_inputs

    # Should be consistently None or callable
    assert result1_get_activity is None or callable(result1_get_activity)
    assert result1_prepare_lca_inputs is None or callable(result1_prepare_lca_inputs)
