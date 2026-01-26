import pytest
from pydantic import ValidationError

from bw2calc.errors import InconsistentLCIA
from bw2calc.method_config import MethodConfig


def test_method_config_valid():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
    }
    assert MethodConfig(**data)

    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("norm", "standard"): [("foo", "a"), ("foo", "b")]},
    }
    assert MethodConfig(**data)

    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("norm", "standard"): [("foo", "a"), ("foo", "b")]},
        "weightings": {("weighting",): [("norm", "standard")]},
    }
    assert MethodConfig(**data)


def test_method_config_len_one_tuples_valid():
    data = {
        "impact_categories": [("a",), ("b",)],
    }
    assert MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), ("b",)]},
    }
    assert MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), ("b",)]},
        "weightings": {("weighting",): [("norm",)]},
    }
    assert MethodConfig(**data)


def test_method_config_weighting_can_refer_to_impact_category():
    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), ("b",)]},
        "weightings": {("weighting",): [("a",), ("b",)]},
    }
    assert MethodConfig(**data)


def test_method_config_weighting_can_refer_to_normalization():
    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), ("b",)]},
        "weightings": {("weighting",): [("norm",)]},
    }
    assert MethodConfig(**data)


def test_method_config_wrong_tuple_types():
    data = {
        "impact_categories": [("a",), (1,)],
    }
    with pytest.raises(ValidationError):
        MethodConfig(**data)

    data = {
        "impact_categories": [("a",), 1],
    }
    with pytest.raises(ValidationError):
        MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), (1,)]},
    }
    with pytest.raises(ValidationError):
        MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [(1,), ("b",)]},
    }
    with pytest.raises(ValidationError):
        MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), ("b",)]},
        "weightings": {("norm",): (1,)},
    }
    with pytest.raises(ValidationError):
        MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), ("b",)]},
        "weightings": {("norm",): 1},
    }
    with pytest.raises(ValidationError):
        MethodConfig(**data)


def test_method_config_missing_normalization_reference():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("norm", "standard"): [("foo", "a"), ("foo", "b"), ("foo", "c")]},
    }
    with pytest.raises(ValueError):
        MethodConfig(**data)


def test_method_config_normalization_overlaps_impact_categories():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("foo", "a"): [("foo", "a"), ("foo", "b")]},
    }
    with pytest.raises(ValueError):
        MethodConfig(**data)


def test_method_config_weighting_overlaps_impact_categories():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("normalization",): [("foo", "a"), ("foo", "b")]},
        "weightings": {("foo", "a"): [("foo", "a"), ("foo", "b")]},
    }
    with pytest.raises(ValueError):
        MethodConfig(**data)


def test_method_config_weighting_overlaps_normalizations():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("normalization",): [("foo", "a"), ("foo", "b")]},
        "weightings": {("normalization",): [("normalization",)]},
    }
    with pytest.raises(ValueError):
        MethodConfig(**data)


def test_method_config_weighting_missing_reference():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("normalization",): [("foo", "a"), ("foo", "b")]},
        "weightings": {("normalization",): [("foo", "c"), ("foo", "a"), ("foo", "b")]},
    }
    with pytest.raises(ValueError):
        MethodConfig(**data)


def test_method_config_missing_ic_for_weightings():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("normalization",): [("foo", "a")]},
    }
    with pytest.raises(InconsistentLCIA):
        MethodConfig(**data)


def test_method_config_weighting_mixed_references():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("normalization",): [("foo", "a"), ("foo", "b")]},
        "weightings": {("weighting",): [("normalization",), ("foo", "a"), ("foo", "b")]},
    }
    with pytest.raises(InconsistentLCIA):
        MethodConfig(**data)


def test_method_config_weighting_missing_ic():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "weightings": {("weighting",): [("foo", "b")]},
    }
    with pytest.raises(InconsistentLCIA):
        MethodConfig(**data)


def test_method_config_weighting_missing_normalization():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {
            ("normalization", "a"): [("foo", "a")],
            ("normalization", "b"): [("foo", "b")],
        },
        "weightings": {("weighting",): [("normalization", "a")]},
    }
    with pytest.raises(InconsistentLCIA):
        MethodConfig(**data)


def test_method_config_string_impact_categories():
    """Test that strings in impact_categories remain as strings."""
    data = {
        "impact_categories": ["a", "b"],
    }
    config = MethodConfig(**data)
    assert config.impact_categories == ["a", "b"]

    data = {
        "impact_categories": [("foo", "a"), "b"],
    }
    config = MethodConfig(**data)
    assert config.impact_categories == [("foo", "a"), "b"]


def test_method_config_string_normalizations():
    """Test that strings in normalizations keys and values remain as strings."""
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {
            "norm": [("foo", "a"), ("foo", "b")],
        },
    }
    config = MethodConfig(**data)
    assert "norm" in config.normalizations
    assert config.normalizations["norm"] == [("foo", "a"), ("foo", "b")]

    data = {
        "impact_categories": ["a", "b"],
        "normalizations": {
            ("norm", "standard"): ["a", "b"],
        },
    }
    config = MethodConfig(**data)
    assert ("norm", "standard") in config.normalizations
    assert config.normalizations[("norm", "standard")] == ["a", "b"]


def test_method_config_string_weightings():
    """Test that strings in weightings keys and values remain as strings."""
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "weightings": {
            "weighting": [("foo", "a"), ("foo", "b")],
        },
    }
    config = MethodConfig(**data)
    assert "weighting" in config.weightings
    assert config.weightings["weighting"] == [("foo", "a"), ("foo", "b")]

    data = {
        "impact_categories": ["a", "b"],
        "normalizations": {"norm": ["a", "b"]},
        "weightings": {
            "weighting": ["norm"],
        },
    }
    config = MethodConfig(**data)
    assert "weighting" in config.weightings
    assert config.weightings["weighting"] == ["norm"]


def test_method_config_mixed_strings_and_tuples():
    """Test that mixing strings and tuples works correctly."""
    data = {
        "impact_categories": [("foo", "a"), "b"],
        "normalizations": {
            ("norm", "standard"): [("foo", "a"), "b"],
            "norm2": ["b"],
        },
        "weightings": {
            ("weighting", "1"): ["norm2"],
            "weighting2": [("norm", "standard")],
        },
    }
    config = MethodConfig(**data)
    assert config.impact_categories == [("foo", "a"), "b"]
    assert ("norm", "standard") in config.normalizations
    assert "norm2" in config.normalizations
    assert config.normalizations[("norm", "standard")] == [("foo", "a"), "b"]
    assert config.normalizations["norm2"] == ["b"]
    assert ("weighting", "1") in config.weightings
    assert "weighting2" in config.weightings
    assert config.weightings[("weighting", "1")] == ["norm2"]
    assert config.weightings["weighting2"] == [("norm", "standard")]


def test_method_config_string_validation_still_works():
    """Test that validation errors still work correctly with strings."""
    # Missing normalization reference
    data = {
        "impact_categories": ["a", "b"],
        "normalizations": {"norm": ["a", "b", "c"]},
    }
    with pytest.raises(ValueError):
        MethodConfig(**data)

    # Overlapping identifiers
    data = {
        "impact_categories": ["a", "b"],
        "normalizations": {"a": ["a", "b"]},
    }
    with pytest.raises(ValueError):
        MethodConfig(**data)

    # Missing weighting reference
    data = {
        "impact_categories": ["a", "b"],
        "normalizations": {"norm": ["a", "b"]},
        "weightings": {"weighting": ["c"]},
    }
    with pytest.raises(ValueError):
        MethodConfig(**data)
