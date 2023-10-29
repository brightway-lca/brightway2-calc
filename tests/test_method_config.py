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
