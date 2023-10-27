import pytest
from pydantic import ValidationError

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


def test_method_config_weighting_can_refer_impact_category():
    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), ("b",)]},
        "weightings": {("weighting",): [("a",)]},
    }
    assert MethodConfig(**data)


def test_method_config_weighting_can_refer_normalization():
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
        assert MethodConfig(**data)

    data = {
        "impact_categories": [("a",), 1],
    }
    with pytest.raises(ValidationError):
        assert MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), (1,)]},
    }
    with pytest.raises(ValidationError):
        assert MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [(1,), ("b",)]},
    }
    with pytest.raises(ValidationError):
        assert MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), ("b",)]},
        "weightings": {("norm",): (1,)},
    }
    with pytest.raises(ValidationError):
        assert MethodConfig(**data)

    data = {
        "impact_categories": [("a",), ("b",)],
        "normalizations": {("norm",): [("a",), ("b",)]},
        "weightings": {("norm",): 1},
    }
    with pytest.raises(ValidationError):
        assert MethodConfig(**data)


def test_method_config_missing_normalization_reference():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("norm", "standard"): [("foo", "c")]},
    }
    with pytest.raises(ValueError):
        assert MethodConfig(**data)


def test_method_config_normalization_overlaps_impact_categories():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("foo", "a"): [("foo", "a")]},
    }
    with pytest.raises(ValueError):
        assert MethodConfig(**data)


def test_method_config_weighting_overlaps_impact_categories():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("normalization",): [("foo", "a")]},
        "weightings": {("foo", "a"): [("foo", "a")]},
    }
    with pytest.raises(ValueError):
        assert MethodConfig(**data)


def test_method_config_weighting_overlaps_normalizations():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("normalization",): [("foo", "a")]},
        "weightings": {("normalization",): [("normalization",)]},
    }
    with pytest.raises(ValueError):
        assert MethodConfig(**data)


def test_method_config_weighting_missing_reference():
    data = {
        "impact_categories": [("foo", "a"), ("foo", "b")],
        "normalizations": {("normalization",): [("foo", "a")]},
        "weightings": {("normalization",): [("foo", "c")]},
    }
    with pytest.raises(ValueError):
        assert MethodConfig(**data)
