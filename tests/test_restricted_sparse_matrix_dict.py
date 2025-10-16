import pytest
from matrix_utils import SparseMatrixDict
from pydantic import ValidationError

from bw2calc.restricted_sparse_matrix_dict import RestrictedSparseMatrixDict, RestrictionsValidator


class Dummy:
    def __init__(self, a):
        self.a = a

    def __matmul__(self, other):
        if isinstance(other, Dummy):
            return self.a + other.a
        return self.a + other


def test_restricted_sparse_matrix_dict():
    smd = SparseMatrixDict({(("one",), "foo"): 1, (("two",), "bar"): 2})
    rsmd = RestrictedSparseMatrixDict(
        {("seven",): [("one",)], ("eight",): [("two",)]},
        {("seven",): Dummy(7), ("eight",): Dummy(8)},
    )

    result = rsmd @ smd
    assert isinstance(result, SparseMatrixDict)
    assert len(result) == 2
    assert result[(("seven",), ("one",), "foo")] == 8
    assert result[(("eight",), ("two",), "bar")] == 10


def test_restrictions_validator():
    assert RestrictionsValidator(restrictions={("seven",): [("one",)], ("eight",): [("two",)]})
    with pytest.raises(ValidationError):
        RestrictionsValidator(restrictions={"seven": [("one",)], ("eight",): [("two",)]})


# Test the updated RestrictedSparseMatrixDict functionality


def test_get_first_element_with_nested_tuple():
    """Test _get_first_element with nested tuple keys."""
    rsmd = RestrictedSparseMatrixDict({("seven",): [("one",)]}, {("seven",): Dummy(7)})

    # Test with nested tuple key
    nested_key = (("some", "lcia"), "functional-unit-id")
    result = rsmd._get_first_element(nested_key)
    assert result == ("some", "lcia")


def test_get_first_element_with_simple_tuple():
    """Test _get_first_element with simple tuple keys."""
    rsmd = RestrictedSparseMatrixDict({("seven",): [("one",)]}, {("seven",): Dummy(7)})

    # Test with simple tuple key
    simple_key = ("some", "lcia")
    result = rsmd._get_first_element(simple_key)
    assert result == ("some", "lcia")


def test_get_first_element_invalid_type():
    """Test _get_first_element with invalid type."""
    rsmd = RestrictedSparseMatrixDict({("seven",): [("one",)]}, {("seven",): Dummy(7)})

    # Test with invalid type
    with pytest.raises(AssertionError, match="Wrong type: <class 'str'> should be tuple"):
        rsmd._get_first_element("invalid")


def test_concatenate_with_nested_tuple():
    """Test _concatenate with nested tuple in second argument."""
    rsmd = RestrictedSparseMatrixDict({("seven",): [("one",)]}, {("seven",): Dummy(7)})

    a = ("normalization", "key")
    b = (("some", "lcia"), ("a", "b"), "functional-unit-id")
    result = rsmd._concatenate(a, b)
    assert result == (("normalization", "key"), ("some", "lcia"), ("a", "b"), "functional-unit-id")


def test_concatenate_with_simple_tuple():
    """Test _concatenate with simple tuple in second argument."""
    rsmd = RestrictedSparseMatrixDict({("seven",): [("one",)]}, {("seven",): Dummy(7)})

    a = ("normalization", "key")
    b = ("some", "lcia")
    result = rsmd._concatenate(a, b)
    assert result == (("normalization", "key"), ("some", "lcia"))


def test_matmul_with_restricted_sparse_matrix_dict():
    """Test matrix multiplication with another RestrictedSparseMatrixDict."""
    # Create two RestrictedSparseMatrixDict instances
    restrictions1 = {
        ("norm", "key1"): [("first", "category")],
        ("norm", "key2"): [("second", "category")],
    }
    rsmd1 = RestrictedSparseMatrixDict(
        restrictions1, {("norm", "key1"): Dummy(10), ("norm", "key2"): Dummy(20)}
    )

    restrictions2 = {
        ("first", "category"): [("process1",)],
        ("second", "category"): [("process2",)],
    }
    rsmd2 = RestrictedSparseMatrixDict(
        restrictions2, {("first", "category"): Dummy(5), ("second", "category"): Dummy(15)}
    )

    result = rsmd1 @ rsmd2

    assert isinstance(result, SparseMatrixDict)
    assert len(result) == 2
    # Check that the correct combinations were created based on restrictions
    assert (("norm", "key1"), ("first", "category")) in result
    assert (("norm", "key2"), ("second", "category")) in result
    # Check that the multiplication was performed correctly
    assert result[(("norm", "key1"), ("first", "category"))] == 15  # 10 + 5
    assert result[(("norm", "key2"), ("second", "category"))] == 35  # 20 + 15


def test_matmul_with_sparse_matrix_dict():
    """Test matrix multiplication with SparseMatrixDict."""
    rsmd = RestrictedSparseMatrixDict(
        {("norm", "key1"): [("first", "category")]}, {("norm", "key1"): Dummy(10)}
    )

    smd = SparseMatrixDict(
        {
            (("first", "category"), "process1"): 5,
            (("second", "category"), "process2"): 15,  # This should be filtered out
        }
    )

    result = rsmd @ smd

    assert isinstance(result, SparseMatrixDict)
    assert len(result) == 1
    # Only the allowed combination should be present
    assert (("norm", "key1"), ("first", "category"), "process1") in result
    assert result[(("norm", "key1"), ("first", "category"), "process1")] == 15  # 10 + 5


def test_matmul_restrictions_filtering():
    """Test that restrictions properly filter out disallowed combinations."""
    rsmd = RestrictedSparseMatrixDict(
        {("norm", "key1"): [("first", "category")]},  # Only allows ("first", "category")
        {("norm", "key1"): Dummy(10)},
    )

    smd = SparseMatrixDict(
        {
            (("first", "category"), "process1"): 5,  # Should be allowed
            (("second", "category"), "process2"): 15,  # Should be filtered out
            (("third", "category"), "process3"): 25,  # Should be filtered out
        }
    )

    result = rsmd @ smd

    assert isinstance(result, SparseMatrixDict)
    assert len(result) == 1
    # Only the allowed combination should be present
    assert (("norm", "key1"), ("first", "category"), "process1") in result
    assert result[(("norm", "key1"), ("first", "category"), "process1")] == 15  # 10 + 5


def test_matmul_with_non_matrix_dict():
    """Test matrix multiplication with non-SparseMatrixDict type."""
    rsmd = RestrictedSparseMatrixDict(
        {("norm", "key1"): [("first", "category")]}, {("norm", "key1"): Dummy(10)}
    )

    # Test with a non-SparseMatrixDict type
    other = "not a matrix dict"

    # This should call the parent class method
    with pytest.raises(TypeError):
        rsmd @ other


def test_matmul_empty_restrictions():
    """Test matrix multiplication with empty restrictions."""
    rsmd = RestrictedSparseMatrixDict({}, {("norm", "key1"): Dummy(10)})  # Empty restrictions

    smd = SparseMatrixDict(
        {(("first", "category"), "process1"): 5, (("second", "category"), "process2"): 15}
    )

    with pytest.raises(KeyError):
        rsmd @ smd


def test_matmul_multiple_restrictions():
    """Test matrix multiplication with multiple restrictions."""
    rsmd = RestrictedSparseMatrixDict(
        {
            ("norm", "key1"): [("first", "category"), ("second", "category")],
            ("norm", "key2"): [("third", "category")],
        },
        {("norm", "key1"): Dummy(10), ("norm", "key2"): Dummy(20)},
    )

    smd = SparseMatrixDict(
        {
            (("first", "category"), "process1"): 5,
            (("second", "category"), "process2"): 15,
            (("third", "category"), "process3"): 25,
            (("fourth", "category"), "process4"): 35,  # Should be filtered out
        }
    )

    result = rsmd @ smd

    assert isinstance(result, SparseMatrixDict)
    assert len(result) == 3
    # Check all allowed combinations are present
    assert (("norm", "key1"), ("first", "category"), "process1") in result
    assert (("norm", "key1"), ("second", "category"), "process2") in result
    assert (("norm", "key2"), ("third", "category"), "process3") in result
    # Check values
    assert result[(("norm", "key1"), ("first", "category"), "process1")] == 15  # 10 + 5
    assert result[(("norm", "key1"), ("second", "category"), "process2")] == 25  # 10 + 15
    assert result[(("norm", "key2"), ("third", "category"), "process3")] == 45  # 20 + 25


def test_initialization_with_validation():
    """Test that initialization validates restrictions using RestrictionsValidator."""
    # Valid restrictions
    valid_restrictions = {("norm", "key1"): [("first", "category")]}
    rsmd = RestrictedSparseMatrixDict(valid_restrictions, {("norm", "key1"): Dummy(10)})
    assert rsmd._restrictions == valid_restrictions

    # Invalid restrictions should raise ValidationError
    with pytest.raises(ValidationError):
        RestrictedSparseMatrixDict(
            {"invalid": [("first", "category")]},  # String key instead of tuple
            {("norm", "key1"): Dummy(10)},
        )


def test_restrictions_attribute():
    """Test that restrictions are properly stored as private attribute."""
    restrictions = {("norm", "key1"): [("first", "category")]}
    rsmd = RestrictedSparseMatrixDict(restrictions, {("norm", "key1"): Dummy(10)})

    assert hasattr(rsmd, "_restrictions")
    assert rsmd._restrictions == restrictions
