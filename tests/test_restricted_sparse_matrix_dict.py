import pytest
from matrix_utils import SparseMatrixDict
from pydantic import ValidationError

from bw2calc.restricted_sparse_matrix_dict import RestrictedSparseMatrixDict, RestrictionsValidator


class Dummy:
    def __init__(self, a):
        self.a = a

    def __matmul__(self, other):
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
