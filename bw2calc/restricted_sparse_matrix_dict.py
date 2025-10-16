from typing import Any

from matrix_utils import SparseMatrixDict
from pydantic import BaseModel


class RestrictionsValidator(BaseModel):
    restrictions: dict[tuple[str, ...], list[tuple[str, ...]]]


class RestrictedSparseMatrixDict(SparseMatrixDict):
    def __init__(self, restrictions: dict, *args, **kwargs):
        """Like SparseMatrixDict, but follows `restrictions` on what can be multiplied.

        Only for use with normalization and weighting."""
        super().__init__(*args, **kwargs)
        RestrictionsValidator(restrictions=restrictions)
        self._restrictions = restrictions

    def _get_first_element(self, elem: Any) -> tuple:
        """Get the first LCIA key from `elem`.

        The keys can have the form `(("some", "lcia"), "functional-unit-id")` or
        `("some", "lcia")."""
        if isinstance(elem[0], tuple):
            return elem[0]
        else:
            assert isinstance(elem, tuple), f"Wrong type: {type(elem)} should be tuple"
            return elem

    def _concatenate(self, a: tuple, b: tuple) -> tuple:
        """Combine `a` and `b` while unwrapping `b`, if necessary."""
        if isinstance(b[0], tuple):
            return (a, *b)
        else:
            return (a, b)

    def __matmul__(self, other: Any) -> SparseMatrixDict:
        """Define logic for `@` matrix multiplication operator.

        Note that the sparse matrix dict must come first, i.e. `self @ other`.
        """
        if isinstance(other, (SparseMatrixDict, RestrictedSparseMatrixDict)):
            return SparseMatrixDict(
                {
                    self._concatenate(a, b): c @ d
                    for a, c in self.items()
                    for b, d in other.items()
                    if self._get_first_element(b) in self._restrictions[a]
                }
            )
        else:
            return super().__matmul__(other)
