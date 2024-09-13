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

    def __matmul__(self, other: Any) -> SparseMatrixDict:
        """Define logic for `@` matrix multiplication operator.

        Note that the sparse matrix dict must come first, i.e. `self @ other`.
        """
        if isinstance(other, (SparseMatrixDict, RestrictedSparseMatrixDict)):
            return SparseMatrixDict(
                {
                    (a, *b): c @ d
                    for a, c in self.items()
                    for b, d in other.items()
                    if b[0] in self._restrictions[a]
                }
            )
        else:
            return super().__matmul__(other)
