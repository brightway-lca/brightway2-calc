from bw_processing import Datapackage
from matrix_utils import MappedMatrix
from scipy import sparse
from typing import Union, Sequence, Any, Callable
import numpy as np
from .errors import MultipleValues


class SingleValueDiagonalMatrix(MappedMatrix):
    """A scipy sparse matrix handler which takes in ``bw_processing`` data packages. Row and column ids are mapped to matrix indices, and a matrix is constructed.

    `indexer_override` allows for custom indexer behaviour. Indexers should follow a simple API: they must support `.__next__()`, and have the attribute `.index`, which returns an integer.

    `custom_filter` allows you to remove some data based on their indices. It is applied to all resource groups. If you need more fine-grained control, process the matrix after construction/iteration. `custom_filter` should take the indices array as an input, and return a Numpy boolean array with the same length as the indices array.

    Args:

        * packages: A list of Ddatapackage objects.
        * matrix: The string identifying the matrix to be built.
        * use_vectors: Flag to use vector data from datapackages
        * use_arrays: Flag to use array data from datapackages
        * use_distributions: Flag to use `stats_arrays` distribution data from datapackages
        * row_mapper: Optional instance of `ArrayMapper`. Used when matrices must align.
        * col_mapper: Optional instance of `ArrayMapper`. Used when matrices must align.
        * seed_override: Optional integer. Overrides the RNG seed given in the datapackage, if any.
        * indexer_override: Parameter for custom indexers. See above.
        * diagonal: If True, only use the `row` indices to build a diagonal matrix.
        * custom_filter: Callable for function to filter data based on `indices` values. See above.
        * empty_ok: If False, raise `AllArraysEmpty` if the matrix would be empty

    """

    def __init__(
        self,
        *,
        packages: Sequence[Datapackage],
        matrix: str,
        dimension: int,
        use_vectors: bool = True,
        use_arrays: bool = True,
        use_distributions: bool = False,
        seed_override: Union[int, None] = None,
        indexer_override: Any = None,
        custom_filter: Union[Callable, None] = None,
    ):
        self.dimension = dimension

        super().__init__(packages=packages, matrix=matrix, use_vectors=use_vectors, use_arrays=use_arrays, use_distributions=use_distributions, seed_override=seed_override, indexer_override=indexer_override, diagonal=True, custom_filter=custom_filter)

        if self.raw_data.shape != (1,):
            raise MultipleValues("Multiple ({}) numerical values found, but only one single numerical value is allowed. Data packages:\n\t{}".format(len(self.raw_data), "\n\t".join([str(x) for x in self.packages])))

    def rebuild_matrix(self):
        self.matrix = sparse.coo_matrix(
            (
                np.ones(self.dimension),
                (np.arange(self.dimension), np.arange(self.dimension)),
            ),
            (self.dimension, self.dimension),
            dtype=np.float64,
        ).tocsr()

        self.raw_data = np.hstack([group.calculate()[2] for group in self.groups])
        self.matrix *= self.raw_data[0]
