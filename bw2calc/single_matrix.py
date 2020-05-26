# -*- coding: utf-8 -*-
from io import BytesIO
from .errors import (
    NonsquareTechnosphere,
    OutsideTechnosphere,
)
from .log_utils import create_logger
from .matrices import SingleMatrixBuilder, MatrixBuilder
from .utils import (
    global_index,
    clean_databases,
    get_filepaths,
    load_arrays,
    mapping,
)
from collections.abc import Mapping
import json
import logging
import numpy as np
import tarfile
import warnings

try:
    from pypardiso import factorized, spsolve
except ImportError:
    from scipy.sparse.linalg import factorized, spsolve
try:
    from presamples import PackagesDataLoader
except ImportError:
    PackagesDataLoader = None


class SingleMatrixLCA:
    """An LCA which puts everything into one matrix.

    Comes with advantages and disadvantages, and designed exclusively for `BONSAI <https://bonsai.uno/>`__ via `beebee <https://github.com/BONSAMURAIS/beebee/>`__."""

    def __init__(
        self,
        demand,
        data_filepath,
        log_config=None,
        presamples=None,
        seed=None,
        override_presamples_seed=False,
    ):
        """Create a new single-matrix LCA calculation.

        Args:
            * *demand* (dict): The demand or functional unit. Needs to be a dictionary to indicate amounts, e.g. ``{("my database", "my process"): 2.5}``.

        Returns:
            A new ``SingleMatrixLCA`` object

        """
        if not isinstance(demand, Mapping):
            raise ValueError("Demand must be a dictionary")

        if log_config:
            create_logger(**log_config)
        self.logger = logging.getLogger("bw2calc")

        self.demand = demand
        self.filepath = data_filepath
        self.seed = seed

        if presamples and PackagesDataLoader is None:
            warnings.warn("Skipping presamples; `presamples` not installed")
            self.presamples = None
        elif presamples:
            # Iterating over a `Campaign` object will also return the presample filepaths
            self.presamples = PackagesDataLoader(
                dirpaths=presamples,
                seed=self.seed if override_presamples_seed else None,
                lca=self,
            )
        else:
            self.presamples = None

        self.logger.info(
            "Created LCA object",
            extra={
                "demand": str(self.demand),
                "data_filepath": self.filepath,
                "presamples": str(self.presamples),
            },
        )

    def build_demand_array(self, demand=None):
        """Turn the demand dictionary into a *NumPy* array of correct size.

        Args:
            * *demand* (dict, optional): Demand dictionary. Optional, defaults to ``self.demand``.

        Returns:
            A 1-dimensional NumPy array

        """
        demand = demand or self.demand
        self.demand_array = np.zeros(len(self.row_dict))
        for key in demand:
            try:
                self.demand_array[self.row_dict[key]] = demand[key]
            except KeyError:
                if key in self.col_dict:
                    raise ValueError(
                        (
                            u"LCA can only be performed on products,"
                            u" not activities ({} is the wrong dimension)"
                        ).format(key)
                    )
                else:
                    raise OutsideTechnosphere(
                        "Can't find key {} in product dictionary".format(key)
                    )

    #########################
    ### Data manipulation ###
    #########################

    def fix_dictionaries(self, row_mapping, col_mapping):
        """Fix the row and column dictionaries from ``{integer: row/col index}`` to ``{label: row/col index}``."""
        self.row_dict = {
            label: self.row_dict[index] for label, index in row_mapping.items()
        }
        self.col_dict = {
            label: self.col_dict[index] for label, index in col_mapping.items()
        }

    def reverse_dict(self):
        """Construct reverse dicts from technosphere and biosphere row and col indices to activity_dict/product_dict/biosphere_dict keys.

        Returns:
            (reversed ``self.dicts.activity``, ``self.dicts.product`` and ``self.dicts.biosphere``)
        """
        rev_row = {v: k for k, v in self.row_dict.items()}
        rev_col = {v: k for k, v in self.col_dict.items()}
        return rev_row, rev_col

    ######################
    ### Data retrieval ###
    ######################

    def load_beebee_data(self, builder=SingleMatrixBuilder):
        """Load ``beebee`` export data package.

        This is a compressed file which contains:

        * A `stats_arrays <https://bitbucket.org/cmutel/stats_arrays>`__ file used to create the single matrix.
        * A mapping dictionary from meaningful labels to the integer row ids
        * A mapping dictionary from meaningful labels to the integer column ids
        * A mapping dictionary from ``{"method URI": {labels}}`` which allows for LCIA sums

        """
        with tarfile.open(self.filepath, "r:bz2") as f:

            # Hack needed because of https://github.com/numpy/numpy/issues/7989
            array_file = BytesIO()
            array_file.write(f.extractfile("array.npy").read())
            array_file.seek(0)

            self.params, self.row_dict, self.col_dict, self.matrix = builder.build(
                array_file
            )
            row_mapping = json.load(f.extractfile("row.mapping"))
            col_mapping = json.load(f.extractfile("col.mapping"))
            categories = json.load(f.extractfile("categories.mapping"))
        if len(self.row_dict) != len(self.col_dict):
            raise NonsquareTechnosphere(
                ("Single matrix is not square: {} columns and {} rows.").format(
                    len(self.row_dict), len(self.col_dict)
                )
            )
        self.fix_dictionaries(row_mapping, col_mapping)

        self.category_index_arrays = {
            name: np.array(
                [(self.row_dict[label], 1) for label in labels],
                dtype=[("INDEX", np.uint32), ("CONSTANT", np.uint32)],
            )
            for name, labels in categories.items()
        }

        # Only need to index here for traditional LCA
        if self.presamples:
            self.presamples.index_arrays(self)
            self.presamples.update_matrices(matrices=("matrix",))

    def decompose_technosphere(self):
        """
Factorize the technosphere matrix into lower and upper triangular matrices, :math:`A=LU`. Does not solve the linear system :math:`Ax=B`.

Doesn't return anything, but creates ``self.solver``.

.. warning:: Incorrect results could occur if a technosphere matrix was factorized, and then a new technosphere matrix was constructed, as ``self.solver`` would still be the factorized older technosphere matrix. You are responsible for deleting ``self.solver`` when doing these types of advanced calculations.

        """
        self.solver = factorized(self.matrix.tocsc())

    def solve_linear_system(self):
        """
Master solution function for linear system :math:`Ax=B`.

    To most numerical analysts, matrix inversion is a sin.

    -- Nicolas Higham, Accuracy and Stability of Numerical Algorithms, Society for Industrial and Applied Mathematics, Philadelphia, PA, USA, 2002, p. 260.

We use `UMFpack <http://www.cise.ufl.edu/research/sparse/umfpack/>`_, which is a very fast solver for sparse matrices.

If the technosphere matrix has already been factorized, then the decomposed technosphere (``self.solver``) is reused. Otherwise the calculation is redone completely.

        """
        if hasattr(self, "solver"):
            return self.solver(self.demand_array)
        else:
            return spsolve(self.matrix, self.demand_array)

    def lcia(self, *args, **kwargs):
        warnings.warn("LCIA does nothing in a SingleMatrixLCA")

    def calculate(self, factorize=False, builder=SingleMatrixBuilder):
        """Calculate an LCA score.

        Creates ``self.supply_array``, a vector of activities, flows, and characterization pathways which satisfy the demand.

        Creates ``self.scores``, a dictionary of ``{'LCIA identifier': LCA score}``.

        Create ``self.contributions``, a dictionary of ``{'LCIA identifier': []}``.

        Args:
            * *factorize* (bool, optional): Factorize the technosphere matrix. Makes additional calculations with the same technosphere matrix much faster. Default is ``False``; not useful is only doing one LCI calculation.
            * *builder* (``SingleMatrixBuilder`` object, optional): Custom matrix builders can be used to manipulate data in creative ways before building the matrices.

        Doesn't return anything.

        """
        self.load_beebee_data(builder)
        self.build_demand_array()
        if factorize:
            self.decompose_technosphere()
        self.calculate_scores()

    def calculate_scores(self):
        self.supply_array = self.solve_linear_system()
        self.scores, self.contributions = {}, {}
        for name, array in self.category_index_arrays.items():
            matrix = MatrixBuilder.build_diagonal_matrix(
                array, self.row_dict, "INDEX", data_label="CONSTANT"
            )
            self.scores[name] = float((self.matrix * matrix * self.supply_array).sum())

    def rebuild_matrix(self, vector):
        """Build a new technosphere matrix using the same row and column indices, but different values. Useful for Monte Carlo iteration or sensitivity analysis.

        Args:
            * *vector* (array): 1-dimensional NumPy array with length (# of technosphere parameters), in same order as ``self.tech_params``.

        Doesn't return anything, but overwrites ``self.technosphere_matrix``.

        """
        self.matrix = SingleMatrixBuilder.build_single_matrix(
            self.params, self.row_dict, self.col_dict, vector
        )

    def redo_calculate(self, demand=None):
        """Redo LCI with same databases but different demand.

        Args:
            * *demand* (dict): A demand dictionary.

        Doesn't return anything, but overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        .. warning:: If you want to redo the LCIA as well, use ``redo_lcia(demand)`` directly.

        """
        assert hasattr(self, "matrix"), "Must do `calculate` first"
        if demand is not None:
            self.build_demand_array(demand)
        self.calculate_scores()
        self.logger.info("Redoing LCI", extra={"demand": str(demand or self.demand)})
