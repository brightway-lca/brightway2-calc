# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from eight import *

from scipy.sparse.linalg import factorized, spsolve
from scipy import sparse
import numpy as np
from .errors import (
    NonsquareTechnosphere,
    OutsideTechnosphere,
)
from .matrices import MatrixBuilder
from .matrices import TechnosphereBiosphereMatrixBuilder as TBMBuilder
from .utils import (
    global_index,
    clean_databases,
    get_filepaths,
    load_arrays,
    mapping,
)
import copy
import numpy as np
try:
    import pandas
except ImportError:
    pandas = None


class LCA(object):
    """A static LCI or LCIA calculation.

    Following the general philosophy of Brightway2, and good software practices, there is a clear separation of concerns between retrieving and formatting data and doing an LCA. Building the necessary matrices is done with MatrixBuilder objects (:ref:`matrixbuilders`). The LCA class only does the LCA calculations themselves.

    """

    #############
    ### Setup ###
    #############

    def __init__(self, demand, method=None, weighting=None,
            normalization=None, database_filepath=None):
        """Create a new LCA calculation.

        Args:
            * *demand* (dict): The demand or functional unit. Needs to be a dictionary to indicate amounts, e.g. ``{("my database", "my process"): 2.5}``.
            * *method* (tuple, optional): LCIA Method tuple, e.g. ``("My", "great", "LCIA", "method")``. Can be omitted if only interested in calculating the life cycle inventory.

        Returns:
            A new LCA object

        """
        if isinstance(demand, (str, tuple, list)):
            raise ValueError("Demand must be a dictionary")
        for key in demand:
            if not key:
                raise ValueError("Invalid demand dictionary")

        clean_databases()
        self._fixed = False

        self.demand = demand
        self.method = method
        self.normalization = normalization
        self.weighting = weighting
        self.database_filepath = database_filepath

        self.database_filepath, \
            self.method_filepath, \
            self.weighting_filepath, \
            self.normalization_filepath = \
            self.get_array_filepaths()

    def get_array_filepaths(self):
        """Use utility functions to get all array filepaths"""
        return (
            get_filepaths(self.demand, "demand"),
            get_filepaths(self.method, "method"),
            get_filepaths(self.weighting, "weighting"),
            get_filepaths(self.normalization, "normalization"),
        )

    def build_demand_array(self, demand=None):
        """Turn the demand dictionary into a *NumPy* array of correct size.

        Args:
            * *demand* (dict, optional): Demand dictionary. Optional, defaults to ``self.demand``.

        Returns:
            A 1-dimensional NumPy array

        """
        demand = demand or self.demand
        self.demand_array = np.zeros(len(self.product_dict))
        for key in demand:
            try:
                self.demand_array[self.product_dict[key]] = demand[key]
            except KeyError:
                if key in self.activity_dict:
                    raise ValueError((u"LCA can only be performed on products,"
                        u" not activities ({} is the wrong dimension)"
                        ).format(key)
                    )
                else:
                    raise OutsideTechnosphere("Can't find key {} in product dictionary".format(key))

    #########################
    ### Data manipulation ###
    #########################

    def fix_dictionaries(self):
        """
Fix technosphere and biosphere dictionaries from this:

.. code-block:: python

    {mapping integer id: matrix row/column index}

To this:

.. code-block:: python

    {(database, key): matrix row/column index}

This isn't needed for the LCA calculation itself, but is helpful when interpreting results.

Doesn't require any arguments or return anything, but changes ``self.activity_dict``, ``self.product_dict`` and ``self.biosphere_dict``.

        """
        if self._fixed:
            # Already fixed - should be idempotent
            return False
        elif not mapping:
            # Don't have access to mapping
            return False
        rev_mapping = {v: k for k, v in mapping.items()}
        self._activity_dict = copy.deepcopy(self.activity_dict)
        self.activity_dict = {
            rev_mapping[k]: v for k, v in self.activity_dict.items()}
        self._product_dict = self.product_dict
        self.product_dict = {
            rev_mapping[k]: v for k, v in self.product_dict.items()}
        self._biosphere_dict = self.biosphere_dict
        self.biosphere_dict = {
            rev_mapping[k]: v for k, v in self.biosphere_dict.items()}
        self._fixed = True
        return True

    def reverse_dict(self):
        """Construct reverse dicts from technosphere and biosphere row and col indices to activity_dict/product_dict/biosphere_dict keys.

        Returns:
            (reversed ``self.activity_dict``, ``self.product_dict`` and ``self.biosphere_dict``)
        """
        rev_activity = {v: k for k, v in self.activity_dict.items()}
        rev_product = {v: k for k, v in self.product_dict.items()}
        rev_bio = {v: k for k, v in self.biosphere_dict.items()}
        return rev_activity, rev_product, rev_bio

    ######################
    ### Data retrieval ###
    ######################

    def load_lci_data(self, builder=TBMBuilder):
        """Load data and create technosphere and biosphere matrices."""
        self._fixed = False
        self.bio_params, self.tech_params, \
            self.biosphere_dict, self.activity_dict, \
            self.product_dict, self.biosphere_matrix, \
            self.technosphere_matrix = \
            builder.build(self.database_filepath)
        if len(self.activity_dict) != len(self.product_dict):
            raise NonsquareTechnosphere((
                "Technosphere matrix is not square: {} activities (columns) and {} products (rows). "
                "Use LeastSquaresLCA to solve this system, or fix the input "
                "data").format(len(self.activity_dict), len(self.product_dict))
            )
        self.fix_dictionaries()

    def load_lcia_data(self, builder=MatrixBuilder):
        """Load data and create characterization matrix.

        This method will filter out regionalized characterization factors. This filtering needs access to ``bw2data`` - therefore, regionalized methods will cause incorrect results if ``bw2data`` is not importable.

        """
        self.cf_params, _, _, self.characterization_matrix = builder.build(
                self.method_filepath,
                "amount",
                "flow",
                "row",
                row_dict=self._biosphere_dict,
                one_d=True,
            )
        if global_index is not None:
            mask = self.cf_params['geo'] == global_index
            self.cf_params = self.cf_params[mask]
            self.characterization_matrix = builder.build_diagonal_matrix(self.cf_params, self._biosphere_dict, "row", "amount")

    def load_normalization_data(self, builder=MatrixBuilder):
        """Load normalization data."""
        self.normalization_params, _, _, self.normalization_matrix = \
            builder.build(
                self.normalization_filepath,
                "amount",
                "flow",
                "index",
                row_dict=self._biosphere_dict,
                one_d=True
            )

    def load_weighting_data(self):
        """Load weighting data, a 1-element array."""
        self.weighting_params = load_arrays(
            self.weighting_filepath
        )
        self.weighting_value = self.weighting_params['amount']

    ####################
    ### Calculations ###
    ####################

    def decompose_technosphere(self):
        """
Factorize the technosphere matrix into lower and upper triangular matrices, :math:`A=LU`. Does not solve the linear system :math:`Ax=B`.

Doesn't return anything, but creates ``self.solver``.

.. warning:: Incorrect results could occur if a technosphere matrix was factorized, and then a new technosphere matrix was constructed, as ``self.solver`` would still be the factorized older technosphere matrix. You are responsible for deleting ``self.solver`` when doing these types of advanced calculations.

        """
        self.solver = factorized(self.technosphere_matrix.tocsc())

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
            return spsolve(
                self.technosphere_matrix,
                self.demand_array)

    def lci(self, factorize=False,
            builder=TBMBuilder):
        """
Calculate a life cycle inventory.

#. Load LCI data, and construct the technosphere and biosphere matrices.
#. Build the demand array
#. Solve the linear system to get the supply array and life cycle inventory.

Args:
    * *factorize* (bool, optional): Factorize the technosphere matrix. Makes additional calculations with the same technosphere matrix much faster. Default is ``False``; not useful is only doing one LCI calculation.
    * *builder* (``MatrixBuilder`` object, optional): Default is ``bw2calc.matrices.TechnosphereBiosphereMatrixBuilder``, which is fine for most cases. Custom matrix builders can be used to manipulate data in creative ways before building the matrices.

.. warning:: Custom matrix builders should inherit from ``TechnosphereBiosphereMatrixBuilder``, because technosphere inputs need to have their signs flipped to be negative, as we do :math:`A^{-1}f` directly instead of :math:`(I - A^{-1})f`.

Doesn't return anything, but creates ``self.supply_array`` and ``self.inventory``.

        """
        self.load_lci_data(builder)
        self.build_demand_array()
        if factorize:
            self.decompose_technosphere()
        self.lci_calculation()

    def lci_calculation(self):
        """The actual LCI calculation.

        Separated from ``lci`` to be reusable in cases where the matrices are already built, e.g. ``redo_lci`` and Monte Carlo classes.

        """
        self.supply_array = self.solve_linear_system()
        # Turn 1-d array into diagonal matrix
        count = len(self.activity_dict)
        self.inventory = self.biosphere_matrix * \
            sparse.spdiags([self.supply_array], [0], count, count)

    def lcia(self, builder=MatrixBuilder):
        """
Calculate the life cycle impact assessment.

#. Load and construct the characterization matrix
#. Multiply the characterization matrix by the life cycle inventory

Args:
    * *builder* (``MatrixBuilder`` object, optional): Default is ``bw2calc.matrices.MatrixBuilder``, which is fine for most cases. Custom matrix builders can be used to manipulate data in creative ways before building the characterization matrix.

Doesn't return anything, but creates ``self.characterized_inventory``.

        """
        assert hasattr(self, "inventory"), "Must do lci first"
        assert self.method, "Must specify a method to perform LCIA"
        self.load_lcia_data(builder)
        self.lcia_calculation()

    def lcia_calculation(self):
        """The actual LCIA calculation.

        Separated from ``lcia`` to be reusable in cases where the matrices are already built, e.g. ``redo_lcia`` and Monte Carlo classes.

        """
        self.characterized_inventory = \
            self.characterization_matrix * self.inventory

    def normalize(self):
        """Multiply characterized inventory by flow-specific normalization factors."""
        assert hasattr(self, "characterized_inventory"), "Must do lcia first"
        if not hasattr(self, "normalization_matrix"):
            self.load_normalization_data()
        self.normalization_calculation()

    def normalization_calculation(self):
        """The actual normalization calculation.

        Creates ``self.normalized_inventory``."""
        self.normalized_inventory = \
            self.normalization_matrix * self.characterized_inventory

    def weight(self):
        """Multiply characterized inventory by weighting value.

        Can be done with or without normalization."""
        assert hasattr(self, "characterized_inventory"), "Must do lcia first"
        if not hasattr(self, "weighting_value"):
            self.load_weighting_data()

    def weighting_calculation(self):
        """The actual weighting calculation.

        Multiples weighting value by normalized inventory, if available, otherwise by characterized inventory.

        Creates ``self.weighted_inventory``."""
        if hasattr(self, "normalized_inventory"):
            obj = self.normalized_inventory
        else:
            obj = self.characterized_inventory
        self.weighted_inventory = self.weighting_value[0] * obj

    @property
    def score(self):
        """
The LCIA score as a ``float``.

Note that this is a `property <http://docs.python.org/2/library/functions.html#property>`_, so it is ``foo.lca``, not ``foo.score()``
        """
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        if self.weighting:
            assert hasattr(self, "weighted_inventory"), "Must do weighting first"
            return float(self.weighted_inventory.sum())
        return float(self.characterized_inventory.sum())

    #########################
    ### Redo calculations ###
    #########################

    def rebuild_technosphere_matrix(self, vector):
        """Build a new technosphere matrix using the same row and column indices, but different values. Useful for Monte Carlo iteration or sensitivity analysis.

        Args:
            * *vector* (array): 1-dimensional NumPy array with length (# of technosphere parameters), in same order as ``self.tech_params``.

        Doesn't return anything, but overwrites ``self.technosphere_matrix``.

        """
        self.technosphere_matrix = MatrixBuilder.build_matrix(
            self.tech_params, self._activity_dict, self._product_dict,
            "row", "col",
            new_data=TBMBuilder.fix_supply_use(self.tech_params, vector.copy())
        )

    def rebuild_biosphere_matrix(self, vector):
        """Build a new biosphere matrix using the same row and column indices, but different values. Useful for Monte Carlo iteration or sensitivity analysis.

        Args:
            * *vector* (array): 1-dimensional NumPy array with length (# of biosphere parameters), in same order as ``self.bio_params``.

        Doesn't return anything, but overwrites ``self.biosphere_matrix``.

        """
        self.biosphere_matrix = MatrixBuilder.build_matrix(
            self.bio_params, self._biosphere_dict, self._activity_dict,
            "row", "col", new_data=vector)

    def rebuild_characterization_matrix(self, vector):
        """Build a new characterization matrix using the same row and column indices, but different values. Useful for Monte Carlo iteration or sensitivity analysis.

        Args:
            * *vector* (array): 1-dimensional NumPy array with length (# of characterization parameters), in same order as ``self.cf_params``.

        Doesn't return anything, but overwrites ``self.characterization_matrix``.

        """
        self.characterization_matrix = MatrixBuilder.build_diagonal_matrix(
            self.cf_params, self._biosphere_dict,
            "row", "row", new_data=vector)

    def switch_method(self, method):
        """Switch to LCIA method `method`"""
        self.method = method
        _, self.method_filepath, _, _ = self.get_array_filepaths()
        self.load_lcia_data()

    def switch_normalization(self, normalization):
        """Switch to LCIA normalization `normalization`"""
        self.normalization = normalization
        _, _, _, self.normalization_filepath = self.get_array_filepaths()
        self.load_normalization_data()

    def switch_weighting(self, weighting):
        """Switch to LCIA weighting `weighting`"""
        self.weighting = weighting
        _, _, self.weighting_filepath, _ = self.get_array_filepaths()
        self.load_weighting_data()

    def redo_lci(self, demand):
        """Redo LCI with same databases but different demand.

        Args:
            * *demand* (dict): A demand dictionary.

        Doesn't return anything, but overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        .. warning:: If you want to redo the LCIA as well, use ``redo_lcia(demand)`` directly.

        """
        assert hasattr(self, "inventory"), "Must do lci first"
        self.build_demand_array(demand)
        self.lci_calculation()

    def redo_lcia(self, demand=None):
        """Redo LCIA, optionally with new demand.

        Args:
            * *demand* (dict, optional): New demand dictionary. Optional, defaults to ``self.demand``.

        Doesn't return anything, but overwrites ``self.characterized_inventory``. If ``demand`` is given, also overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        """
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        if demand:
            self.redo_lci(demand)
        self.lcia_calculation()

    def to_dataframe(self, cutoff=200):
        """Return all nonzero elements of characterized inventory as Pandas dataframe"""
        assert mapping, "This method doesn't work with independent LCAs"
        assert pandas, "This method requires the `pandas` (http://pandas.pydata.org/) library"
        assert hasattr(self, "characterized_inventory"), "Must do LCIA calculation first"

        from bw2data import get_activity

        coo = self.characterized_inventory.tocoo()
        stacked = np.vstack([np.abs(coo.data), coo.row, coo.col, coo.data])
        stacked.sort()
        rev_activity, _, rev_bio = self.reverse_dict()
        length = stacked.shape[1]

        data = []
        for x in range(min(cutoff, length)):
            if stacked[3, length - x - 1] == 0.:
                continue
            activity = get_activity(rev_activity[stacked[2, length - x - 1]])
            flow = get_activity(rev_bio[stacked[1, length - x - 1]])
            data.append((
                activity['name'],
                flow['name'],
                activity.get('location'),
                stacked[3, length - x - 1]
            ))
        return pandas.DataFrame(
            data,
            columns=['Activity', 'Flow', 'Region', 'Amount']
        )

    ####################
    ### Contribution ###
    ####################

    def top_emissions(self, **kwargs):
        """Call ``bw2analyzer.ContributionAnalyses.annotated_top_emissions``"""
        try:
            from bw2analyzer import ContributionAnalysis
        except ImportError:
            raise ImportError("`bw2analyzer` is not installed")
        return ContributionAnalysis().annotated_top_emissions(self, **kwargs)

    def top_activities(self, **kwargs):
        """Call ``bw2analyzer.ContributionAnalyses.annotated_top_processes``"""
        try:
            from bw2analyzer import ContributionAnalysis
        except ImportError:
            raise ImportError("`bw2analyzer` is not installed")
        return ContributionAnalysis().annotated_top_processes(self, **kwargs)
