# -*- coding: utf-8 -*
from __future__ import division
from brightway2 import config as base_config
from brightway2 import databases, mapping, \
    Method, Weighting, Normalization
from scipy.sparse.linalg import factorized, spsolve
from scipy import sparse
import numpy as np
from .errors import OutsideTechnosphere
from .matrices import MatrixBuilder
from .matrices import TechnosphereBiosphereMatrixBuilder as TBMBuilder
from .utils import load_arrays


class LCA(object):
    """A static LCI or LCIA calculation.

    Following the general philosophy of Brightway2, and good software practices, there is a clear separation of concerns between retrieving and formatting data and doing an LCA. Building the necessary matrices is done with MatrixBuilder objects (:ref:`matrixbuilders`). The LCA class only does the LCA calculations themselves.

    """

    #############
    ### Setup ###
    #############

    def __init__(self, demand, method=None, weighting=None,
            normalization=None, config=None):
        """Create a new LCA calculation.

        Args:
            * *demand* (dict): The demand or functional unit. Needs to be a dictionary to indicate amounts, e.g. ``{("my database", "my process"): 2.5}``. Dictionary keys are the same as `Database <http://brightway2.readthedocs.org/en/latest/key-concepts.html#databases>`_ keys.
            * *method* (tuple, optional): `LCIA Method <http://brightway2.readthedocs.org/en/latest/key-concepts.html#impact-assessment-methods>`_ tuple, e.g. ``("My", "great", "LCIA", "method")``. Can be omitted if only interested in calculating the life cycle inventory.
            * *config* (``bw2data.config`` object, optional): Specify a different configuration directory to load data. Optional, and not useful 99 percent of the time.

        Returns:
            A new LCA object

        """
        self._mapped_dict = True
        self.dirpath = (config or base_config).dir
        if isinstance(demand, (basestring, tuple, list)):
            raise ValueError("Demand must be a dictionary")
        self.demand = demand
        self.method = method
        self.weighting = weighting
        self.normalization = normalization
        self.databases = self.get_databases(demand)

    def get_databases(self, demand):
        """Get list of databases needed for demand/functional unit.

        Args:
            * *demand* (dict): Demand dictionary.

        Returns:
            A set of database names

        """
        def extend(seeds):
            return set.union(seeds,
                             set.union(*[set(databases[obj]['depends'])
                                         for obj in seeds]))

        seed = {key[0] for key in demand}
        extended = extend(seed)
        # depends can have loops, so no simple recursive search; need to check
        # membership
        while extended != seed:
            seed = extended
            extended = extend(seed)
        return extended

    def build_demand_array(self, demand=None):
        """Turn the demand dictionary into a *NumPy* array of correct size.

        Args:
            * *demand* (dict, optional): Demand dictionary. Optional, defaults to ``self.demand``.

        Returns:
            A 1-dimensional NumPy array

        """
        demand = demand or self.demand
        self.demand_array = np.zeros(len(self.technosphere_dict))
        for key in demand:
            if self._mapped_dict:
                self.demand_array[self.technosphere_dict[mapping[key]]] = \
                demand[key]
            else:
                self.demand_array[self.technosphere_dict[key]] = demand[key]


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

Doesn't require any arguments or return anything, but changes ``self.technosphere_dict`` and ``self.biosphere_dict``.

        """
        if not self._mapped_dict:
            # Already reversed - should be idempotent
            return False
        rev_mapping = {v: k for k, v in mapping.iteritems()}
        self.technosphere_dict = {
            rev_mapping[k]: v for k, v in self.technosphere_dict.iteritems()}
        self.biosphere_dict = {
            rev_mapping[k]: v for k, v in self.biosphere_dict.iteritems()}
        self._mapped_dict = False
        return True

    def reverse_dict(self):
        """Construct reverse dicts from technosphere and biosphere row and col indices to technosphere_dict/biosphere_dict keys.

        Returns:
            (reversed ``self.technosphere_dict``, reversed ``self.biosphere_dict``)
        """
        rev_tech = {v: k for k, v in self.technosphere_dict.iteritems()}
        rev_bio = {v: k for k, v in self.biosphere_dict.iteritems()}
        return rev_tech, rev_bio

    ######################
    ### Data retrieval ###
    ######################

    def load_lci_data(self, builder=TBMBuilder):
        """Load data and create technosphere and biosphere matrices."""
        self.bio_params, self.tech_params, \
            self.biosphere_dict, self.technosphere_dict, \
            self.biosphere_matrix, self.technosphere_matrix = \
            builder.build(self.dirpath, self.databases)

    def load_lcia_data(self, builder=MatrixBuilder):
        """Load data and create characterization matrix."""
        self.cf_params, _, _, self.characterization_matrix = builder.build(
                self.dirpath,
                [Method(self.method).filename],
                "amount",
                "flow",
                "row",
                row_dict=self.biosphere_dict,
                one_d=True
            )

    def load_normalization_data(self, builder=MatrixBuilder):
        """Load normalization data."""
        self.normalization_params, _, _, self.normalization_matrix = \
            builder.build(
                self.dirpath,
                [Normalization(self.normalization).filename],
                "amount",
                "flow",
                "index",
                row_dict=self.biosphere_dict,
                one_d=True
            )

    def load_weighting_data(self):
        """Load weighting data, a 1-element array."""
        self.weighting_params = load_arrays(
            self.dirpath,
            [Weighting(self.weighting).filename]
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
        count = len(self.technosphere_dict)
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
            self.weighted_inventory = \
                self.weighting_value[0] * self.normalized_inventory
        else:
            self.weighted_inventory = \
                self.weighting_value[0] * self.characterized_inventory

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
            self.tech_params, self.technosphere_dict, self.technosphere_dict,
            "row", "col",
            new_data=TBMBuilder.fix_supply_use(self.tech_params, vector)
        )

    def rebuild_biosphere_matrix(self, vector):
        """Build a new biosphere matrix using the same row and column indices, but different values. Useful for Monte Carlo iteration or sensitivity analysis.

        Args:
            * *vector* (array): 1-dimensional NumPy array with length (# of biosphere parameters), in same order as ``self.bio_params``.

        Doesn't return anything, but overwrites ``self.biosphere_matrix``.

        """
        self.biosphere_matrix = MatrixBuilder.build_matrix(
            self.bio_params, self.biosphere_dict, self.technosphere_dict,
            "row", "col", new_data=vector)

    def rebuild_characterization_matrix(self, vector):
        """Build a new characterization matrix using the same row and column indices, but different values. Useful for Monte Carlo iteration or sensitivity analysis.

        Args:
            * *vector* (array): 1-dimensional NumPy array with length (# of characterization parameters), in same order as ``self.cf_params``.

        Doesn't return anything, but overwrites ``self.characterization_matrix``.

        """
        self.characterization_matrix = MatrixBuilder.build_diagonal_matrix(
            self.cf_params, self.biosphere_dict,
            "row", "row", new_data=vector)

    def redo_lci(self, demand):
        """Redo LCI with same databases but different demand.

        Args:
            * *demand* (dict): A demand dictionary.

        Doesn't return anything, but overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        .. warning:: If you want to redo the LCIA as well, use ``redo_lcia(demand)`` directly.

        """
        assert hasattr(self, "inventory"), "Must do lci first"
        try:
            self.build_demand_array(demand)
        except KeyError:
            raise OutsideTechnosphere
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
