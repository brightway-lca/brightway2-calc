from . import prepare_lca_inputs, factorized, spsolve, __version__, PYPARDISO
import bw_processing as bwp
import matrix_utils as mu
from .errors import (
    EmptyBiosphere,
    NonsquareTechnosphere,
    OutsideTechnosphere,
)

from .dictionary_manager import DictionaryManager
from .utils import consistent_global_index, wrap_functional_unit, get_datapackage
from .single_value_diagonal_matrix import SingleValueDiagonalMatrix
from collections.abc import Mapping, Iterator
from scipy import sparse
from functools import partial
from fs.base import FS
from typing import Iterable, Union, Optional
from pathlib import Path

import datetime
import logging
import numpy as np
import pandas
import warnings

logger = logging.getLogger("bw2calc")


class LCA(Iterator):
    """An LCI or LCIA calculation.

    Compatible with Brightway2 and 2.5 semantics. Can be static, stochastic, or iterative (scenario-based), depending on the ``data_objs`` input data..

    """

    #############
    ### Setup ###
    #############

    def __init__(
        self,
        demand: dict,
        # Brightway 2 calling convention
        method: Optional[tuple] = None,
        weighting: Optional[str] = None,
        normalization: Optional[str] = None,
        # Brightway 2.5 calling convention
        data_objs: Optional[Iterable[Union[Path, FS, bwp.DatapackageBase]]] = None,
        remapping_dicts: Optional[Iterable[dict]] = None,
        log_config: Optional[dict] = None,
        seed_override: Optional[int] = None,
        use_arrays: bool = False,
        use_distributions: bool = False,
    ):
        """Create a new LCA calculation.

        Args:
            * *demand* (dict): The demand or functional unit. Needs to be a dictionary to indicate amounts, e.g. ``{7: 2.5}``.
            * *method* (tuple, optional): LCIA Method tuple, e.g. ``("My", "great", "LCIA", "method")``. Can be omitted if only interested in calculating the life cycle inventory.

        Returns:
            A new LCA object

        """
        if not isinstance(demand, Mapping):
            raise ValueError("Demand must be a dictionary")
        # TODO: check after matrix building
        # for key in demand:
        #     if not key:
        #         raise ValueError("Invalid demand dictionary")

        if data_objs is None:
            self.ensure_bw2data_available()
            demand, self.packages, remapping_dicts = prepare_lca_inputs(
                demand=demand,
                method=method,
                weighting=weighting,
                normalization=normalization,
            )
            self.method = method
            self.weighting = weighting
            self.normalization = normalization
        else:
            self.packages = [get_datapackage(obj) for obj in data_objs]

        self.dicts = DictionaryManager()
        self.demand = demand
        self.use_arrays = use_arrays
        self.use_distributions = use_distributions
        self.remapping_dicts = remapping_dicts or {}
        self.seed_override = seed_override

        message = """Initialized LCA object. Demand: {demand}, data_objs: {data_objs}""".format(demand=self.demand, data_objs=self.packages)
        logger.info(
            message,
            extra={
                "demand": wrap_functional_unit(self.demand),
                "data_objs": str(self.packages),
                "bw2calc": __version__,
                "pypardiso": PYPARDISO,
                "numpy": np.__version__,
                "matrix_utils": mu.__version__,
                "bw_processing": bwp.__version__,
                "utc": datetime.datetime.utcnow(),
            },
        )

    def __next__(self) -> None:
        matrices = ["technosphere_mm", "biosphere_mm", "characterization_mm", "normalization_mm", "weighting_mm"]
        for matrix in matrices:
            if hasattr(self, matrix):
                obj = getattr(self, matrix)
                next(obj)
                message = """Iterating {matrix}. Indexers: {indexer_state}""".format(matrix=matrix, indexer_state=[(str(p), p.indexer.index) for p in obj.packages])
                logger.debug(
                    message,
                    extra={
                        "matrix": matrix,
                        "indexers": [(str(p), p.indexer.index) for p in obj.packages],
                        "matrix_sum": obj.matrix.sum(),
                        "utc": datetime.datetime.utcnow(),
                    },
                )
        if hasattr(self, "inventory"):
            self.lci_calculation()
        if hasattr(self, "characterized_inventory"):
            self.lcia_calculation()

    def ensure_bw2data_available(self):
        """Raises ``ImportError`` is bw2data not available or version < 4."""
        if prepare_lca_inputs is None:
            raise ImportError("bw2data version >= 4 not found")

    def build_demand_array(self, demand: Optional[dict] = None) -> None:
        """Turn the demand dictionary into a *NumPy* array of correct size.

        Args:
            * *demand* (dict, optional): Demand dictionary. Optional, defaults to ``self.demand``.

        Returns:
            A 1-dimensional NumPy array

        """
        demand = demand or self.demand
        self.demand_array = np.zeros(len(self.dicts.product))
        for key in demand:
            try:
                self.demand_array[self.dicts.product[key]] = demand[key]
            except KeyError:
                if key in self.dicts.activity:
                    raise ValueError(
                        f"LCA can only be performed on products, not activities ({key} is the wrong dimension)"
                    )
                else:
                    raise OutsideTechnosphere(
                        f"Can't find key {key} in product dictionary"
                    )

    ######################
    ### Data retrieval ###
    ######################

    def load_lci_data(self) -> None:
        """Load inventory data and create technosphere and biosphere matrices."""
        self.technosphere_mm = mu.MappedMatrix(
            packages=self.packages,
            matrix="technosphere_matrix",
            use_arrays=self.use_arrays,
            use_distributions=self.use_distributions,
            seed_override=self.seed_override,
        )
        self.technosphere_matrix = self.technosphere_mm.matrix
        self.dicts.product = partial(self.technosphere_mm.row_mapper.to_dict)
        self.dicts.activity = partial(self.technosphere_mm.col_mapper.to_dict)

        if len(self.technosphere_mm.row_mapper) != len(self.technosphere_mm.col_mapper):
            raise NonsquareTechnosphere(
                (
                    "Technosphere matrix is not square: {} activities (columns) and {} products (rows). "
                    "Use LeastSquaresLCA to solve this system, or fix the input "
                    "data"
                ).format(
                    len(self.technosphere_mm.col_mapper),
                    len(self.technosphere_mm.row_mapper),
                )
            )

        self.biosphere_mm = mu.MappedMatrix(
                packages=self.packages,
                matrix="biosphere_matrix",
                use_arrays=self.use_arrays,
                use_distributions=self.use_distributions,
                seed_override=self.seed_override,
                col_mapper=self.technosphere_mm.col_mapper,
                empty_ok=True
            )
        self.biosphere_matrix = self.biosphere_mm.matrix
        self.dicts.biosphere = partial(self.biosphere_mm.row_mapper.to_dict)

        if self.biosphere_mm.matrix.shape[0] == 0:
            warnings.warn(
                "No valid biosphere flows found. No inventory results can "
                "be calculated, `lcia` will raise an error"
            )

    def remap_inventory_dicts(self) -> None:
        """Remap ``self.dicts.activity|product|biosphere`` and ``self.demand`` from database integer IDs to keys (``(database name, code)``).

        Uses remapping dictionaries in ``self.remapping_dicts``."""
        if "product" in self.remapping_dicts:
            self.demand = {
                self.remapping_dicts["product"][k]: v for k, v in self.demand.items()
            }

        for label in ("activity", "product", "biosphere"):
            if label in self.remapping_dicts:
                getattr(self.dicts, label).remap(self.remapping_dicts[label])

    def load_lcia_data(
        self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]] = None
    ) -> None:
        """Load data and create characterization matrix.

        This method will filter out regionalized characterization factors.

        """
        global_index, kwargs = consistent_global_index(data_objs or self.packages), {}
        if global_index is not None:
            kwargs["custom_filter"] = lambda x: x["col"] == global_index

        self.characterization_mm = mu.MappedMatrix(
            packages=data_objs or self.packages,
            matrix="characterization_matrix",
            use_arrays=self.use_arrays,
            use_distributions=self.use_distributions,
            seed_override=self.seed_override,
            row_mapper=self.biosphere_mm.row_mapper,
            diagonal=True,
            custom_filter=kwargs.get("custom_filter")
        )
        self.characterization_matrix = self.characterization_mm.matrix

    def load_normalization_data(self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]]) -> None:
        """Load normalization data."""
        self.normalization_mm = mu.MappedMatrix(
            packages=data_objs or self.packages,
            matrix="normalization_matrix",
            use_arrays=self.use_arrays,
            use_distributions=self.use_distributions,
            seed_override=self.seed_override,
            row_mapper=self.biosphere_mm.row_mapper,
            diagonal=True,
        )
        self.normalization_matrix = self.normalization_mm.matrix

    def load_weighting_data(self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]]) -> None:
        """Load normalization data."""
        self.weighting_mm = SingleValueDiagonalMatrix(
            packages=data_objs or self.packages,
            matrix="weighting_matrix",
            dimension=len(self.biosphere_mm.row_mapper),
            use_arrays=self.use_arrays,
            use_distributions=self.use_distributions,
            seed_override=self.seed_override,
        )
        self.weighting_matrix = self.weighting_mm.matrix


    ####################
    ### Calculations ###
    ####################

    def decompose_technosphere(self) -> None:
        """
        Factorize the technosphere matrix into lower and upper triangular matrices, :math:`A=LU`. Does not solve the linear system :math:`Ax=B`.

        Doesn't return anything, but creates ``self.solver``.

        .. warning:: Incorrect results could occur if a technosphere matrix was factorized, and then a new technosphere matrix was constructed, as ``self.solver`` would still be the factorized older technosphere matrix. You are responsible for deleting ``self.solver`` when doing these types of advanced calculations.

        """
        self.solver = factorized(self.technosphere_matrix.tocsc())

    def solve_linear_system(self) -> None:
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
            return spsolve(self.technosphere_matrix, self.demand_array)

    def lci(self, factorize: bool = False) -> None:
        """
        Calculate a life cycle inventory.

        #. Load LCI data, and construct the technosphere and biosphere matrices.
        #. Build the demand array
        #. Solve the linear system to get the supply array and life cycle inventory.

        Args:
            * *factorize* (bool, optional): Factorize the technosphere matrix. Makes additional calculations with the same technosphere matrix much faster. Default is ``False``; not useful is only doing one LCI calculation.
            * *builder* (``MatrixBuilder`` object, optional): Default is ``bw2calc.matrices.MatrixBuilder``, which is fine for most cases. Custom matrix builders can be used to manipulate data in creative ways before building the matrices.

        Doesn't return anything, but creates ``self.supply_array`` and ``self.inventory``.

        """
        self.load_lci_data()
        self.build_demand_array()
        if factorize:
            self.decompose_technosphere()
        self.lci_calculation()

    def lci_calculation(self) -> None:
        """The actual LCI calculation.

        Separated from ``lci`` to be reusable in cases where the matrices are already built, e.g. ``redo_lci`` and Monte Carlo classes.

        """
        self.supply_array = self.solve_linear_system()
        # Turn 1-d array into diagonal matrix
        count = len(self.dicts.activity)
        self.inventory = self.biosphere_matrix * sparse.spdiags(
            [self.supply_array], [0], count, count
        )

    def lcia(self) -> None:
        """
        Calculate the life cycle impact assessment.

        #. Load and construct the characterization matrix
        #. Multiply the characterization matrix by the life cycle inventory

        Args:
            * *builder* (``MatrixBuilder`` object, optional): Default is ``bw2calc.matrices.MatrixBuilder``, which is fine for most cases. Custom matrix builders can be used to manipulate data in creative ways before building the characterization matrix.

        Doesn't return anything, but creates ``self.characterized_inventory``.

        """
        assert hasattr(self, "inventory"), "Must do lci first"
        if not self.dicts.biosphere:
            raise EmptyBiosphere

        self.load_lcia_data()
        self.lcia_calculation()

    def lcia_calculation(self) -> None:
        """The actual LCIA calculation.

        Separated from ``lcia`` to be reusable in cases where the matrices are already built, e.g. ``redo_lcia`` and Monte Carlo classes.

        """
        self.characterized_inventory = self.characterization_matrix * self.inventory

    def normalize(self) -> None:
        """Multiply characterized inventory by flow-specific normalization factors."""
        assert hasattr(self, "characterized_inventory"), "Must do lcia first"
        if not hasattr(self, "normalization_matrix"):
            self.load_normalization_data()
        self.normalization_calculation()

    def normalization_calculation(self) -> None:
        """The actual normalization calculation.

        Creates ``self.normalized_inventory``."""
        self.normalized_inventory = (
            self.normalization_matrix * self.characterized_inventory
        )

    def weighting(self) -> None:
        """Multiply characterized inventory by weighting value.

        Can be done with or without normalization."""
        assert hasattr(self, "characterized_inventory"), "Must do lcia first"
        if not hasattr(self, "weighting_value"):
            self.load_weighting_data()

    def weighting_calculation(self) -> None:
        """The actual weighting calculation.

        Multiples weighting value by normalized inventory, if available, otherwise by characterized inventory.

        Creates ``self.weighted_inventory``."""
        if hasattr(self, "normalized_inventory"):
            obj = self.normalized_inventory
        else:
            obj = self.characterized_inventory
        self.weighted_inventory = self.weighting_value[0] * obj

    @property
    def score(self) -> float:
        """
        The LCIA score as a ``float``.

        Note that this is a `property <http://docs.python.org/2/library/functions.html#property>`_, so it is ``foo.lca``, not ``foo.score()``
        """
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        # if hasattr(self, "weighting"):
        #     assert hasattr(self, "weighted_inventory"), "Must do weighting first"
        #     return float(self.weighted_inventory.sum())
        return float(self.characterized_inventory.sum())

    #########################
    ### Redo calculations ###
    #########################

    def switch_method(
        self, method=Union[tuple, Iterable[Union[FS, bwp.DatapackageBase]]]
    ) -> None:
        """Switch to LCIA method `method`"""
        if isinstance(method, tuple):
            self.ensure_bw2data_available()
            _, data_objs, _ = prepare_lca_inputs(method=method)
            self.method = method
        else:
            data_objs = method
        self.load_lcia_data(data_objs=data_objs)

        message = """Switched LCIA method. data_objs: {data_objs}""".format(data_objs=data_objs)
        logger.info(
            message,
            extra={
                "data_objs": str(data_objs),
                "utc": datetime.datetime.utcnow(),
            },
        )

    def switch_normalization(self, normalization):
        """Switch to LCIA normalization `normalization`"""
        self.normalization = normalization
        _, _, _, self.normalization_filepath = self.get_array_filepaths()
        self.load_normalization_data()

        message = """Switched LCIA method. data_objs: {data_objs}""".format(data_objs=data_objs)
        logger.info(
            message,
            extra={
                "data_objs": str(data_objs),
                "utc": datetime.datetime.utcnow(),
            },
        )

    def switch_weighting(self, weighting):
        """Switch to LCIA weighting `weighting`"""
        self.weighting = weighting
        _, _, self.weighting_filepath, _ = self.get_array_filepaths()
        self.load_weighting_data()
        self.logger.info("Switching LCIA weighting", extra={"weighting": weighting})

    def redo_lci(self, demand: Optional[dict] = None) -> None:
        """Redo LCI with same databases but different demand.

        Args:
            * *demand* (dict): A demand dictionary.

        Doesn't return anything, but overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        .. warning:: If you want to redo the LCIA as well, use ``redo_lcia(demand)`` directly.

        """
        assert hasattr(self, "inventory"), "Must do lci first"
        if demand is not None:
            self.build_demand_array(demand)
            self.demand = demand
        self.lci_calculation()
        # self.logger.info(
        #     "Redoing LCI", extra={"demand": wrap_functional_unit(demand or self.demand)}
        # )

    def redo_lcia(self, demand: Optional[dict] = None) -> None:
        """Redo LCIA, optionally with new demand.

        Args:
            * *demand* (dict, optional): New demand dictionary. Optional, defaults to ``self.demand``.

        Doesn't return anything, but overwrites ``self.characterized_inventory``. If ``demand`` is given, also overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        """
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        if demand is not None:
            self.redo_lci(demand)
            self.demand = demand
        self.lcia_calculation()
        # self.logger.info(
        #     "Redoing LCIA",
        #     extra={"demand": wrap_functional_unit(demand or self.demand)},
        # )

    # def to_dataframe(self, cutoff=200):
    #     """Return all nonzero elements of characterized inventory as Pandas dataframe"""
    #     assert mapping, "This method doesn't work with independent LCAs"
    #     assert (
    #         pandas
    #     ), "This method requires the `pandas` (http://pandas.pydata.org/) library"
    #     assert hasattr(
    #         self, "characterized_inventory"
    #     ), "Must do LCIA calculation first"

    #     from bw2data import get_activity

    #     coo = self.characterized_inventory.tocoo()
    #     stacked = np.vstack([np.abs(coo.data), coo.row, coo.col, coo.data])
    #     stacked.sort()
    #     rev_activity, _, rev_bio = self.reverse_dict()
    #     length = stacked.shape[1]

    #     data = []
    #     for x in range(min(cutoff, length)):
    #         if stacked[3, length - x - 1] == 0.0:
    #             continue
    #         activity = get_activity(rev_activity[stacked[2, length - x - 1]])
    #         flow = get_activity(rev_bio[stacked[1, length - x - 1]])
    #         data.append(
    #             (
    #                 activity["name"],
    #                 flow["name"],
    #                 activity.get("location"),
    #                 stacked[3, length - x - 1],
    #             )
    #         )
    #     return pandas.DataFrame(data, columns=["Activity", "Flow", "Region", "Amount"])

    ####################
    ### Contribution ###
    ####################

    # def top_emissions(self, **kwargs):
    #     """Call ``bw2analyzer.ContributionAnalyses.annotated_top_emissions``"""
    #     try:
    #         from bw2analyzer import ContributionAnalysis
    #     except ImportError:
    #         raise ImportError("`bw2analyzer` is not installed")
    #     return ContributionAnalysis().annotated_top_emissions(self, **kwargs)

    # def top_activities(self, **kwargs):
    #     """Call ``bw2analyzer.ContributionAnalyses.annotated_top_processes``"""
    #     try:
    #         from bw2analyzer import ContributionAnalysis
    #     except ImportError:
    #         raise ImportError("`bw2analyzer` is not installed")
    #     return ContributionAnalysis().annotated_top_processes(self, **kwargs)

    def has(self, label: str) -> bool:
        """Shortcut to find out if matrix data for type ``{label}_matrix`` is present in the given data objects.

        Returns a boolean. Will return ``True`` even if data for a zero-dimensional matrix is given."""
        return any(
            True
            for package in self.packages
            for resource in package.resources
            if resource["matrix"] == f"{label}_matrix"
        )
