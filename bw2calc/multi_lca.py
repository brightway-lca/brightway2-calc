import datetime
import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Optional, Union, Sequence

import bw_processing as bwp
import matrix_utils as mu
import numpy as np
from fs.base import FS
from pydantic import BaseModel
from scipy import sparse

from . import PYPARDISO, __version__
from .dictionary_manager import DictionaryManager
from .errors import InconsistentLCIADatapackages, OutsideTechnosphere
from .lca import LCABase
from .method_config import MethodConfig
from .single_value_diagonal_matrix import SingleValueDiagonalMatrix
from .utils import consistent_global_index, get_datapackage, wrap_functional_unit

logger = logging.getLogger("bw2calc")


class DemandsValidator(BaseModel):
    demands: dict[str, dict[int, float]]


class MultiLCA(LCABase):
    matrix_labels = [
        "technosphere_mm",
        "biosphere_mm",
    ]
    matrix_list_labels = [
        "characterization_mm_list",
        "normalization_mm_list",
        "weighting_mm_list",
    ]

    """
    Perform LCA on multiple demands and multiple impact categories.

    Builds only *one* technosphere and biosphere matrix which can cover all demands.

    Differs from the base `LCA` class in that:

    * `supply_arrays` in place of `supply array`; Numpy array with dimensions (biosphere flows,
        demand dictionaries)
    * `inventories` in place of `inventory`; List of matrices with length number of demand
        dictionaries
    * `characterized_inventories` in place of `characterized_inventory`; like `inventories`
    * `scores` instead of `score`; Numpy array with dimensions (demand dictionaries, impact
        categories)

    The input arguments are also different, and are mostly plural.

    Instantiation requires separate `data_objs` for the method, normalization, and weighting, as
    these are all impact category-specific. Use a `bw2data` compatibility function to get data
    prepared in the correct way.

    This class supports both stochastic and static LCA, and can use a variety of ways to describe
    uncertainty. The input flags `use_arrays` and `use_distributions` control some of this
    stochastic behaviour. See the
    [documentation for `matrix_utils`](https://github.com/brightway-lca/matrix_utils) for more
    information on the technical implementation.

    Parameters
    ----------
    demands : Iterable[dict[int: float]]
        The demands for which the LCA will be calculated. The keys **must be** integers ids.
    inventory_data_objs : list[bw_processing.Datapackage]
        List of `bw_processing.Datapackage` objects **for the inventory**. Can be loaded via
        `bw2data.prepare_lca_inputs` or constructed manually. Should include data for all needed
        matrices.
    methods : List[bw_processing.Datapackage]
        Tuple defining the LCIA method, such as `('foo', 'bar')`. Only needed if not passing
        `data_objs`.
    weighting : List[bw_processing.Datapackage]
        Tuple defining the LCIA weighting, such as `('foo', 'bar')`. Only needed if not passing
        `data_objs`.
    weighting : List[bw_processing.Datapackage]
        String defining the LCIA normalization, such as `'foo'`. Only needed if not passing
        `data_objs`.
    remapping_dicts : dict[str : dict]
        Dict of remapping dictionaries that link Brightway `Node` ids to `(database, code)` tuples.
        `remapping_dicts` can provide such remapping for any of `activity`, `product`, `biosphere`.
    log_config : dict
        Optional arguments to pass to logging. Not yet implemented.
    seed_override : int
        RNG seed to use in place of `Datapackage` seed, if any.
    use_arrays : bool
        Use arrays instead of vectors from the given `data_objs`
    use_distributions : bool
        Use probability distributions from the given `data_objs`
    selective_use : dict[str : dict]
        Dictionary that gives more control on whether `use_arrays` or `use_distributions` should be
        used. Has the form `{matrix_label: {"use_arrays"|"use_distributions": bool}`. Standard
        matrix labels are `technosphere_matrix`, `biosphere_matrix`, and `characterization_matrix`.
    """

    def __init__(
        self,
        demands: Sequence[Mapping],
        method_config: dict,
        inventory_data_objs: Iterable[Union[Path, FS, bwp.DatapackageBase]],
        method_data_objs: Optional[Iterable[Union[Path, FS, bwp.DatapackageBase]]] = None,
        normalization_data_objs: Optional[Iterable[Union[Path, FS, bwp.DatapackageBase]]] = None,
        weighting_data_objs: Optional[Iterable[Union[Path, FS, bwp.DatapackageBase]]] = None,
        remapping_dicts: Optional[Iterable[dict]] = None,
        log_config: Optional[dict] = None,
        seed_override: Optional[int] = None,
        use_arrays: Optional[bool] = False,
        use_distributions: Optional[bool] = False,
        selective_use: Optional[dict] = None,
    ):
        # Resolve potential iterator
        self.demands = list(demands)

        # Validation checks
        DemandsValidator(demands)
        MethodConfig(method_config)

        self.packages = [get_datapackage(obj) for obj in inventory_data_objs]

        if method_data_objs is not None:
            self.method_packages = [get_datapackage(obj) for obj in method_data_objs]
        else:
            self.method_packages = []
        if weighting_data_objs is not None:
            self.weighting_packages = [get_datapackage(obj) for obj in weighting_data_objs]
        else:
            self.method_packages = []
        if normalization_data_objs is not None:
            self.normalization_packages = [get_datapackage(obj) for obj in normalization_data_objs]
        else:
            self.normalization_packages = []

        if (
            self.method_packages
            and self.weighting_packages
            and len(self.method_packages) != len(self.weighting_packages)
        ):
            raise InconsistentLCIADatapackages(
                "Found {} methods and {} weightings (must be the same)".format(
                    len(self.method_packages), len(self.weighting_packages)
                )
            )
        elif (
            self.method_packages
            and self.normalization_packages
            and len(self.method_packages) != len(self.normalization_packages)
        ):
            raise InconsistentLCIADatapackages(
                "Found {} methods and {} normalizations (must be the same)".format(
                    len(self.method_packages), len(self.normalization_packages)
                )
            )

        self.dicts = DictionaryManager()
        self.use_arrays = use_arrays
        self.use_distributions = use_distributions
        self.selective_use = selective_use or {}
        self.remapping_dicts = remapping_dicts or {}
        self.seed_override = seed_override

        message = (
            """Initialized MultiLCA object. Demands: {demands}, data_objs: {data_objs}""".format(
                demands=self.demands, data_objs=self.packages
            )
        )
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

    ####################
    # Modified methods #
    ####################

    def redo_lci(self) -> None:
        # Don't allow new demand
        return super().redo_lci()

    def redo_lcia(self) -> None:
        # Don't allow new demand
        return super().redo_lcia()

    ####################
    # LCA Calculations #
    ####################

    def __next__(self) -> None:
        skip_first_iteration = (
            hasattr(self, "keep_first_iteration_flag") and self.keep_first_iteration_flag
        )

        for matrix in self.matrix_labels:
            if not skip_first_iteration and hasattr(self, matrix):
                obj = getattr(self, matrix)
                next(obj)
                message = """Iterating {matrix}. Indexers: {indexer_state}""".format(
                    matrix=matrix,
                    indexer_state=[(str(p), p.indexer.index) for p in obj.packages],
                )
                logger.debug(
                    message,
                    extra={
                        "matrix": matrix,
                        "indexers": [(str(p), p.indexer.index) for p in obj.packages],
                        "matrix_sum": obj.matrix.sum(),
                        "utc": datetime.datetime.utcnow(),
                    },
                )

        if not skip_first_iteration and hasattr(self, "after_matrix_iteration"):
            self.after_matrix_iteration()

        if skip_first_iteration:
            delattr(self, "keep_first_iteration_flag")

        if hasattr(self, "inventory"):
            self.lci_calculation()
        if hasattr(self, "characterized_inventory"):
            self.lcia_calculation()

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
                        f"LCA can only be performed on products, not activities ({key} is the"
                        + " wrong dimension)"
                    )
                else:
                    raise OutsideTechnosphere(f"Can't find key {key} in product dictionary")

    ##################
    # Data retrieval #
    ##################

    def load_lcia_data(
        self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]] = None
    ) -> None:
        """Load data and create characterization matrix.

        This method will filter out regionalized characterization factors.

        """
        global_index = consistent_global_index(data_objs or self.packages)
        fltr = (lambda x: x["col"] == global_index) if global_index is not None else None

        use_arrays, use_distributions = self.check_selective_use("characterization_matrix")

        try:
            self.characterization_mm = mu.MappedMatrix(
                packages=data_objs or self.packages,
                matrix="characterization_matrix",
                use_arrays=use_arrays,
                use_distributions=use_distributions,
                seed_override=self.seed_override,
                row_mapper=self.biosphere_mm.row_mapper,
                diagonal=True,
                custom_filter=fltr,
            )
        except mu.errors.AllArraysEmpty:
            raise ValueError("Given `method` or `data_objs` have no characterization data")
        self.characterization_matrix = self.characterization_mm.matrix
        if len(self.characterization_matrix.data) == 0:
            warnings.warn("All values in characterization matrix are zero")

    def load_normalization_data(
        self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]] = None
    ) -> None:
        """Load normalization data."""
        use_arrays, use_distributions = self.check_selective_use("normalization_matrix")

        self.normalization_mm = mu.MappedMatrix(
            packages=data_objs or self.packages,
            matrix="normalization_matrix",
            use_arrays=use_arrays,
            use_distributions=use_distributions,
            seed_override=self.seed_override,
            row_mapper=self.biosphere_mm.row_mapper,
            diagonal=True,
        )
        self.normalization_matrix = self.normalization_mm.matrix

    def load_weighting_data(
        self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]] = None
    ) -> None:
        """Load normalization data."""
        use_arrays, use_distributions = self.check_selective_use("weighting_matrix")

        self.weighting_mm = SingleValueDiagonalMatrix(
            packages=data_objs or self.packages,
            matrix="weighting_matrix",
            dimension=len(self.biosphere_mm.row_mapper),
            use_arrays=use_arrays,
            use_distributions=use_distributions,
            seed_override=self.seed_override,
        )
        self.weighting_matrix = self.weighting_mm.matrix

    ################
    # Calculations #
    ################

    def lci_calculation(self) -> None:
        """The actual LCI calculation.

        Separated from ``lci`` to be reusable in cases where the matrices are already built, e.g.
        ``redo_lci`` and Monte Carlo classes.

        """
        self.supply_array = self.solve_linear_system()
        # Turn 1-d array into diagonal matrix
        count = len(self.dicts.activity)
        self.inventory = self.biosphere_matrix * sparse.spdiags(
            [self.supply_array], [0], count, count
        )

    def lcia_calculation(self) -> None:
        """The actual LCIA calculation.

        Separated from ``lcia`` to be reusable in cases where the matrices are already built, e.g.
        ``redo_lcia`` and Monte Carlo classes.

        """
        self.characterized_inventory = self.characterization_matrix * self.inventory

    def normalization_calculation(self) -> None:
        """The actual normalization calculation.

        Creates ``self.normalized_inventory``."""
        self.normalized_inventory = self.normalization_matrix * self.characterized_inventory

    def weighting_calculation(self) -> None:
        """The actual weighting calculation.

        Multiples weighting value by normalized inventory, if available, otherwise by characterized
        inventory.

        Creates ``self.weighted_inventory``."""
        if hasattr(self, "normalized_inventory"):
            obj = self.normalized_inventory
        else:
            obj = self.characterized_inventory
        self.weighted_inventory = self.weighting_matrix * obj

    @property
    def score(self) -> float:
        """
        The LCIA score as a ``float``.

        Note that this is a `property <http://docs.python.org/2/library/functions.html#property>`_,
        so it is ``foo.lca``, not ``foo.score()``
        """
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        if hasattr(self, "weighted_inventory"):
            return float(self.weighted_inventory.sum())
        elif hasattr(self, "normalized_inventory"):
            return float(self.normalized_inventory.sum())
        else:
            return float(self.characterized_inventory.sum())

    def check_demand(self, demand: Optional[dict] = None):
        if demand is None:
            return
        else:
            for key in demand:
                if key not in self.dicts.product and not isinstance(key, int):
                    raise KeyError(
                        f"Key '{key}' not in product dictionary; make sure to pass the integer id"
                        + ", not a key like `('foo', 'bar')` or an `Actiivity` or `Node` object."
                    )
