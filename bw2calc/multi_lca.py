import datetime
import logging
import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import bw_processing as bwp
import matrix_utils as mu
import numpy as np
from fs.base import FS
from pydantic import BaseModel
from scipy import sparse

from . import PYPARDISO, __version__, spsolve
from .dictionary_manager import DictionaryManager
from .errors import OutsideTechnosphere
from .lca import LCABase
from .method_config import MethodConfig

# from .single_value_diagonal_matrix import SingleValueDiagonalMatrix
from .utils import consistent_global_index, get_datapackage

logger = logging.getLogger("bw2calc")


class DemandsValidator(BaseModel):
    demands: dict[str, dict[int, float]]


class MultiLCA(LCABase):
    """
    Perform LCA on multiple demands, impact categories, and normalization and weighting sets.

    Builds only *one* technosphere and biosphere matrix which can cover all demands.

    Main differences from the base `LCA` class:

    * Many attributes are plural, such as `supply_arrays`, `inventories`, characterization_matrices`
    * `demands` must be a dictionary with `str` keys, e.g. `{'my truck': {12345: 1}}`
    * `demands` must have integer IDs; you can't pass `('database', 'code')` or `Activity` objects.
    * Calculation results are a dictionary with keys of functional units and impact categories

    The calculation procedure is the same as for singular LCA: `lci()`, `lcia()`, and `next()`. See
    the LCA documentation on this behaviour and input arguments.

    Parameters
    ----------
    demands : dict[str, dict[int, float]]
        The demands for which the LCA will be calculated. The keys identify functional unit sets.
    method_config : dict
        Dictionary satisfying the `MethodConfig` specification.
    data_objs : list[bw_processing.Datapackage]
        List of `bw_processing.Datapackage` objects. Should include data for all needed matrices.
    remapping_dicts : dict[str, dict]
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
    selective_use : dict[str, dict]
        Dictionary that gives more control on whether `use_arrays` or `use_distributions` should be
        used. Has the form `{matrix_label: {"use_arrays"|"use_distributions": bool}`. Standard
        matrix labels are `technosphere_matrix`, `biosphere_matrix`, and `characterization_matrix`.
    """

    matrix_labels = [
        "technosphere_mm",
        "biosphere_mm",
    ]
    matrix_list_labels = [
        "characterization_mm_list",
        "normalization_mm_list",
        "weighting_mm_list",
    ]

    def __init__(
        self,
        demands: dict[str, dict[int, float]],
        method_config: dict,
        data_objs: Iterable[Union[Path, FS, bwp.DatapackageBase]],
        remapping_dicts: Optional[Iterable[dict]] = None,
        log_config: Optional[dict] = None,
        seed_override: Optional[int] = None,
        use_arrays: Optional[bool] = False,
        use_distributions: Optional[bool] = False,
        selective_use: Optional[dict] = None,
    ):
        # Validation checks
        # DemandsValidator(demands=demands)
        MethodConfig(**method_config)

        self.demands = demands
        self.config = method_config
        self.packages = [get_datapackage(obj) for obj in data_objs]
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
                "demands": self.demands,
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

    # Don't allow new demand
    def redo_lci(self) -> None:
        return super().redo_lci()

    def lci(self) -> None:
        return super().lci()

    def redo_lcia(self) -> None:
        return super().redo_lcia()

    def lcia(self) -> None:
        return super().lcia()

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

    def build_demand_array(self, demands: Optional[dict] = None) -> None:
        """Turn the demand dictionary into a *NumPy* array of correct size.

        Args:
            * *demand* (dict, optional): Demand dictionary. Optional, defaults to ``self.demand``.

        Returns:
            A 1-dimensional NumPy array

        """
        demands = self.demands if demands is None else demands
        self.demand_arrays = {}

        for key, value in demands.items():
            array = np.zeros(len(self.dicts.product))

            for process_id, process_amount in value.items():
                try:
                    array[self.dicts.product[process_id]] = process_amount
                except KeyError:
                    if process_id in self.dicts.activity:
                        raise ValueError(
                            f"LCA can only be performed on products, not activities ({process_id} "
                            + "is the wrong dimension)"
                        )
                    else:
                        raise OutsideTechnosphere(
                            f"Can't find key {process_id} in product dictionary"
                        )

            self.demand_arrays[key] = array

    ##################
    # Data retrieval #
    ##################

    def filter_package_identifier(
        self, data_objs: Iterable[bwp.DatapackageBase], identifier: list[str]
    ) -> list[bwp.DatapackageBase]:
        """Filter the datapackage resources in `data_objs` whose "identifier" attribute equals
        `identifier`.

        Used in splitting up impact categories, normalization, and weighting matrices."""
        return [dp.filter_by_attribute("identifier", identifier) for dp in data_objs]

    def load_lcia_data(self, data_objs: Optional[Iterable[bwp.DatapackageBase]] = None) -> None:
        """Load data and create characterization matrix.

        This method will filter out regionalized characterization factors.

        """
        global_index = consistent_global_index(data_objs or self.packages)
        fltr = (lambda x: x["col"] == global_index) if global_index is not None else None

        use_arrays, use_distributions = self.check_selective_use("characterization_matrix")

        self.characterization_mm_list = {}
        self.characterization_matrices = {}

        for ic in self.config["impact_categories"]:
            try:
                mm = mu.MappedMatrix(
                    packages=self.filter_package_identifier(
                        data_objs=data_objs or self.packages, identifier=list(ic)
                    ),
                    matrix="characterization_matrix",
                    use_arrays=use_arrays,
                    use_distributions=use_distributions,
                    seed_override=self.seed_override,
                    row_mapper=self.biosphere_mm.row_mapper,
                    diagonal=True,
                    custom_filter=fltr,
                )
                self.characterization_mm_list[ic] = mm
                self.characterization_matrices[ic] = mm.matrix
                if len(mm.matrix.data) == 0:
                    warnings.warn(f"All values in characterization matrix for {ic} are zero")
            except mu.errors.AllArraysEmpty:
                raise ValueError(
                    f"Given `method` or `data_objs` for impact category {ic} have no "
                    + "characterization data"
                )

    # def load_normalization_data(
    #     self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]] = None
    # ) -> None:
    #     """Load normalization data."""
    #     use_arrays, use_distributions = self.check_selective_use("normalization_matrix")

    #     self.normalization_mm = mu.MappedMatrix(
    #         packages=data_objs or self.packages,
    #         matrix="normalization_matrix",
    #         use_arrays=use_arrays,
    #         use_distributions=use_distributions,
    #         seed_override=self.seed_override,
    #         row_mapper=self.biosphere_mm.row_mapper,
    #         diagonal=True,
    #     )
    #     self.normalization_matrix = self.normalization_mm.matrix

    # def load_weighting_data(
    #     self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]] = None
    # ) -> None:
    #     """Load normalization data."""
    #     use_arrays, use_distributions = self.check_selective_use("weighting_matrix")

    #     self.weighting_mm = SingleValueDiagonalMatrix(
    #         packages=data_objs or self.packages,
    #         matrix="weighting_matrix",
    #         dimension=len(self.biosphere_mm.row_mapper),
    #         use_arrays=use_arrays,
    #         use_distributions=use_distributions,
    #         seed_override=self.seed_override,
    #     )
    #     self.weighting_matrix = self.weighting_mm.matrix

    ################
    # Calculations #
    ################

    def decompose_technosphere(self) -> None:
        raise NotImplementedError

    def lci_calculation(self) -> None:
        """The actual LCI calculation.

        Separated from ``lci`` to be reusable in cases where the matrices are already built, e.g.
        ``redo_lci`` and Monte Carlo classes.

        """
        count = len(self.dicts.activity)
        solutions = spsolve(
            self.technosphere_matrix, np.vstack([arr for arr in self.demand_arrays.values()]).T
        )
        self.supply_arrays = {name: arr for name, arr in zip(self.demands, solutions.T)}
        # Turn 1-d array into diagonal matrix
        self.inventories = {
            name: self.biosphere_matrix @ sparse.spdiags([arr], [0], count, count)
            for name, arr in self.supply_arrays.items()
        }

    def lcia_calculation(self) -> None:
        """The actual LCIA calculation.

        Separated from ``lcia`` to be reusable in cases where the matrices are already built, e.g.
        ``redo_lcia`` and Monte Carlo classes.

        """
        self.characterized_inventories = {}

        for demand_name, mat_inv in self.inventories.items():
            for ic_name, mat_ia in self.characterization_matrices.items():
                self.characterized_inventories[(demand_name, ic_name)] = mat_ia @ mat_inv

    # def normalization_calculation(self) -> None:
    #     """The actual normalization calculation.

    #     Creates ``self.normalized_inventory``."""
    #     self.normalized_inventory = self.normalization_matrix * self.characterized_inventory

    # def weighting_calculation(self) -> None:
    #     """The actual weighting calculation.

    #     Multiples weighting value by normalized inventory, if available, otherwise by
    #   characterized inventory.

    #     Creates ``self.weighted_inventory``."""
    #     if hasattr(self, "normalized_inventory"):
    #         obj = self.normalized_inventory
    #     else:
    #         obj = self.characterized_inventory
    #     self.weighted_inventory = self.weighting_matrix * obj

    @property
    def scores(self) -> dict:
        """
        The LCIA score as a ``float``.

        Note that this is a `property <http://docs.python.org/2/library/functions.html#property>`_,
        so it is ``foo.lca``, not ``foo.score()``
        """
        assert hasattr(self, "characterized_inventories"), "Must do LCIA first"
        return {key: arr.sum() for key, arr in self.characterized_inventories.items()}
        # if hasattr(self, "weighted_inventory"):
        #     return float(self.weighted_inventory.sum())
        # elif hasattr(self, "normalized_inventory"):
        #     return float(self.normalized_inventory.sum())
        # else:
        #     return float(self.characterized_inventory.sum())


class MultiLCAResult:
    def __init__(self):
        """Container for storing and interpreting `MultiLCA` results."""
        pass
