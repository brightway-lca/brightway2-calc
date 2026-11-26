import logging
import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import bw_processing as bwp
import matrix_utils as mu
import numpy as np
from fsspec import AbstractFileSystem
from pydantic import BaseModel
from scipy import sparse

from bw2calc import PYPARDISO, __version__
from bw2calc.calculation_config import CalculationConfig, resolve_calculation_config
from bw2calc.dictionary_manager import DictionaryManager
from bw2calc.errors import OutsideTechnosphere
from bw2calc.lca_base import LCABase
from bw2calc.method_config import MethodConfig
from bw2calc.restricted_sparse_matrix_dict import RestrictedSparseMatrixDict
from bw2calc.single_value_diagonal_matrix import SingleValueDiagonalMatrix
from bw2calc.utils import GlobalIndexFilter, convert_tuple_to_list, get_datapackage, utc_now

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
    the LCA documentation for these methods and their input arguments.

    Parameters
    ----------
    demands : dict[str, dict[int, float]]
        The demands for which the LCA will be calculated. The keys identify functional unit sets.
    method_config : dict | MethodConfig
        Dictionary satisfying the `MethodConfig` specification or `MethodConfig` instance.
    data_objs : list[bw_processing.Datapackage]
        List of `bw_processing.Datapackage` objects. Should include data for all needed matrices.
    remapping_dicts : dict[str, dict]
        Dict of remapping dictionaries that link Brightway `Node` ids to `(database, code)` tuples.
        `remapping_dicts` can provide such remapping for any of `activity`, `product`, `biosphere`.
    log_config : dict
        Optional arguments to pass to logging. Not yet implemented.
    calculation_config : CalculationConfig | dict
        How matrices are built and sampled: `use_arrays`, `use_distributions`, the RNG `seed`,
        `derive_group_seeds`, and per-matrix overrides of these. See `CalculationConfig`.
        Keyword-only.
    seed_override, use_arrays, use_distributions, selective_use
        Deprecated; use `calculation_config` instead. `selective_use` corresponds to
        `CalculationConfig.matrices`, and `seed_override` to `CalculationConfig.seed`.
        Passing any of these together with `calculation_config` raises an error.
    """

    matrix_labels = [
        "technosphere_mm",
        "biosphere_mm",
    ]
    matrix_list_labels = [
        "characterization_mm_dict",
        "normalization_mm_dict",
        "weighting_mm_dict",
    ]

    def __init__(
        self,
        demands: dict[str, dict[int, float]],
        method_config: Union[dict, MethodConfig],
        data_objs: Iterable[Union[Path, AbstractFileSystem, bwp.DatapackageBase]],
        remapping_dicts: Optional[Iterable[dict]] = None,
        log_config: Optional[dict] = None,
        seed_override: Optional[int] = None,
        use_arrays: Optional[bool] = None,
        use_distributions: Optional[bool] = None,
        selective_use: Optional[dict] = None,
        *,
        calculation_config: Optional[Union[CalculationConfig, dict]] = None,
    ):
        # Validation checks
        DemandsValidator(demands=demands)
        self.calculation_config = resolve_calculation_config(
            calculation_config,
            seed_override=seed_override,
            use_arrays=use_arrays,
            use_distributions=use_distributions,
            selective_use=selective_use,
        )
        if isinstance(method_config, MethodConfig):
            method_config = {
                key: value for key, value in method_config.model_dump().items() if value is not None
            }
        MethodConfig(**method_config)

        self.demands = demands
        self.config = method_config
        self.packages = [get_datapackage(obj) for obj in data_objs]
        self.dicts = DictionaryManager()
        self.remapping_dicts = remapping_dicts or {}

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
                "utc": utc_now(),
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
        skip_first_iteration = getattr(self, "keep_first_iteration_flag", False)

        if not skip_first_iteration:
            self._delete_solver_state()

            for matrix in self.matrix_labels:
                if hasattr(self, matrix):
                    obj = getattr(self, matrix)
                    next(obj)
                    message = """Iterating matrix {matrix}. Indexers: {indexer_state}""".format(
                        matrix=matrix,
                        indexer_state=[(str(p), p.indexer.index) for p in obj.packages],
                    )
                    logger.debug(
                        message,
                        extra={
                            "matrix": matrix,
                            "indexers": [(str(p), p.indexer.index) for p in obj.packages],
                            "matrix_sum": obj.matrix.sum(),
                            "utc": utc_now(),
                        },
                    )

            for matrix_dict in self.matrix_list_labels:
                if hasattr(self, matrix_dict):
                    obj = getattr(self, matrix_dict)
                    next(obj)
                    message = """Iterating matrix dict {matrix}. Indexer: {indexer_state}""".format(
                        matrix=matrix, indexer_state=obj.global_indexer.index
                    )
                    logger.debug(
                        message,
                        extra={
                            "matrix_dict": matrix_dict,
                            "indexer": obj.global_indexer.index,
                            "matrix_sums": [mm.matrix.sum() for mm in obj.values()],
                            "utc": utc_now(),
                        },
                    )

            if hasattr(self, "after_matrix_iteration"):
                self.after_matrix_iteration()

        # Avoid this conversion each time we do a calculation in the future
        # See https://github.com/haasad/PyPardiso/issues/75#issuecomment-2186825609
        # Nothing to convert if `next()` is called before the matrices are loaded
        if PYPARDISO and hasattr(self, "technosphere_matrix"):
            self.technosphere_matrix = self.technosphere_matrix.tocsr()

        if skip_first_iteration:
            delattr(self, "keep_first_iteration_flag")

        self._calculation()

    def _calculation(self) -> None:
        """Factored out to allow subclasses to easily modify behaviour."""
        if hasattr(self, "inventories"):
            self.lci_calculation()
        if hasattr(self, "characterized_inventories"):
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
                array[self.product_index(process_id)] = process_amount

            self.demand_arrays[key] = array

    def product_index(self, product_id: int) -> int:
        """Matrix row index of the product ``product_id``."""
        try:
            return self.dicts.product[product_id]
        except KeyError as exc:
            if product_id in self.dicts.activity:
                raise ValueError(
                    f"LCA can only be performed on products, not processes ({product_id} "
                    + "is a process id)"
                ) from exc
            else:
                raise OutsideTechnosphere(
                    f"Can't find key {product_id} in product dictionary"
                ) from exc

    ##################
    # Data retrieval #
    ##################

    def filter_package_by_identifier(
        self,
        data_objs: Iterable[bwp.DatapackageBase],
        identifier: Union[list[str], tuple[str, ...], str],
    ) -> list[bwp.DatapackageBase]:
        """Filter the datapackage resources in `data_objs` whose "identifier" attribute equals
        input argument `identifier`.

        Used in splitting up impact categories, normalization, and weighting matrices."""
        identifier = convert_tuple_to_list(identifier)
        return [dp.filter_by_attribute("identifier", identifier) for dp in data_objs]

    def load_lcia_data(self, data_objs: Optional[Iterable[bwp.DatapackageBase]] = None) -> None:
        """Load data and create characterization matrices.

        This method will filter out regionalized characterization factors.

        """
        # Iterated once per impact category
        data_objs = list(data_objs or self.packages)
        # Each resource group is filtered with its own global index. Only the packages
        # filtered to an impact category are registered, so only their indices are loaded
        fltr = GlobalIndexFilter()

        self.characterization_mm_dict = mu.MappedMatrixDict(
            packages={
                ic: list(
                    fltr.through(
                        self.filter_package_by_identifier(
                            data_objs=data_objs, identifier=convert_tuple_to_list(ic)
                        )
                    )
                )
                for ic in self.config["impact_categories"]
            },
            matrix="characterization_matrix",
            **self.matrix_kwargs("characterization_matrix"),
            row_mapper=self.biosphere_mm.row_mapper,
            col_mapper=None,
            diagonal=True,
            custom_filter=fltr,
        )
        for key, value in self.characterization_mm_dict.items():
            if len(value.matrix.data) == 0:
                warnings.warn(f"All values in characterization matrix for {key} are zero")

        self.characterization_matrices = mu.SparseMatrixDict(
            [(key, value.matrix) for key, value in self.characterization_mm_dict.items()]
        )

    def load_normalization_data(
        self, data_objs: Optional[Iterable[bwp.DatapackageBase]] = None
    ) -> None:
        """Load normalization data."""
        self.normalization_mm_dict = mu.MappedMatrixDict(
            packages={
                nrml: self.filter_package_by_identifier(
                    data_objs=data_objs or self.packages, identifier=convert_tuple_to_list(nrml)
                )
                for nrml in self.config["normalizations"]
            },
            matrix="normalization_matrix",
            **self.matrix_kwargs("normalization_matrix"),
            row_mapper=self.biosphere_mm.row_mapper,
            diagonal=True,
        )
        for key, value in self.normalization_mm_dict.items():
            if len(value.matrix.data) == 0:
                warnings.warn(f"All values in normalization matrix for {key} are zero")

        self.normalization_matrices = RestrictedSparseMatrixDict(
            self.config["normalizations"],
            [(key, value.matrix) for key, value in self.normalization_mm_dict.items()],
        )

    def load_weighting_data(
        self, data_objs: Optional[Iterable[bwp.DatapackageBase]] = None
    ) -> None:
        """Load weighting data."""
        self.weighting_mm_dict = mu.MappedMatrixDict(
            packages={
                wng: self.filter_package_by_identifier(
                    data_objs=data_objs or self.packages, identifier=convert_tuple_to_list(wng)
                )
                for wng in self.config["weightings"]
            },
            matrix="weighting_matrix",
            row_mapper=None,
            dimension=len(self.biosphere_mm.row_mapper),
            **self.matrix_kwargs("weighting_matrix"),
            matrix_class=SingleValueDiagonalMatrix,
            diagonal=True,
        )
        for key, value in self.weighting_mm_dict.items():
            if len(value.matrix.data) == 0:
                warnings.warn(f"All values in weighting matrix for {key} are zero")

        self.weighting_matrices = RestrictedSparseMatrixDict(
            self.config["weightings"],
            [(key, value.matrix) for key, value in self.weighting_mm_dict.items()],
        )

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
        demand_matrix = np.vstack([arr for arr in self.demand_arrays.values()]).T
        self.supply_arrays = {
            name: arr
            for name, arr in zip(
                self.demands,
                self._spsolve(self.technosphere_matrix, demand_matrix).reshape(count, -1).T,
            )
        }
        # Turn 1-d array into diagonal matrix
        self.inventories = mu.SparseMatrixDict(
            [
                (name, self.biosphere_matrix @ sparse.spdiags([arr], [0], count, count))
                for name, arr in self.supply_arrays.items()
            ]
        )

    def lcia_calculation(self) -> None:
        """The actual LCIA calculation.

        Separated from ``lcia`` to be reusable in cases where the matrices are already built, e.g.
        ``redo_lcia`` and Monte Carlo classes.

        """
        self.characterized_inventories = self.characterization_matrices @ self.inventories
        if hasattr(self, "normalization_matrices"):
            self.normalization_calculation()
        if hasattr(self, "weighting_matrices"):
            self.weighting_calculation()

    def normalization_calculation(self) -> None:
        """The actual normalization calculation.

        Creates ``self.normalized_inventories``."""
        self.normalized_inventories = self.normalization_matrices @ self.characterized_inventories

    def weighting_calculation(self) -> None:
        """The actual weighting calculation.

          Multiplies weighting value by normalized inventories, if available, otherwise by
        characterized inventories.

          Creates ``self.weighted_inventories``."""
        if hasattr(self, "normalized_inventories"):
            self.weighted_inventories = self.weighting_matrices @ self.normalized_inventories
        else:
            self.weighted_inventories = self.weighting_matrices @ self.characterized_inventories

    @property
    def scores(self) -> dict:
        """
        The LCIA score as a ``float``.

        Note that this is a `property <http://docs.python.org/2/library/functions.html#property>`_,
        so it is ``foo.lca``, not ``foo.score()``
        """
        if not hasattr(self, "characterized_inventories"):
            raise ValueError("Must do LCIA first")

        if hasattr(self, "weighted_inventories"):
            return {key: arr.sum() for key, arr in self.weighted_inventories.items()}
        elif hasattr(self, "normalized_inventories"):
            return {key: arr.sum() for key, arr in self.normalized_inventories.items()}
        else:
            return {key: arr.sum() for key, arr in self.characterized_inventories.items()}
