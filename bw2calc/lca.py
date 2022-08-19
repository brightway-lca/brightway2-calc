import datetime
import logging
import warnings
from collections.abc import Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Iterable, Optional, Union, Callable
from numbers import Number

import bw_processing as bwp
import matrix_utils as mu
import numpy as np
import pandas as pd
from fs.base import FS
from scipy import sparse

from . import PYPARDISO, __version__, factorized, prepare_lca_inputs, spsolve
from .dictionary_manager import DictionaryManager
from .errors import EmptyBiosphere, NonsquareTechnosphere, OutsideTechnosphere
from .single_value_diagonal_matrix import SingleValueDiagonalMatrix
from .utils import consistent_global_index, get_datapackage, wrap_functional_unit

try:
    from bw2data import get_node
except ImportError:
    get_node = None

logger = logging.getLogger("bw2calc")


class LCA(Iterator):
    """An LCI or LCIA calculation.

    Compatible with Brightway2 and 2.5 semantics. Can be static, stochastic, or iterative (scenario-based), depending on the ``data_objs`` input data..

    """
    matrix_labels = [
        "technosphere_mm",
        "biosphere_mm",
        "characterization_mm",
        "normalization_mm",
        "weighting_mm",
    ]

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

        message = """Initialized LCA object. Demand: {demand}, data_objs: {data_objs}""".format(
            demand=self.demand, data_objs=self.packages
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

    def keep_first_iteration(self):
        """Set a flag to use the current values as first element when iterating.

        When creating the class instance, we already use the first index. This method allows us to use the values for the first index.

        Note that the methods ``.lci_calculation()`` and ``.lcia_calculation()`` will be called on the current values, even if these calculations have already been done."""
        self.keep_first_iteration_flag = True

    def __next__(self) -> None:
        skip_first_iteration = hasattr(self, "keep_first_iteration_flag") and self.keep_first_iteration_flag

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

    def load_lci_data(self, nonsquare_ok=False) -> None:
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

        if (
            len(self.technosphere_mm.row_mapper) != len(self.technosphere_mm.col_mapper)
            and not nonsquare_ok
        ):
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
            empty_ok=True,
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
        global_index = consistent_global_index(data_objs or self.packages)
        fltr = (
            (lambda x: x["col"] == global_index) if global_index is not None else None
        )

        self.characterization_mm = mu.MappedMatrix(
            packages=data_objs or self.packages,
            matrix="characterization_matrix",
            use_arrays=self.use_arrays,
            use_distributions=self.use_distributions,
            seed_override=self.seed_override,
            row_mapper=self.biosphere_mm.row_mapper,
            diagonal=True,
            custom_filter=fltr,
        )
        self.characterization_matrix = self.characterization_mm.matrix

    def load_normalization_data(
        self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]] = None
    ) -> None:
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

    def load_weighting_data(
        self, data_objs: Optional[Iterable[Union[FS, bwp.DatapackageBase]]] = None
    ) -> None:
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
        if PYPARDISO:
            warnings.warn("PARDISO installed; this is a no-op")
        else:
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

    def lci(self, demand: Optional[dict] = None, factorize: bool = False) -> None:
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
        if not hasattr(self, "technosphere_matrix"):
            self.load_lci_data()
        if demand is not None:
            self.__check_demand(demand)
            self.build_demand_array(demand)
            self.demand = demand
        else:
            self.build_demand_array()
        if factorize and not PYPARDISO:
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

    def lcia(self, demand: Optional[dict] = None) -> None:
        """
        Calculate the life cycle impact assessment.

        #. Load and construct the characterization matrix
        #. Multiply the characterization matrix by the life cycle inventory

        Doesn't return anything, but creates ``self.characterized_inventory``.

        """
        assert hasattr(self, "inventory"), "Must do lci first"
        if not self.dicts.biosphere:
            raise EmptyBiosphere

        if not hasattr(self, "characterization_matrix"):
            self.load_lcia_data()
        if demand is not None:
            self.__check_demand(demand)
            self.lci(demand=demand)
            self.demand = demand
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
        """Backwards compatibility. Switching to verb form consistent with ``.normalize``."""
        warnings.warn('Please switch to `.weight`', DeprecationWarning)
        return self.weight()

    def weight(self) -> None:
        """Multiply characterized inventory by weighting value.

        Can be done with or without normalization."""
        assert hasattr(self, "characterized_inventory"), "Must do lcia first"
        if not hasattr(self, "weighting_value"):
            self.load_weighting_data()
        self.weighting_calculation()

    def weighting_calculation(self) -> None:
        """The actual weighting calculation.

        Multiples weighting value by normalized inventory, if available, otherwise by characterized inventory.

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

        Note that this is a `property <http://docs.python.org/2/library/functions.html#property>`_, so it is ``foo.lca``, not ``foo.score()``
        """
        assert hasattr(self, "characterized_inventory"), "Must do LCIA first"
        if hasattr(self, "weighted_inventory"):
            return float(self.weighted_inventory.sum())
        elif hasattr(self, "normalized_inventory"):
            return float(self.normalized_inventory.sum())
        else:
            return float(self.characterized_inventory.sum())

    #########################
    ### Redo calculations ###
    #########################

    def _switch(
        self,
        obj: Union[tuple, Iterable[Union[FS, bwp.DatapackageBase]]],
        label: str,
        matrix: str,
        func: Callable,
    ) -> None:
        """Switch a method, weighting, or normalization"""
        if isinstance(obj, tuple):
            self.ensure_bw2data_available()
            _, data_objs, _ = prepare_lca_inputs(**{label: obj})
            setattr(self, label, obj)
        else:
            data_objs = list(obj)
        self.packages = [
            pkg.exclude({"matrix": matrix}) for pkg in self.packages
        ] + data_objs
        func(data_objs=data_objs)

        logger.info(
            f"""Switched LCIA {label}. data_objs: {data_objs}""",
            extra={
                "data_objs": str(data_objs),
                "utc": datetime.datetime.utcnow(),
            },
        )

    def switch_method(
        self, method=Union[tuple, Iterable[Union[FS, bwp.DatapackageBase]]]
    ) -> None:
        """Load a new method and replace ``.characterization_mm`` and ``.characterization_matrix``.

        Does not do any new calculations or change ``.characterized_inventory``."""
        self._switch(
            obj=method,
            label="method",
            matrix="characterization_matrix",
            func=self.load_lcia_data,
        )

    def switch_normalization(
        self, normalization=Union[tuple, Iterable[Union[FS, bwp.DatapackageBase]]]
    ) -> None:
        """Load a new normalization and replace ``.normalization_mm`` and ``.normalization_matrix``.

        Does not do any new calculations or change ``.normalized_inventory``."""
        self._switch(
            obj=normalization,
            label="normalization",
            matrix="normalization_matrix",
            func=self.load_normalization_data,
        )

    def switch_weighting(
        self, weighting=Union[tuple, Iterable[Union[FS, bwp.DatapackageBase]]]
    ) -> None:
        """Load a new weighting and replace ``.weighting_mm`` and ``.weighting_matrix``.

        Does not do any new calculations or change ``.weighted_inventory``."""
        self._switch(
            obj=weighting,
            label="weighting",
            matrix="weighting_matrix",
            func=self.load_weighting_data,
        )

    def invert_technosphere_matrix(self):
        """Use pardiso to efficiently calculate the inverse of the technosphere matrix."""
        assert hasattr(self, "inventory"), "Must do lci first"
        assert PYPARDISO, "pardiso solver needed for efficient matrix inversion"

        MESSAGE = """Technosphere matrix inversion is often not the most efficient approach.
    See https://github.com/brightway-lca/brightway2-calc/issues/35"""
        warnings.warn(MESSAGE)

        self.inverted_technosphere_matrix = spsolve(
            self.technosphere_matrix, np.eye(*self.technosphere_matrix.shape)
        )
        return self.inverted_technosphere_matrix

    def __check_demand(self, demand: Optional[dict] = None):
        if demand is None:
            return
        else:
            for key in demand:
                if key not in self.dicts.product and not isinstance(key, int):
                    raise KeyError(f"Key '{key}' not in product dictionary; make sure to pass the integer id, not a key like `('foo', 'bar')` or an `Actiivity` or `Node` object.")

    def redo_lci(self, demand: Optional[dict] = None) -> None:
        """Redo LCI with same databases but different demand.

        Args:
            * *demand* (dict): A demand dictionary.

        Doesn't return anything, but overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        .. warning:: If you want to redo the LCIA as well, use ``redo_lcia(demand)`` directly.

        """
        warnings.warn('Please use .lci(demand=demand) instead of `redo_lci`.', DeprecationWarning)
        self.lci(demand=demand)

    def redo_lcia(self, demand: Optional[dict] = None) -> None:
        """Redo LCIA, optionally with new demand.

        Args:
            * *demand* (dict, optional): New demand dictionary. Optional, defaults to ``self.demand``.

        Doesn't return anything, but overwrites ``self.characterized_inventory``. If ``demand`` is given, also overwrites ``self.demand_array``, ``self.supply_array``, and ``self.inventory``.

        """
        warnings.warn('Please use .lcia(demand=demand) instead of `redo_lci`.', DeprecationWarning)
        self.lcia(demand=demand)

    def to_dataframe(self, matrix_label: str = "characterized_inventory", row_dict: Optional[dict] = None, col_dict: Optional[dict] = None, annotate: bool = True, cutoff: Number = 200, cutoff_mode: str = "number") -> pd.DataFrame:
        """Return all nonzero elements of the given matrix as a Pandas dataframe.

        The LCA class instance must have the matrix ``matrix_label`` already; common labels are:

        * characterized_inventory
        * inventory
        * technosphere_matrix
        * biosphere_matrix
        * characterization_matrix

        For these common matrices, we already have ``row_dict`` and ``col_dict`` which link row and column indices to database ids. For other matrices, or if you have a custom mapping dictionary, override ``row_dict`` and/or ``col_dict``. They have the form ``{matrix index: identifier}``.

        If ``bw2data`` is installed, this function will try to look up metadata on the row and column objects. To turn this off, set ``annotate`` to ``False``.

        Instead of returning all possible values, you can apply a cutoff. This cutoff can be specified in two ways, controlled by ``cutoff_mode``, which should be either ``fraction`` or ``number``.

        If ``cutoff_mode`` is ``number`` (the default), then ``cutoff`` is the number of rows in the DataFrame. Data values are first sorted by their absolute value, and then the largest ``cutoff`` are taken.

        If ``cutoff_mode`` is ``fraction``, then only values whose absolute value is greater than ``cutoff * total_score`` are taken. ``cutoff`` must be between 0 and 1.

        The returned DataFrame will have the following columns:

        * amount
        * col_index
        * row_index

        If row or columns dictionaries are available, the following columns are added:

        * col_id
        * row_id

        If ``bw2data`` is available, then the following columns are added:

        * col_code
        * col_database
        * col_location
        * col_name
        * col_reference_product
        * col_type
        * col_unit
        * row_categories
        * row_code
        * row_database
        * row_location
        * row_name
        * row_type
        * row_unit
        * source_product

        Returns a pandas ``DataFrame``.

        """
        matrix = getattr(self, matrix_label).tocoo()

        dict_mapping = {
            'characterized_inventory': (self.dicts.biosphere.reversed, self.dicts.activity.reversed),
            'inventory': (self.dicts.biosphere.reversed, self.dicts.activity.reversed),
            'technosphere_matrix': (self.dicts.product.reversed, self.dicts.activity.reversed),
            'biosphere_matrix': (self.dicts.biosphere.reversed, self.dicts.activity.reversed),
            'characterization_matrix': (self.dicts.biosphere.reversed, self.dicts.biosphere.reversed),
        }
        if not row_dict:
            try:
                row_dict, _ = dict_mapping[matrix_label]
            except KeyError:
                row_dict = None
        if not col_dict:
            try:
                _, col_dict = dict_mapping[matrix_label]
            except KeyError:
                col_dict = None

        sorter = np.argsort(np.abs(matrix.data))[::-1]
        matrix.data = matrix.data[sorter]
        matrix.row = matrix.row[sorter]
        matrix.col = matrix.col[sorter]

        if cutoff is not None:
            if cutoff_mode == 'fraction':
                if not (0 < cutoff < 1):
                    raise ValueError("fraction `cutoff` value must be between 0 and 1")
                total = matrix.data.sum()
                mask = (np.abs(matrix.data) > (total * cutoff))
                matrix.data = matrix.data[mask]
                matrix.row = matrix.row[mask]
                matrix.col = matrix.col[mask]
            elif cutoff_mode == 'number':
                matrix.data = matrix.data[:int(cutoff)]
                matrix.row = matrix.row[:int(cutoff)]
                matrix.col = matrix.col[:int(cutoff)]
            else:
                raise ValueError("Can't understand cutoff mode")

        df_data = {
            'row_index': matrix.row,
            'col_index': matrix.col,
            'amount': matrix.data,
        }
        if row_dict:
            df_data['row_id'] = np.array([row_dict[i] for i in matrix.row])
        if col_dict:
            df_data['col_id'] = np.array([col_dict[i] for i in matrix.col])
        df = pd.DataFrame(df_data)

        def metadata_dataframe(objs, prefix):
            def dict_for_obj(obj, prefix):
                dct = {
                    f"{prefix}id": obj["id"],
                    f"{prefix}database": obj["database"],
                    f"{prefix}code": obj["code"],
                    f"{prefix}name": obj.get("name"),
                    f"{prefix}location": obj.get("location"),
                    f"{prefix}unit": obj.get("unit"),
                    f"{prefix}type":  obj.get("type", "process"),
                }
                if prefix == "col_":
                    dct["col_reference_product"] = obj.get("reference product")
                else:
                    dct["row_categories"] = (
                        "::".join(obj["categories"]) if obj.get("categories") else None
                    )
                    dct["row_product"] = obj.get("reference product")
                return dct

            return pd.DataFrame(
                [dict_for_obj(obj, prefix) for obj in objs]
            )

        if get_node and annotate:
            if row_dict:
                row_metadata_df = metadata_dataframe(
                    objs=[get_node(id=i) for i in np.unique(df_data['row_id'])],
                    prefix="row_",
                )
                df = df.merge(row_metadata_df, on="row_id")
            if col_dict:
                col_metadata_df = metadata_dataframe(
                    objs=[get_node(id=i) for i in np.unique(df_data['col_id'])],
                    prefix="col_",
                )
                df = df.merge(col_metadata_df, on="col_id")

        return df

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
