import datetime
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Optional, Union

import bw_processing as bwp
import matrix_utils as mu
import numpy as np
from fs.base import FS

from . import PYPARDISO, __version__
from .dictionary_manager import DictionaryManager
from .errors import InconsistentLCIADatapackages
from .lca import LCA
from .utils import get_datapackage, wrap_functional_unit

logger = logging.getLogger("bw2calc")


class MultiLCA(LCA):
    """
    Perform LCA on multiple demands and multiple impact categories.

    Builds only *one* technosphere and biosphere matrix which can cover all demands.

    Differs from the base `LCA` class in that:

    * `supply_arrays` in place of `supply array`; Numpy array with dimensions (biosphere flows, demand dictionaries)
    * `inventories` in place of `inventory`; List of matrices with length number of demand dictionaries
    * `characterized_inventories` in place of `characterized_inventory`; like `inventories`
    * `scores` instead of `score`; Numpy array with dimensions (demand dictionaries, impact categories)

    The input arguments are also different, and are mostly plural.

    Instantiation requires separate `data_objs` for the method, normalization, and weighting, as these are all impact category-specific. Use a `bw2data` compatibility function to get data prepared in the correct way.

    This class supports both stochastic and static LCA, and can use a variety of ways to describe uncertainty. The input flags `use_arrays` and `use_distributions` control some of this stochastic behaviour. See the [documentation for `matrix_utils`](https://github.com/brightway-lca/matrix_utils) for more information on the technical implementation.

    Parameters
    ----------
    demands : Iterable[dict[int: float]]
        The demands for which the LCA will be calculated. The keys **must be** integers ids.
    inventory_data_objs : list[bw_processing.Datapackage]
        List of `bw_processing.Datapackage` objects **for the inventory**. Can be loaded via `bw2data.prepare_lca_inputs` or constructed manually. Should include data for all needed matrices.
    methods : List[bw_processing.Datapackage]
        Tuple defining the LCIA method, such as `('foo', 'bar')`. Only needed if not passing `data_objs`.
    weighting : List[bw_processing.Datapackage]
        Tuple defining the LCIA weighting, such as `('foo', 'bar')`. Only needed if not passing `data_objs`.
    weighting : List[bw_processing.Datapackage]
        String defining the LCIA normalization, such as `'foo'`. Only needed if not passing `data_objs`.
    remapping_dicts : dict[str : dict]
        Dict of remapping dictionaries that link Brightway `Node` ids to `(database, code)` tuples. `remapping_dicts` can provide such remapping for any of `activity`, `product`, `biosphere`.
    log_config : dict
        Optional arguments to pass to logging. Not yet implemented.
    seed_override : int
        RNG seed to use in place of `Datapackage` seed, if any.
    use_arrays : bool
        Use arrays instead of vectors from the given `data_objs`
    use_distributions : bool
        Use probability distributions from the given `data_objs`
    selective_use : dict[str : dict]
        Dictionary that gives more control on whether `use_arrays` or `use_distributions` should be used. Has the form `{matrix_label: {"use_arrays"|"use_distributions": bool}`. Standard matrix labels are `technosphere_matrix`, `biosphere_matrix`, and `characterization_matrix`.
    """

    def __init__(
        self,
        demands: Iterable[Mapping],
        inventory_data_objs: Iterable[Union[Path, FS, bwp.DatapackageBase]],
        method_data_objs: Optional[
            Iterable[Union[Path, FS, bwp.DatapackageBase]]
        ] = None,
        normalization_data_objs: Optional[
            Iterable[Union[Path, FS, bwp.DatapackageBase]]
        ] = None,
        weighting_data_objs: Optional[
            Iterable[Union[Path, FS, bwp.DatapackageBase]]
        ] = None,
        remapping_dicts: Optional[Iterable[dict]] = None,
        log_config: Optional[dict] = None,
        seed_override: Optional[int] = None,
        use_arrays: Optional[bool] = False,
        use_distributions: Optional[bool] = False,
        selective_use: Optional[dict] = False,
    ):
        # Resolve potential iterator
        self.demands = list(demands)
        for i, fu in enumerate(self.demands):
            if not isinstance(fu, Mapping):
                raise ValueError(f"Demand section {i}: {fu} not a dictionary")

        self.packages = [get_datapackage(obj) for obj in inventory_data_objs]

        if method_data_objs is not None:
            self.method_packages = [get_datapackage(obj) for obj in method_data_objs]
        else:
            self.method_packages = []
        if weighting_data_objs is not None:
            self.weighting_packages = [
                get_datapackage(obj) for obj in weighting_data_objs
            ]
        else:
            self.method_packages = []
        if normalization_data_objs is not None:
            self.normalization_packages = [
                get_datapackage(obj) for obj in normalization_data_objs
            ]
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

        message = """Initialized MultiLCA object. Demands: {demands}, data_objs: {data_objs}""".format(
            demand=self.demands, data_objs=self.packages
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
