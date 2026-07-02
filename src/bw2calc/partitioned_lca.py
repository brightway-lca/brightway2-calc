from collections.abc import Iterator
from typing import Optional

import bw_processing as bwp
import matrix_utils as mu
import numpy as np

from bw2calc.errors import (
    CyclicDependencyGraph,
    DemandInStaticDatabase,
    MissingDatabaseDependencies,
    OutsideTechnosphere,
    StaticDependsOnStochastic,
)
from bw2calc.lca import LCA
from bw2calc.utils import get_datapackage


def _find_production_exchanges(mm: mu.MappedMatrix):
    """Run all production-exchange heuristics and return whatever was found.

    Same logic as ``bw_graph_tools.guess_production_exchanges`` but does not raise
    ``UnclearProductionExchange`` when some columns are unresolved.  This is intentional:
    we call this on a non-square stochastic-only matrix whose unresolved rows are exactly
    the interface products we want to identify.

    Retains the other checks from ``guess_production_exchanges``: raises ``ValueError``
    for an empty matrix or mismatched row/column result arrays.

    Returns ``(row_indices, col_indices)`` — integer arrays of matrix row/column indices
    where production exchanges were found.  Unresolved columns are simply absent.
    """
    from bw_graph_tools.matrix_tools import (  # noqa: PLC0415
        gpe_fifth_heuristic,
        gpe_first_heuristic,
        gpe_fourth_heuristic,
        gpe_second_heuristic,
        gpe_third_heuristic,
    )

    if mm.matrix.shape[0] == 0:
        raise ValueError("Empty matrix")

    row, col = gpe_first_heuristic(mm)
    row, col = gpe_second_heuristic(mm, row, col)
    row, col = gpe_third_heuristic(mm, row, col)
    row, col = gpe_fourth_heuristic(mm, row, col)
    row, col = gpe_fifth_heuristic(mm, row, col)

    if row.shape != col.shape:
        raise ValueError("Guessed row indices do not match guessed column indices.")

    return row, col


class PartitionedMonteCarloLCA(Iterator):
    """Monte Carlo LCA that pre-solves a static background system once.

    Splits the full system into a static (background) part and a stochastic (foreground) part.
    The static system is solved deterministically for each product demanded across the
    static/stochastic boundary (interface products), producing aggregated biosphere vectors.
    These are stored in an in-memory dynamic datapackage that is combined with the stochastic
    packages for each Monte Carlo iteration.

    This avoids rebuilding and solving the (typically large) background matrix on every
    iteration — only the foreground matrix is resampled.

    Parameters
    ----------
    demand : dict
        Functional unit: ``{activity_or_product_id: amount}``. Must be in the stochastic system.
    static_databases : list[str]
        Names of databases to treat as static (e.g. ``["biosphere3", "ecoinvent 3.10"]``).
    data_objs : list
        All datapackages: stochastic LCI + static LCI + LCIA method. Packages for databases
        listed in ``static_databases`` are identified by their ``metadata["name"]`` field, which
        must equal ``bw_processing.clean_datapackage_name(database_name)``.
    seed_override : int, optional
        RNG seed passed to the inner stochastic LCA.

    Notes
    -----
    All LCI datapackages must contain a ``database_dependencies`` key in their metadata,
    which is written by ``bw2data >= 4.7``.

    Design note: composition vs. subclassing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This class uses **composition**: it builds an internal ``._lca`` instance from the
    stochastic packages plus the pre-solved dynamic datapackage, and delegates properties
    to it.  An alternative would be to subclass ``LCA`` and override ``load_lci_data()``
    to perform the same partitioning — classify packages, pre-solve the static system,
    inject the dynamic datapackage, replace ``self.packages``, then call
    ``super().load_lci_data()``.  That approach would give full inheritance of all current
    and future ``LCA`` attributes without explicit delegation.

    The reason composition was chosen instead:

    * **Package list mutation.** The override would need to replace ``self.packages``
      (set from ``data_objs`` in ``__init__``) with the filtered stochastic+dynamic list
      mid-lifecycle, which is non-obvious and makes ``self.packages`` inconsistent with
      ``self.data_objs``.
    * **Matrix semantics.**  Whether composed or subclassed, ``technosphere_matrix`` is
      the *reduced* aggregated-proxy matrix, not the full combined static+stochastic
      system.  Subclassing makes this less visible rather than resolving it.
    """

    def __init__(
        self,
        demand: dict[int, float],
        static_databases: list,
        data_objs: list,
        seed_override: Optional[int] = None,
    ):
        if not all(isinstance(k, int) for k in demand):
            raise TypeError("All demand keys must be integers (activity/product database IDs).")
        self.demand = demand
        self.static_databases = list(static_databases)
        self.seed_override = seed_override

        packages = [get_datapackage(obj) for obj in data_objs]
        (
            self.static_packages,
            self.stochastic_packages,
            self.method_packages,
        ) = self._classify_packages(packages)
        self._validate()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def lci(self) -> None:
        """Pre-solve the static system, build the dynamic datapackage, and run the first LCI."""
        dynamic_dp = self._build_dynamic_datapackage()

        self._lca = LCA(
            demand=self.demand,
            data_objs=self.stochastic_packages + self.method_packages + [dynamic_dp],
            use_distributions=True,
            seed_override=self.seed_override,
        )
        self._lca.lci()

    def lcia(self) -> None:
        self._lca.lcia()

    def keep_first_iteration(self) -> None:
        self._lca.keep_first_iteration()

    def __next__(self) -> None:
        next(self._lca)

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def score(self) -> float:
        return self._lca.score

    @property
    def inventory(self):
        return self._lca.inventory

    @property
    def supply_array(self):
        return self._lca.supply_array

    @property
    def characterized_inventory(self):
        return self._lca.characterized_inventory

    @property
    def dicts(self):
        return self._lca.dicts

    @property
    def technosphere_matrix(self):
        return self._lca.technosphere_matrix

    @property
    def biosphere_matrix(self):
        return self._lca.biosphere_matrix

    @property
    def characterization_matrix(self):
        return self._lca.characterization_matrix

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _classify_packages(self, packages):
        """Classify packages into static LCI, stochastic LCI, and method buckets.

        Classification happens at the resource-group level so that a single datapackage
        containing groups for different matrices or different databases is split correctly.
        Each resource group has a single matrix label, so we use ``dp.groups`` to iterate
        and ``dp.exclude({"group": name})`` to produce filtered sub-packages.

        Group-to-database matching uses the bw2data naming convention where a group is
        named ``clean_datapackage_name(db_name + " " + matrix_type)`` — i.e. it starts
        with ``clean_datapackage_name(db_name)`` followed by ``_``.  If no group-name
        match is found, the check falls back to the package-level ``metadata["name"]``.
        """
        static_names = {bwp.clean_datapackage_name(db) for db in self.static_databases}

        static, stochastic, method = [], [], []
        for dp in packages:
            for group_name, group_dp in dp.groups.items():
                matrix = group_dp.resources[0].get("matrix", "")
                filtered = dp.filter_by_attribute("group", group_name)

                if matrix == "characterization_matrix":
                    method.append(filtered)
                elif matrix in ("technosphere_matrix", "biosphere_matrix"):
                    # bw2data names groups as clean_datapackage_name(db + " " + matrix_type),
                    # so a static group name starts with clean_datapackage_name(db) + "_".
                    # Fall back to the package-level name for the one-package-per-database case.
                    is_static = any(
                        group_name == sn or group_name.startswith(sn + "_") for sn in static_names
                    )
                    if not is_static:
                        is_static = dp.metadata.get("name", "") in static_names
                    (static if is_static else stochastic).append(filtered)

        return static, stochastic, method

    def _validate(self) -> None:
        for dp in self.static_packages + self.stochastic_packages:
            if "database_dependencies" not in dp.metadata:
                raise MissingDatabaseDependencies(
                    f"Package '{dp.metadata.get('name')}' is missing 'database_dependencies' "
                    "metadata. Reprocess the database with bw2data >= 4.7."
                )

        stochastic_names = {dp.metadata.get("name", "") for dp in self.stochastic_packages}

        for dp in self.static_packages:
            for dep in dp.metadata.get("database_dependencies", []):
                if bwp.clean_datapackage_name(dep) in stochastic_names:
                    raise StaticDependsOnStochastic(
                        f"Static database '{dp.metadata.get('name')}' depends on stochastic "
                        f"database '{dep}'. Static databases must only depend on other static "
                        "databases."
                    )

        graph: dict[str, set] = {}
        for dp in self.static_packages + self.stochastic_packages:
            name = dp.metadata.get("name", "")
            deps = {
                bwp.clean_datapackage_name(d) for d in dp.metadata.get("database_dependencies", [])
            }
            graph.setdefault(name, set()).update(deps)
        self._check_for_cycles({k: list(v) for k, v in graph.items()})

    @staticmethod
    def _check_for_cycles(graph: dict) -> None:
        visited: set = set()
        in_stack: set = set()

        def dfs(node: str) -> None:
            visited.add(node)
            in_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in in_stack:
                    raise CyclicDependencyGraph(
                        f"Cycle detected in database dependency graph involving '{node}'"
                    )
            in_stack.discard(node)

        for node in graph:
            if node not in visited:
                dfs(node)

    def _build_dynamic_datapackage(self) -> bwp.DatapackageBase:
        """Solve the static system for each interface product; return dynamic datapackage."""
        dynamic_dp = bwp.create_datapackage(name="partitioned_mc_static_background")

        if not self.static_packages:
            return dynamic_dp

        # Build stochastic technosphere (deterministic) to identify interface products.
        # We use _find_production_exchanges (not guess_production_exchanges) because the
        # stochastic-only matrix is non-square — the unresolved rows are exactly the
        # interface products we want.
        stochastic_tech_mm = mu.MappedMatrix(
            packages=self.stochastic_packages,
            matrix="technosphere_matrix",
            use_arrays=False,
            use_distributions=False,
        )

        row_existing, col_existing = _find_production_exchanges(stochastic_tech_mm)

        # Map stochastic matrix row index → product db id (used here and below)
        stoch_product_reversed = {v: k for k, v in stochastic_tech_mm.row_mapper.to_dict().items()}

        stochastic_produced_ids = {stoch_product_reversed[r] for r in row_existing}
        for demand_key in self.demand:
            if demand_key not in stochastic_produced_ids:
                raise DemandInStaticDatabase(
                    f"Demand key {demand_key} does not have a production exchange in the "
                    "stochastic system. The functional unit must be in the stochastic "
                    "(foreground) system, not in a static background database."
                )

        all_rows = np.arange(stochastic_tech_mm.matrix.shape[0])
        interface_row_indices = np.setdiff1d(all_rows, row_existing)

        if not interface_row_indices.size:
            return dynamic_dp

        # Build static LCA (deterministic; dummy demand, only matrices needed)
        from bw_graph_tools.matrix_tools import guess_production_exchanges  # noqa: PLC0415

        static_lca = LCA(
            demand={0: 1.0},
            data_objs=self.static_packages,
            use_distributions=False,
        )
        static_lca.load_lci_data()
        static_lca.decompose_technosphere()

        # Map static product db id → static activity db id (via production exchanges)
        static_row_existing, static_col_existing = guess_production_exchanges(
            static_lca.technosphere_mm
        )
        static_product_reversed = static_lca.dicts.product.reversed  # {matrix_idx: db_id}
        static_activity_reversed = static_lca.dicts.activity.reversed  # {matrix_idx: db_id}

        static_producer: dict = {}
        for r, c in zip(static_row_existing, static_col_existing):
            static_producer[static_product_reversed[r]] = static_activity_reversed[c]

        static_biosphere_reversed = static_lca.dicts.biosphere.reversed  # {matrix_idx: db_id}

        tech_rows, tech_cols, tech_data = [], [], []
        bio_rows, bio_cols, bio_data = [], [], []

        for row_idx in interface_row_indices:
            product_id = stoch_product_reversed[row_idx]

            if product_id not in static_producer:
                raise OutsideTechnosphere(
                    f"Interface product with id {product_id} has no identified production "
                    "exchange in the static system. Check that all required databases are "
                    "listed in static_databases."
                )

            proxy_activity_id = static_producer[product_id]

            # Solve static system for 1 unit of this interface product
            demand_array = np.zeros(len(static_lca.dicts.product))
            demand_array[static_lca.dicts.product[product_id]] = 1.0
            supply_array = static_lca.solve_linear_system(demand_array)

            aggregated_bio = np.asarray(static_lca.biosphere_matrix @ supply_array).flatten()

            # Production exchange: proxy_activity produces 1 unit of the interface product
            tech_rows.append(product_id)
            tech_cols.append(proxy_activity_id)
            tech_data.append(1.0)

            # Aggregated biosphere: all non-zero flows from the static supply chain
            for bio_idx in np.flatnonzero(aggregated_bio):
                bio_rows.append(static_biosphere_reversed[bio_idx])
                bio_cols.append(proxy_activity_id)
                bio_data.append(float(aggregated_bio[bio_idx]))

        tech_indices = np.array(list(zip(tech_rows, tech_cols)), dtype=bwp.INDICES_DTYPE)
        dynamic_dp.add_persistent_vector(
            matrix="technosphere_matrix",
            indices_array=tech_indices,
            data_array=np.array(tech_data),
        )

        if bio_rows:
            bio_indices = np.array(list(zip(bio_rows, bio_cols)), dtype=bwp.INDICES_DTYPE)
            dynamic_dp.add_persistent_vector(
                matrix="biosphere_matrix",
                indices_array=bio_indices,
                data_array=np.array(bio_data),
            )

        return dynamic_dp
