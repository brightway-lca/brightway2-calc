import warnings
from collections.abc import Iterator
from functools import partial
from typing import Optional, Tuple

import matrix_utils as mu
import numpy as np

from . import PYPARDISO, factorized, spsolve
from .errors import EmptyBiosphere, NonsquareTechnosphere


class LCABase(Iterator):
    """Base class for single and multi LCA classes"""

    def keep_first_iteration(self):
        """Set a flag to use the current values as first element when
        iterating.

        When creating the class instance, we already use the first index. This
        method allows us to use the values for the first index.

        Note that the methods ``.lci_calculation()`` and
        ``.lcia_calculation()`` will be called on the current values, even if
        these calculations have already been done.
        """
        self.keep_first_iteration_flag = True

    def check_selective_use(self, matrix_label: str) -> Tuple[bool, bool]:
        return (
            self.selective_use.get(matrix_label, {}).get("use_arrays", self.use_arrays),
            self.selective_use.get(matrix_label, {}).get(
                "use_distributions", self.use_distributions
            ),
        )

    def load_lci_data(self, nonsquare_ok=False) -> None:
        """Load inventory data and create technosphere and biosphere matrices."""
        use_arrays, use_distributions = self.check_selective_use("technosphere_matrix")

        self.technosphere_mm = mu.MappedMatrix(
            packages=self.packages,
            matrix="technosphere_matrix",
            use_arrays=use_arrays,
            use_distributions=use_distributions,
            seed_override=self.seed_override,
        )
        self.dicts.product = partial(self.technosphere_mm.row_mapper.to_dict)
        self.dicts.activity = partial(self.technosphere_mm.col_mapper.to_dict)
        self.technosphere_matrix = self.technosphere_mm.matrix

        # Avoid this conversion each time we do a calculation in the future
        # See https://github.com/haasad/PyPardiso/issues/75#issuecomment-2186825609
        if PYPARDISO:
            self.technosphere_matrix = self.technosphere_matrix.tocsr()

        if (
            len(self.technosphere_mm.row_mapper) != len(self.technosphere_mm.col_mapper)
            and not nonsquare_ok
        ):
            raise NonsquareTechnosphere(
                (
                    "Technosphere matrix is not square: {} activities "
                    "(columns) and {} products (rows). Use LeastSquaresLCA to "
                    "solve this system, or fix the input data"
                ).format(
                    len(self.technosphere_mm.col_mapper),
                    len(self.technosphere_mm.row_mapper),
                )
            )

        use_arrays, use_distributions = self.check_selective_use("biosphere_matrix")

        self.biosphere_mm = mu.MappedMatrix(
            packages=self.packages,
            matrix="biosphere_matrix",
            use_arrays=use_arrays,
            use_distributions=use_distributions,
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
        """Remap ``self.dicts.activity|product|biosphere`` and ``self.demand``
        from database integer IDs to keys (``(database name, code)``).

        Uses remapping dictionaries in ``self.remapping_dicts``."""
        if getattr(self, "_remapped", False):
            warnings.warn("Remapping has already been done; returning without changing data")
            return

        if "product" in self.remapping_dicts:
            self.demand = {self.remapping_dicts["product"][k]: v for k, v in self.demand.items()}

        for label in ("activity", "product", "biosphere"):
            if label in self.remapping_dicts:
                getattr(self.dicts, label).remap(self.remapping_dicts[label])

        self._remapped = True

    def decompose_technosphere(self) -> None:
        """
        Factorize the technosphere matrix into lower and upper triangular
        matrices, :math:`A=LU`. Does not solve the linear system :math:`Ax=B`.

        Doesn't return anything, but creates ``self.solver``.

        .. warning:: Incorrect results could occur if a technosphere matrix was
        factorized, and then a new technosphere matrix was constructed, as
        ``self.solver`` would still be the factorized older technosphere
        matrix. You are responsible for deleting ``self.solver`` when doing
        these types of advanced calculations.

        """
        if PYPARDISO:
            warnings.warn("PARDISO installed; this is a no-op")
        else:
            self.solver = factorized(self.technosphere_matrix)

    def solve_linear_system(self, demand: Optional[np.ndarray] = None) -> None:
        """
        Master solution function for linear system :math:`Ax=B`.

            To most numerical analysts, matrix inversion is a sin.

            -- Nicolas Higham, Accuracy and Stability of Numerical Algorithms,
            Society for Industrial and Applied Mathematics, Philadelphia, PA,
            USA, 2002, p. 260.

        We use `pypardiso <https://github.com/haasad/PyPardisoProject>`_ or
        `UMFpack <http://www.cise.ufl.edu/research/sparse/umfpack/>`_, which is
        a very fast solver for sparse matrices.

        If the technosphere matrix has already been factorized, then the
        decomposed technosphere (``self.solver``) is reused. Otherwise the
        calculation is redone completely.

        """
        if demand is None:
            demand = self.demand_array
        if hasattr(self, "solver"):
            return self.solver(demand)
        else:
            return spsolve(self.technosphere_matrix, demand)

    def lci(self, demand: Optional[dict] = None, factorize: bool = False) -> None:
        """
        Calculate a life cycle inventory.

        #. Load LCI data, and construct the technosphere and biosphere
            matrices.
        #. Build the demand array
        #. Solve the linear system to get the supply array and life cycle
            inventory.

        Args:
            * *factorize* (bool, optional): Factorize the technosphere matrix.
            Makes additional calculations with the same technosphere matrix
            much faster. Default is ``False``; not useful is only doing one LCI
            calculation.
            * *builder* (``MatrixBuilder`` object, optional): Default is
            ``bw2calc.matrices.MatrixBuilder``, which is fine for most cases.
            Custom matrix builders can be used to manipulate data in creative
            ways before building the matrices.

        Doesn't return anything, but creates ``self.supply_array`` and
        ``self.inventory``.

        """
        if not hasattr(self, "technosphere_matrix"):
            self.load_lci_data()
        if demand is not None:
            self.check_demand(demand)
            self.build_demand_array(demand)
            self.demand = demand
        else:
            self.build_demand_array()
        if factorize and not PYPARDISO:
            self.decompose_technosphere()
        self.lci_calculation()

    def lcia(self, demand: Optional[dict] = None) -> None:
        """
        Calculate the life cycle impact assessment.

        #. Load and construct the characterization matrix
        #. Multiply the characterization matrix by the life cycle inventory

        Doesn't return anything, but creates ``self.characterized_inventory``.

        """
        assert hasattr(self, "inventory") or hasattr(self, "inventories"), "Must do lci first"
        if not self.dicts.biosphere:
            raise EmptyBiosphere

        if not (
            hasattr(self, "characterization_matrix") or hasattr(self, "characterization_matrices")
        ):
            self.load_lcia_data()
        if demand is not None:
            self.check_demand(demand)
            self.lci(demand=demand)
            self.demand = demand
        self.lcia_calculation()

    def normalize(self) -> None:
        """
        Multiply characterized inventory by flow-specific normalization factors.
        """
        if not (
            hasattr(self, "characterized_inventory") or hasattr(self, "characterized_inventories")
        ):
            raise ValueError("Must do lcia first")
        if not hasattr(self, "normalization_matrix"):
            self.load_normalization_data()
        self.normalization_calculation()

    def weight(self) -> None:
        """Multiply characterized inventory by weighting value.

        Can be done with or without normalization."""
        if not (
            hasattr(self, "characterized_inventory") or hasattr(self, "characterized_inventories")
        ):
            raise ValueError("Must do lcia first")
        if not hasattr(self, "weighting_value"):
            self.load_weighting_data()
        self.weighting_calculation()

    def invert_technosphere_matrix(self):
        """Use one-shot approach to efficiently calculate the inverse of the
        technosphere matrix by simultaneously solving ``Ax=b`` for all ``b``.

        Technosphere matrix inversion is often not the most efficient approach.
        See https://github.com/brightway-lca/brightway2-calc/issues/35

        See `Intel forum <https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/ How-to-find-inverse-of-a-sparse-matrix-using-pardiso/m-p/1165970#M28249>`__
        for a discussion on why we use this approach."""  # noqa: E501
        assert hasattr(self, "inventory"), "Must do lci first"

        if not PYPARDISO:
            warnings.warn(
                "Performance is much better with pypardiso (not available on MacOS ARM machines)"
            )

        self.inverted_technosphere_matrix = spsolve(
            self.technosphere_matrix, np.eye(*self.technosphere_matrix.shape)
        )
        return self.inverted_technosphere_matrix

    def has(self, label: str) -> bool:
        """Shortcut to find out if matrix data for type ``{label}_matrix`` is
        present in the given data objects.

        Returns a boolean. Will return ``True`` even if data for a
        zero-dimensional matrix is given.
        """
        return any(
            True
            for package in self.packages
            for resource in package.resources
            if resource["matrix"] == f"{label}_matrix"
        )

    #################
    # Compatibility #
    #################

    @property
    def activity_dict(self):
        warnings.warn(
            "This method is deprecated, please use `.dicts.activity` instead",
            DeprecationWarning,
        )
        return self.dicts.activity

    @property
    def product_dict(self):
        warnings.warn(
            "This method is deprecated, please use `.dicts.product` instead",
            DeprecationWarning,
        )
        return self.dicts.product

    @property
    def biosphere_dict(self):
        warnings.warn(
            "This method is deprecated, please use `.dicts.biosphere` instead",
            DeprecationWarning,
        )
        return self.dicts.biosphere

    def reverse_dict(self):
        warnings.warn(
            "This method is deprecated, please use `.dicts.X.reversed` directly",
            DeprecationWarning,
        )
        return (
            self.dicts.activity.reversed,
            self.dicts.product.reversed,
            self.dicts.biosphere.reversed,
        )

    def redo_lci(self, demand: Optional[dict] = None) -> None:
        """Redo LCI with same databases but different demand.

        Args:
            * *demand* (dict): A demand dictionary.

        Doesn't return anything, but overwrites ``self.demand_array``,
        ``self.supply_array``, and ``self.inventory``.

        .. warning:: If you want to redo the LCIA as well, use
        ``redo_lcia(demand)`` directly.

        """
        warnings.warn("Please use .lci(demand=demand) instead of `redo_lci`.", DeprecationWarning)
        self.lci(demand=demand)

    def redo_lcia(self, demand: Optional[dict] = None) -> None:
        """Redo LCIA, optionally with new demand.

        Args:
            * *demand* (dict, optional): New demand dictionary. Optional,
            defaults to ``self.demand``.

        Doesn't return anything, but overwrites
        ``self.characterized_inventory``. If ``demand`` is given, also
        overwrites ``self.demand_array``, ``self.supply_array``, and
        ``self.inventory``.

        """
        warnings.warn("Please use .lcia(demand=demand) instead of `redo_lci`.", DeprecationWarning)
        self.lcia(demand=demand)

    def weighting(self) -> None:
        """
        Backwards compatibility. Switching to verb form consistent with
        ``.normalize``.
        """
        warnings.warn("Please switch to `.weight`", DeprecationWarning)
        return self.weight()
