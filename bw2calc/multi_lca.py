import numpy as np
from bw2data import calculation_setups

from .lca import LCA


class InventoryMatrices:
    def __init__(self, biosphere_matrix, supply_arrays):
        self.biosphere_matrix = biosphere_matrix
        self.supply_arrays = supply_arrays

    def __getitem__(self, fu_index):
        if fu_index is Ellipsis:
            raise ValueError("Must specify integer indices")

        return self.biosphere_matrix * self.supply_arrays[fu_index]


class MultiLCA:
    """Wrapper class for performing LCA calculations with many functional units and LCIA methods.

    Needs to be passed a ``calculation_setup`` name.

    This class does not subclass the `LCA` class, and performs all calculations upon instantiation.

    Initialization creates `self.results`, which is a NumPy array of LCA scores, with rows of functional units and columns of LCIA methods. Ordering is the same as in the `calculation_setup`.

    """

    def __init__(self, cs_name, log_config=None):
        if calculation_setups is None:
            raise ImportError
        assert cs_name in calculation_setups
        try:
            cs = calculation_setups[cs_name]
        except KeyError:
            raise ValueError("{} is not a known `calculation_setup`.".format(cs_name))
        self.func_units = cs["inv"]
        self.methods = cs["ia"]
        self.lca = LCA(demand=self.all, method=self.methods[0], log_config=log_config)
        self.lca.logger.info(
            {
                "message": "Started MultiLCA calculation",
                "methods": list(self.methods),
                "functional units": [wrap_functional_unit(o) for o in self.func_units],
            }
        )
        self.lca.lci()
        self.method_matrices = []
        self.supply_arrays = []
        self.results = np.zeros((len(self.func_units), len(self.methods)))
        for method in self.methods:
            self.lca.switch_method(method)
            self.method_matrices.append(self.lca.characterization_matrix)

        for row, func_unit in enumerate(self.func_units):
            self.lca.redo_lci(func_unit)
            self.supply_arrays.append(self.lca.supply_array)

            for col, cf_matrix in enumerate(self.method_matrices):
                self.lca.characterization_matrix = cf_matrix
                self.lca.lcia_calculation()
                self.results[row, col] = self.lca.score

        self.inventory = InventoryMatrices(
            self.lca.biosphere_matrix, self.supply_arrays
        )

    @property
    def all(self):
        """Get all possible databases by merging all functional units"""
        return {key: 1 for func_unit in self.func_units for key in func_unit}
