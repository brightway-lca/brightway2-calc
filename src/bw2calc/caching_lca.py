from scipy import sparse

from .lca import LCA
from .result_cache import ResultCache


class CachingLCA(LCA):
    """Custom class which caches supply vectors.

    Cache resets upon iteration. If you do weird stuff outside of iteration you should probably
    use the regular LCA class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = ResultCache()

    def __next__(self) -> None:
        self.cache.reset()
        super().__next__(self)

    def lci_calculation(self) -> None:
        """The actual LCI calculation.

        Separated from ``lci`` to be reusable in cases where the matrices are already built, e.g.
        ``redo_lci`` and Monte Carlo classes.

        """
        if hasattr(self, "cache") and len(self.demand) == 1:
            key, value = list(self.demand.items())[0]
            try:
                self.supply_array = self.cache[key] * value
            except KeyError:
                self.supply_array = self.solve_linear_system()
                self.cache.add(key, self.supply_array.reshape((-1, 1)) / value)
        else:
            self.supply_array = self.solve_linear_system()
        # Turn 1-d array into diagonal matrix
        count = len(self.dicts.activity)
        self.inventory = self.biosphere_matrix @ sparse.spdiags(
            [self.supply_array], [0], count, count
        )
