class BW2CalcError(Exception):
    """Base class for bw2calc errors"""

    pass


class OutsideTechnosphere(BW2CalcError):
    """The given demand array activity is not in the technosphere matrix"""

    pass


class EfficiencyWarning(RuntimeWarning):
    """Least squares is much less efficient than direct computation for square, full-rank
    matrices"""

    pass


class NoSolutionFound(UserWarning):
    """No solution to set of linear equations found within given constraints"""

    pass


class NonsquareTechnosphere(BW2CalcError):
    """The given data do not form a square technosphere matrix"""

    pass


class MalformedFunctionalUnit(BW2CalcError):
    """The given functional unit cannot be understood"""

    pass


class EmptyBiosphere(BW2CalcError):
    """Can't do impact assessment with no biosphere flows"""

    pass


class AllArraysEmpty(BW2CalcError):
    """Can't load the numpy arrays if all of them are empty"""

    pass


class NoArrays(BW2CalcError):
    """No arrays for given matrix"""

    pass


class InconsistentGlobalIndex(BW2CalcError):
    """LCIA matrices are diagonal, and use the ``col`` field for regionalization. If multiple LCIA
    datapackages are present, they must use the same value for ``GLO``, the global location, in
    order for filtering for site-generic LCIA to work correctly."""

    pass


class MultipleValues(BW2CalcError):
    """Multiple values are present, but only one value is expected"""

    pass


class InconsistentLCIA(BW2CalcError):
    """Provided weighting or normalization doesn't fit the impact category"""

    pass


class MissingDatabaseDependencies(BW2CalcError):
    """A datapackage is missing the 'database_dependencies' metadata field required for
    partitioned Monte Carlo. Reprocess the database with bw2data >= 4.7."""

    pass


class CyclicDependencyGraph(BW2CalcError):
    """The database dependency graph contains a cycle, making partitioned Monte Carlo impossible"""

    pass


class StaticDependsOnStochastic(BW2CalcError):
    """A database marked as static has a dependency on a database marked as stochastic.
    Static databases must only depend on other static databases."""

    pass


class DemandInStaticDatabase(BW2CalcError):
    """The functional unit demand points to an activity in a static database.
    The demand must be in the stochastic (foreground) system."""

    pass
