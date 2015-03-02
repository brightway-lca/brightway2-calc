class OutsideTechnosphere(StandardError):
    """The given demand array activity is not in the technosphere matrix"""
    pass


class EfficiencyWarning(RuntimeWarning):
    """Least squares is much less efficient than direct computation for square, full-rank matrices"""
    pass


class NoSolutionFound(UserWarning):
    """No solution to set of linear equations found within given constraints"""
    pass


class NonsquareTechnosphere(StandardError):
    """The given data do not form a square technosphere matrix"""
    pass
