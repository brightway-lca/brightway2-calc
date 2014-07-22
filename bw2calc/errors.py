class OutsideTechnosphere(StandardError):
    """LCA for the given activity, as it is not in the technosphere matrix"""
    pass


class EfficiencyWarning(RuntimeWarning):
    """Least squares is much less efficient than direct computation for square, full-rank matrices"""
    pass


class NoSolutionFound(UserWarning):
    """No solution to set of linear equations found within given constraints"""
    pass
