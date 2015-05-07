# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

class OutsideTechnosphere(Exception):
    """The given demand array activity is not in the technosphere matrix"""
    pass


class EfficiencyWarning(RuntimeWarning):
    """Least squares is much less efficient than direct computation for square, full-rank matrices"""
    pass


class NoSolutionFound(UserWarning):
    """No solution to set of linear equations found within given constraints"""
    pass


class NonsquareTechnosphere(Exception):
    """The given data do not form a square technosphere matrix"""
    pass
