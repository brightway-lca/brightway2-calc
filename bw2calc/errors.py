# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from eight import *

class BW2CalcError(Exception):
    """Base class for bw2calc errors"""
    pass

class OutsideTechnosphere(BW2CalcError):
    """The given demand array activity is not in the technosphere matrix"""
    pass


class EfficiencyWarning(RuntimeWarning):
    """Least squares is much less efficient than direct computation for square, full-rank matrices"""
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
