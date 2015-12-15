# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from eight import *


class IndepentLCAMixin(object):
    """Mixin that allows `method`, etc. to be filepaths instead of DataStore object names.

    Removes dependency on `bw2data`."""
    def get_array_filepaths(self):
        """Pass through already correct values"""
        assert self._databases, "Must specify `databases` filepaths in independent LCA"
        return (
            self._databases,
            self.method,
            self.weighting,
            self.normalization,
        )
