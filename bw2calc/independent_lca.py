# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from eight import *
import os


class IndepentLCAMixin(object):
    """Mixin that allows `method`, etc. to be filepaths instead of DataStore object names.

    Removes dependency on `bw2data`."""
    def get_array_filepaths(self):
        """Pass through already correct values"""
        assert self.database_filepath, "Must specify `database_filepath` in independent LCA"
        for collection in (self.database_filepath, self.method, self.weighting, self.normalization):
            if collection is not None:
                for fp in collection:
                    assert os.path.exists(fp), "Can't find file {}".format(fp)
        return (
            self.database_filepath,
            self.method,
            self.weighting,
            self.normalization,
        )

    def fix_dictionaries(self):
        """Don't adjust dictionaries even if ``bw2data`` is present, as functional unit is an integer."""
        self._activity_dict = self.activity_dict
        self._product_dict = self.product_dict
        self._biosphere_dict = self.biosphere_dict
