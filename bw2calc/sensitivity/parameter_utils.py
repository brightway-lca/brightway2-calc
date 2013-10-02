# -*- coding: utf-8 -*
from ..mc_vector import ParameterVectorLCA
from bw2data import Database


class ParameterNaming(object):
    """Translate parameter indices into something meaningful to humans."""
    def __init__(self, lca):
        assert isinstance(lca, ParameterVectorLCA)
        self.databases = {"biosphere": Database("biosphere").load()}
        self.lca = lca
        self.positions = self.lca.positions
        self.lca.fix_dictionaries()
        self.rt, self.rb = self.lca.reverse_dict()

    def lookup(self, index):
        kind = self.get_kind(index)
        if kind == "weighting":
            return "Weighting"
        elif kind == "bio":
            offset = self.positions['tech'][1]
            row_key = self.rb[self.lca.bio_params[index - offset]['row']]
            input_ds = self.databases["biosphere"][row_key]
            col_key = self.rt[self.lca.bio_params[index - offset]['col']]
            if col_key[0] not in self.databases:
                self.databases[col_key[0]] = Database(col_key[0]).load()
            output_ds = self.databases[col_key[0]][col_key]
            return "Biosphere: %s (%s) to %s" % (
                input_ds['name'],
                "-".join(input_ds['categories']),
                output_ds['name']
            )
        elif kind == "tech":
            row_key = self.rt[self.lca.tech_params[index]['row']]
            if row_key[0] not in self.databases:
                self.databases[row_key[0]] = Database(row_key[0]).load()
            input_ds = self.databases[row_key[0]][row_key]
            col_key = self.rt[self.lca.tech_params[index]['col']]
            if col_key[0] not in self.databases:
                self.databases[col_key[0]] = Database(col_key[0]).load()
            output_ds = self.databases[col_key[0]][col_key]
            return "Technosphere: %s (%s) to %s (%s)" % (
                input_ds['name'],
                input_ds['location'],
                output_ds['name'],
                output_ds['location']
            )
        elif kind == "cf":
            offset = self.positions['bio'][1]
            row_key = self.rb[self.lca.cf_params[index - offset]['index']]
            ds = self.databases["biosphere"][row_key]
            return "CF: %s (%s)" % (ds['name'], "-".join(ds['categories']))

    def get_kind(self, index):
        for key, (lower, upper) in self.positions.iteritems():
            if lower <= index < upper:
                return key
        raise ValueError("Can't find index %s in ``positions``" % index)
