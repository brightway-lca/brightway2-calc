# -*- coding: utf-8 -*
from __future__ import division
from .lca import LCA
from bw2data.proxies import CompressedSparseMatrixProxy
from scipy import sparse
import numpy as np


class SimpleRegionalizedLCA(LCA):
    """A semi-regionalized LCA calculation where the inventory and impact assessment spatial scales match."""

    def build_characterization_matrix(self, vector=None):
        """Build a regionalized characterization matrix.

        Throughout this calculation, locations refer to integer indices retrieved from the geomapper, not the string codes themselves. We operate directly on the paramter arrays, as this is much more efficient.

        We do this by first retrieving the regionalized characterization factors for each location where characterization factors are avaiable. The intermediate data structure ``regionalized_dict`` has the following structure:

        .. code:: python

            {location_id: (biosphere matrix row number, cf value)}

        We then use the ``np.unique`` function to retrieve all technosphere processes, and an index into ``self.tech_params`` for each of them. We can then use this index to get a location for each technosphere process.

        The characterization matrix has dimensions (number of biosphere flows, number of technosphere flows). For each column, we lookup the location code, and then retrieve the cf amounts and row indices from the ``regionalized_dict``. We can then build the ``characterization_matrix``. 

        .. note:: There is a lot of duplicate data in ``characterization_matrix``, as characterization factors are provided for each technosphere process, regardless of whether that technosphere location has been seen already.

        """
        vector = self.cf_params['amount'] if vector is None else vector
        # count = len(self.biosphere_dict)
        regionalized_dict = {}
        for index in vector.shape[0]:
            regionalized_dict.setdefault(
                int(self.cf_params["geo"][index]), []).append(
                (int(self.cf_params["index"][index]), vector[index]))
        for key, data in regionalized_dict.iteritems():
            regionalized_dict[key] = (
            np.array([x[0] for x in regionalized_dict]),
            np.array([x[1] for x in regionalized_dict]))
        # Get location codes for technosphere processes
        tech_columns, tech_param_indices = np.unique(
            self.tech_params["col"], return_index=True)
        tech_geo_indices = self.tech_params["geo"][tech_param_indices]
        cfs, rows, cols = [], [], []
        for i, col_index in enumerate(tech_columns):
            geo = int(tech_geo_indices[i])
            these_cfs = regionalized_dict[geo][1]
            cfs.append(these_cfs)
            cols.append(np.ones(these_cfs.shape[0]) * col_index)
            rows.append(regionalized_dict[geo][0])
        self.characterization_matrix = sparse.coo_matrix(
            (np.hstack(cfs), (np.hstack(rows), np.hstack(cols))),
            (len(self.biosphere_dict), len(self.technosphere_dict))
            ).tocsr()

    def lcia_calculation(self):
        """Multiply the characterization matrix by the life cycle inventory. Uses ``.multiply`` for point-wise multiplication, as both matrices have the same dimensions."""
        self.characterized_inventory = CompressedSparseMatrixProxy(
            self.characterization_matrix.multiply(self.inventory.data),
            self.biosphere_dict, self.technosphere_dict)
