# -*- coding: utf-8 -*
from __future__ import division
from . import LCA
from brightway2 import mapping
from heapq import heappush, heappop
import numpy as np
from scipy import sparse


class GraphTraversal(object):
    """Master class for graph traversal."""
    def __init__(self, demand, method, cutoff=0.01, max_calc=1e5,
            disaggregate=False):
        self.cutoff = cutoff
        self.max_calc = max_calc
        # Unroll circular references
        self.disaggregate = disaggregate
        self.counter = 0
        self.heap = []

        # Edge format is (to, from, amount)
        # Don't know LCA score of edge until LCA of input is calculated
        self.edges = []
        # nodes is just {id: cumulative LCA score (normalized)}
        self.nodes = {}

        self.lca = LCA(demand, method)
        self.lca.lci()
        self.lca.lcia()
        self.lca.decompose_technosphere()
        self.supply = self.lca.solve_linear_system()
        self.score = self.lca.score
        if self.score == 0:
            raise ValueError("Zero total LCA score makes traversal impossible")

        self.nodes[-1] = self.score
        for activity, value in demand.iteritems():
            index = self.lca.technosphere_dict[mapping[activity]]
            heappush(self.heap, (1, index))
            # -1 is a special index for total demand, which can be
            # composite. Initial edges are inputs to the
            # functional unit.
            self.edges.append((-1, index, value))

        # Create matrix of LCIA CFs times biosphere flows, as these don't
        # change. This is also the unit score of each activity.
        self.characterized_biosphere = np.array(
            (self.lca.characterization_matrix.data * \
            self.lca.biosphere_matrix.data).sum(axis=0)).ravel()

    def cum_score(self, index):
        demand = np.zeros((self.supply.shape[0],))
        demand[index] = 1
        return float((self.characterized_biosphere * self.lca.solver(demand)
            ).sum())

    def unit_score(self, index):
        return float(self.characterized_biosphere[index])

    def calculate(self):
        """
Build a directed graph of the supply chain.

Use a heap queue to store a sorted list of processes that need to be examined,
and traverse the graph using an "importance-first" search.
        """
        while self.heap and self.counter < self.max_calc:
            pscore, pindex = heappop(self.heap)
            col = self.lca.technosphere_matrix.data[:, pindex].tocoo()
            children = [(col.row[i], -1 * col.data[i]) for i in xrange(
                col.row.shape[0])]
            for activity, amount in children:
                # Skip diagonal values
                if activity == pindex:
                    continue
                # Edge format is (to, from, amount)
                self.edges.append((pindex, activity, amount))
                if activity in self.nodes:
                    continue
                score = self.cum_score(activity)
                self.counter += 1
                if abs(score) < abs(self.score * self.cutoff):
                    continue
                self.nodes[activity] = score
                heappush(self.heap, (1. / score, activity))

    def rationalize(self):
        """
Take the raw data and turn it a structured graph.

1. Create a list of nodes

2. Create a list of edges

3. Parse through the list of edges, eliminating nodes which have small individual impacts.
        """
        rev_mapping = dict([(v, k) for k, v in mapping.data.iteritems()])
        rev_tech = dict([(v, k) for k, v in self.lca.technosphere_dict.iteritems()])
        self.nodes = dict([(k, {
            "cum_score": v,
            "ind_score": float(self.characterized_biosphere[k] * \
                self.supply[k]),
            "id": "Functional unit" if k == -1 else rev_mapping[rev_tech[k]]
            }) for k, v in self.nodes.iteritems() if k >= 0])
        to_delete = set([key for key, value in self.nodes.iteritems(
            ) if value["ind_score"] < 0.01 * self.score])
        self.nodes = dict([(k, v) for k, v in self.nodes.iteritems() \
            if k not in to_delete])
        # Edge format is (to, from, amount)
        count = self.characterized_biosphere.shape[0]
        # Ignore functional unit for now
        edges = [x for x in self.edges if x[0] >= 0]
        matrix = sparse.coo_matrix((
            [x[2] for x in edges],
            ([x[1] for x in edges], [x[0] for x in edges])),
            (count, count))
        for node in to_delete:
            col = matrix.tocsc()[:, node].tocoo()
            row = matrix.tocsr()[node, :].tocoo()
            rows = [matrix.row]
            cols = [matrix.col]
            data = [matrix.data]

            for i in xrange(col.data.shape[0]):
                cols.append(row.col)
                rows.append(row.row)
                data.append(col.data[i] * row.data)

            cols = np.hstack(cols)
            mask = cols != node
            matrix = sparse.coo_matrix((np.hstack(data)[mask],
                (np.hstack(rows)[mask], cols[mask])), (count, count)
                ).tocsr().tocoo()

        self.edges = [{
            "source": int(matrix.row[i]),
            "target": int(matrix.col[i]),
            "value": float(self.characterized_biosphere[matrix.row[i]] * \
                matrix.data[i])
            } for i in xrange(matrix.data.shape[0])]
        return matrix
        # TODO: Append functional unit
