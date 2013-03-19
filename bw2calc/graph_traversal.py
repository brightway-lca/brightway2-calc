# -*- coding: utf-8 -*
from __future__ import division
from . import LCA
from brightway2 import mapping, Database
from heapq import heappush, heappop
import numpy as np


class GraphTraversal(object):
    """Master class for graph traversal."""
    def calculate(self, demand, method, cutoff=0.005, max_calc=1e5):
        counter = 0

        lca, supply, score = self.build_lca(demand, method)
        if score == 0:
            raise ValueError("Zero total LCA score makes traversal impossible")

        # Create matrix of LCIA CFs times biosphere flows, as these don't
        # change. This is also the unit score of each activity.
        characterized_biosphere = np.array((
            lca.characterization_matrix.data *
            lca.biosphere_matrix.data).sum(axis=0)).ravel()

        heap, nodes, edges = self.initialize_heap(
            demand, lca, supply, characterized_biosphere)
        nodes, edges, counter = self.traverse(
            heap, nodes, edges, counter, max_calc, cutoff, score, supply,
            characterized_biosphere, lca)
        nodes = self.add_metadata(nodes, lca)

        return {
            'nodes': nodes,
            'edges': edges,
            'lca': lca,
            'counter': counter,
        }

    def initialize_heap(self, demand, lca, supply, characterized_biosphere):
        heap, nodes, edges = [], {}, []
        for activity, value in demand.iteritems():
            index = lca.technosphere_dict[mapping[activity]]
            heappush(heap, (1, index))
            nodes[index] = {
                "amount": supply[index],
                "cum": self.cumulative_score(
                    index, supply, characterized_biosphere, lca),
                "ind": self.unit_score(index, supply, characterized_biosphere)
            }
            # -1 is a special index for total demand, which can be
            # composite. Initial edges are inputs to the
            # functional unit.
            edges.append({
                "to": -1,
                "from": index,
                "amount": value,
                "impact": lca.score,
            })
        return heap, nodes, edges

    def build_lca(self, demand, method):
        lca = LCA(demand, method)
        lca.lci()
        lca.lcia()
        lca.decompose_technosphere()
        return lca, lca.solve_linear_system(), lca.score

    def cumulative_score(self, index, supply, characterized_biosphere, lca):
        demand = np.zeros((supply.shape[0],))
        demand[index] = supply[index]
        return float((characterized_biosphere * lca.solver(demand)).sum())

    def unit_score(self, index, supply, characterized_biosphere):
        return float(characterized_biosphere[index] * supply[index])

    def traverse(self, heap, nodes, edges, counter, max_calc, cutoff,
                 total_score, supply, characterized_biosphere, lca):
        """
Build a directed graph of the supply chain.

Use a heap queue to store a sorted list of processes that need to be examined,
and traverse the graph using an "importance-first" search.
        """
        while heap and counter < max_calc:
            parent_score_inverted, parent_index = heappop(heap)
            # parent_score = 1 / parent_score_inverted
            col = lca.technosphere_matrix.data[:, parent_index].tocoo()
            # Multiply by -1 because technosphere values are negative
            # (consumption of inputs)
            children = [(col.row[i], -1 * col.data[i]) for i in xrange(
                col.row.shape[0])]
            for activity, amount in children:
                # Skip values on technosphere diagonal or coproducts
                if activity == parent_index or amount <= 0:
                    continue
                counter += 1
                cumulative_score = self.cumulative_score(
                    activity, supply, characterized_biosphere, lca)
                if abs(cumulative_score) < abs(total_score * cutoff):
                    continue
                # Edge format is (to, from, mass amount, cumulative impact)
                edges.append({
                    "to": parent_index,
                    "from": activity,
                    # The cumulative impact directly due to this link (weight)
                    # Amount of this link * amount of parent demanding link
                    "amount": amount * nodes[parent_index]["amount"],
                    # Amount of this input
                    "impact": amount * nodes[parent_index]["amount"]
                    # times impact per unit of this input
                    * cumulative_score / supply[activity]
                })
                # Want multiple incoming edges, but don't add existing node
                if activity in nodes:
                    continue
                nodes[activity] = {
                    # Total amount of this flow supplied
                    "amount": supply[activity],
                    # Cumulative score from all flows of this activity
                    "cum": cumulative_score,
                    # Individual score attributable to environmental flows
                    # coming directory from or to this activity
                    "ind": self.unit_score(activity, supply,
                                           characterized_biosphere)
                }
                heappush(heap, (1 / cumulative_score, activity))

        return nodes, edges, counter

    def add_metadata(self, nodes, lca):
        rm = dict([(v, k) for k, v in mapping.data.iteritems()])
        rt = dict([(v, k) for k, v in lca.technosphere_dict.iteritems()])
        lookup = dict([(index, self.get_code(index, rm, rt)) for index in nodes if index != -1])
        new_nodes = [(-1, {
            "code": "fu",
            "cum": lca.score,
            "ind": 1e-6 * lca.score,
            "amount": 1,
            "name": "Functional unit",
            "cat": "Functional unit"
        })]
        for key, value in nodes.iteritems():
            if key == -1:
                continue
            code = lookup[key]
            db_data = Database(code[0]).load()
            value.update({
                "code": code,
                "name": db_data[code]["name"],
                "cat": db_data[code]["categories"][0],
            })
            new_nodes.append((key, value))
        return dict(new_nodes)

    def get_code(self, index, rev_mapping, rev_tech):
        return rev_mapping[rev_tech[index]]


def edge_cutter(nodes, edges, total, limit=0.0025):
    """The default graph traversal includes links which might be of small magnitude. This function cuts links that have small cumulative impact."""
    to_delete = []
    for i, e in enumerate(edges):
        if e["impact"] < (total * limit):
            to_delete.append(i)
        else:
            continue
            # print e
            # print nodes[e[1]]
            # print (e[2] / nodes[e[1]]["amount"] * nodes[e[1]]["cum"]) / (total * limit)
    return [e for i, e in enumerate(edges) if i not in to_delete]


def node_pruner(nodes, edges):
    """Remove nodes which have no links remaining after edge cutting."""
    good_nodes = set([e["from"] for e in edges]).union(
        set([e["to"] for e in edges]))
    return dict([(k, v) for k, v in nodes.iteritems() if k in good_nodes])


def extract_edges(arr, mapping, ignore):
    edges = []
    for i in range(arr.shape[0]):
        if mapping[i] in ignore:
            continue
        for j in range(arr.shape[1]):
            if mapping[j] in ignore or i == j or arr[i, j] == 0:
                continue
            edges.append((mapping[j], mapping[i], float(arr[i, j])))
    return edges


def rationalize_supply_chain(nodes, edges, total, limit=0.005):
    """
This class takes nodes and edges, and removes nodes to edges with low individual scores and reroutes the edges.
    """
    nodes_to_delete = [key for key, value in nodes.iteritems() if
                       value["ind"] < (total * limit) and key != -1]
    size = len(nodes)
    arr = np.zeros((size, size), dtype=np.float32)
    arr_map = dict([(key, index) for index, key in enumerate(sorted(nodes.keys()))])
    rev_map = dict([(v, k) for k, v in arr_map.iteritems()])
    for outp, inp, amount in edges:
        arr[arr_map[inp], arr_map[outp]] = amount
    for node in nodes_to_delete:
        index = arr_map[node]
        increment = (arr[:, index].reshape((-1, 1)) * arr[index, :].reshape((1, -1)))
        arr += increment
    new_edges = []
    new_edges = extract_edges(arr, rev_map, nodes_to_delete)
    new_nodes = dict([(k, v) for k, v in nodes.iteritems() if k not in nodes_to_delete])
    return new_nodes, new_edges


def d3_fd_graph_formatter(nodes, edges, total):
        # Sort node ids by highest cumulative score first
        node_ids = [x[1] for x in sorted(
            [(v["cum"], k) for k, v in nodes.iteritems()])]
        new_nodes = [nodes[i] for i in node_ids]
        lookup = dict([(key, index) for index, key in enumerate(node_ids)])
        new_edges = [{
            "source": lookup[e["to"]],
            "target": lookup[e["from"]],
            "amount": e["impact"]
        } for e in edges]
        return {"edges": new_edges, "nodes": new_nodes,
                "total": total}
