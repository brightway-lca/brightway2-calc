from . import spsolve
from heapq import heappush, heappop
import numpy as np
import warnings


class GraphTraversal:
    """
Traverse a supply chain, following paths of greatest impact.

This implementation uses a queue of datasets to assess. As the supply chain is traversed, datasets inputs are added to a list sorted by LCA score. Each activity in the sorted list is assessed, and added to the supply chain graph, as long as its impact is above a certain threshold, and the maximum number of calculations has not been exceeded.

Because the next dataset assessed is chosen by its impact, not its position in the graph, this is neither a breadth-first nor a depth-first search, but rather "importance-first".

This class is written in a functional style - no variables are stored in *self*, only methods.

Should be used by calling the ``calculate`` method.

.. warning:: Graph traversal with multioutput processes only works when other inputs are substituted (see `Multioutput processes in LCA <http://chris.mutel.org/multioutput.html>`__ for a description of multiputput process math in LCA).

    """

    def calculate(self, lca, cutoff=0.005, max_calc=1e5, skip_coproducts=False):
        """
Traverse the supply chain graph.

Args:
    * *lca* (dict): An instance of ``bw2calc.lca.LCA``.
    * *cutoff* (float, default=0.005): Cutoff criteria to stop LCA calculations. Relative score of total, i.e. 0.005 will cutoff if a dataset has a score less than 0.5 percent of the total.
    * *max_calc* (int, default=10000): Maximum number of LCA calculations to perform.

Returns:
    Dictionary of nodes, edges, and number of LCA calculations.

        """
        if not hasattr(lca, "supply_array"):
            lca.lci()
        if not hasattr(lca, "characterized_inventory"):
            lca.lcia()

        supply = lca.supply_array.copy()
        score = lca.score

        if score == 0:
            raise ValueError("Zero total LCA score makes traversal impossible")

        # Create matrix of LCIA CFs times biosphere flows, as these don't
        # change. This is also the unit score of each activity.
        characterized_biosphere = np.array(
            (lca.characterization_matrix * lca.biosphere_matrix).sum(axis=0)
        ).ravel()

        heap, nodes, edges = self.initialize_heap(lca, supply, characterized_biosphere)
        nodes, edges, counter = self.traverse(
            heap,
            nodes,
            edges,
            0,
            max_calc,
            cutoff,
            score,
            supply,
            characterized_biosphere,
            lca,
            skip_coproducts,
        )

        return {
            "nodes": nodes,
            "edges": edges,
            "counter": counter,
        }

    def initialize_heap(self, lca, supply, characterized_biosphere):
        """
Create a `priority queue <http://docs.python.org/2/library/heapq.html>`_ or ``heap`` to store inventory datasets, sorted by LCA score.

Populates the heap with each activity in ``demand``. Initial nodes are the *functional unit*, i.e. the complete demand, and each activity in the *functional unit*. Initial edges are inputs from each activity into the *functional unit*.

The *functional unit* is an abstract dataset (as it doesn't exist in the matrix), and is assigned the index ``-1``.

        """
        heap, edges = [], []
        nodes = {-1: {"amount": 1, "cum": lca.score, "ind": 1e-6 * lca.score}}
        for index, amount in enumerate(lca.demand_array):
            if amount == 0:
                continue
            cum_score = self.cumulative_score(
                index, supply, characterized_biosphere, lca
            )
            heappush(heap, (abs(1 / cum_score), index))
            nodes[index] = {
                "amount": float(supply[index]),
                "cum": cum_score,
                "ind": self.unit_score(index, supply, characterized_biosphere),
            }
            edges.append(
                {
                    "to": -1,
                    "from": index,
                    "amount": amount,
                    "exc_amount": amount,
                    "impact": cum_score * amount / float(supply[index]),
                }
            )
        return heap, nodes, edges

    def cumulative_score(self, index, supply, characterized_biosphere, lca):
        """Compute cumulative LCA score for a given activity"""
        demand = np.zeros((supply.shape[0],))
        demand[index] = (supply[index] *
                         # Normalize by the production amount
                         lca.technosphere_matrix[index, index])
        return float((characterized_biosphere * spsolve(lca.technosphere_matrix, demand)).sum())

    def unit_score(self, index, supply, characterized_biosphere):
        """Compute the LCA impact caused by the direct emissions and resource consumption of a given activity"""
        return float(characterized_biosphere[index] * supply[index])

    def traverse(
        self,
        heap,
        nodes,
        edges,
        counter,
        max_calc,
        cutoff,
        total_score,
        supply,
        characterized_biosphere,
        lca,
        skip_coproducts,
    ):
        """
Build a directed graph by traversing the supply chain.

Node ids are actually technosphere row/col indices, which makes lookup easier.

Returns:
    (nodes, edges, number of calculations)

        """
        # static_databases = {name for name in databases if databases[name].get("static")}
        # reverse = lca.dicts.activity.reversed

        while heap:
            if counter >= max_calc:
                warnings.warn("Stopping traversal due to calculation count.")
                break
            parent_index = heappop(heap)[1]
            # Skip links from static databases
            # if static_databases and reverse[parent_index][0] in static_databases:
            #     continue

            # Assume that this activity produces its reference product
            scale_value = lca.technosphere_matrix[parent_index, parent_index]
            if scale_value == 0:
                raise ValueError(
                    "Can't rescale activities that produce zero reference product"
                )
            col = lca.technosphere_matrix[:, parent_index].tocoo()
            # Multiply by -1 because technosphere values are negative
            # (consumption of inputs) and rescale
            children = [
                (int(col.row[i]), float(-1 * col.data[i] / scale_value))
                for i in range(col.row.shape[0])
            ]
            for activity, amount in children:
                # Skip values on technosphere diagonal
                if activity == parent_index:
                    continue
                # Skip negative coproducts
                if skip_coproducts and amount <= 0:
                    continue
                counter += 1
                cumulative_score = self.cumulative_score(
                    activity, supply, characterized_biosphere, lca
                )
                if abs(cumulative_score) < abs(total_score * cutoff):
                    continue

                # flow between activity and parent (Multiply by -1 because technosphere values are negative)
                flow = (
                    -1.0
                    * lca.technosphere_matrix[activity, parent_index]
                    * supply[parent_index]
                )
                total_activity_output = (
                    lca.technosphere_matrix[activity, activity] * supply[activity]
                )

                # Edge format is (to, from, mass amount, cumulative impact)
                edges.append(
                    {
                        "to": parent_index,
                        "from": activity,
                        # Amount of this link * amount of parent demanding link
                        "amount": flow,
                        # Raw exchange value
                        "exc_amount": amount,
                        # Impact related to this flow
                        "impact": flow / total_activity_output * cumulative_score,
                    }
                )
                # Want multiple incoming edges, but don't add existing node
                if activity in nodes:
                    continue
                nodes[activity] = {
                    # Total amount of this flow supplied
                    "amount": total_activity_output,
                    # Cumulative score from all flows of this activity
                    "cum": cumulative_score,
                    # Individual score attributable to environmental flows
                    # coming directory from or to this activity
                    "ind": self.unit_score(activity, supply, characterized_biosphere),
                }
                heappush(heap, (abs(1 / cumulative_score), activity))

        return nodes, edges, counter
