import warnings
from heapq import heappop, heappush
import itertools
from functools import lru_cache

import numpy as np
from scipy import sparse

from . import spsolve, LCA


class CachingSolver:
    def __init__(self, lca):
        self.nrows = len(lca.demand_array)
        self.technosphere_matrix = lca.technosphere_matrix

    @lru_cache(maxsize=None)
    def __call__(self, product_index, amount=1):
        demand = np.zeros(self.nrows)
        demand[product_index] = amount
        return spsolve(self.technosphere_matrix, demand)


class AssumedDiagonalGraphTraversal:
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
        demand[index] = (
            supply[index]
            *
            # Normalize by the production amount
            lca.technosphere_matrix[index, index]
        )
        return float(
            (characterized_biosphere * spsolve(lca.technosphere_matrix, demand)).sum()
        )

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


class GraphTraversal:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Please use `AssumedDiagonalGraphTraversal` instead of `GraphTraversal`",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class MultifunctionalGraphTraversal:
    """
    Traverse a supply chain, following paths of greatest impact. Can handle the differentiation between products and activities, and makes no assumptions about multifunctionality, substitution, or the special status of numbers on the diagonal.

    As soon as non-diagonal values are allowed, we lose any concept of a reference product. This means that we can trace the edges for an activity (both inputs and outputs, though in the matrix there is no functional difference), but we can't for a product, as we can't use the graph structure to determine *which activity* produced the product. There could be more than one, or even zero, depending on how your mental model of substitution works. Our algorithm is therefore:

    1. Start with products (initially the products in the functional unit)
    2. For each product, determine which activities produced it by solving the linear system
    3a. For each of these activities, add on to our list of products to consider by looking at the edges for that activity, and excluding the edge which led to our original product
    3b. If we have already examined this activity, don't visit it again
    4. Keep iterating over the list of products until we run out of activities or hit our calculation limit

    The ``.calculate()`` function therefore returns the following:

    .. code-block:: python

        {
            'counter': int, # Number of LCA calculations done,
            'products': {
                id: {  # id is either the database integer id (if `translate_indices` is True) or the matrix row index
                    'amount': float # Total amount of this product produced to satisfy the functional unit
                    'supply_chain_score': float # The total impact of producing this product
                }
            },
            'activities': {
                id: {  # id is either the database integer id (if `translate_indices` is True) or the matrix column index
                    'amount': float # Total amount of this activity produced to satisfy the entire functional unit
                    'direct_score': float # The impact of the direct emissions associated to this activity and its amount
            },
            'edges': [{
                'target': int,  # product id if type is activity else activity id
                'source': int,  # activity id if type is product else product id
                'type': str,  # 'product' or 'activity'
                'amount': float,  # Total amount of the flow
                'exc_amount': float,  # Value given in the technosphere matrix
                'supply_chain_score': float,  # Total impact from the production of this product. Only for type 'product'
                'direct_score': float,  # Impact from direct emissions of this activity. Only for type 'activity'
            }]
        }

    As in AssumedDiagonalGraphTraversal, we use a priority queue to examine products in order of their total impact.

    This class is written in a functional style, with only class methods.

    """

    @classmethod
    def calculate(cls, lca: LCA, cutoff: float = 0.005, max_calc: int = 1e5, translate_indices: bool = True):
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

        score = lca.score

        if score == 0:
            raise ValueError("Zero total LCA score makes traversal impossible")

        solver = CachingSolver(lca)

        heap, activities, products, edges, counter = cls.initialize_heap(lca, solver, translate_indices, 0)
        activities, products, edges, counter = cls.traverse(
            heap=heap,
            solver=solver,
            activities=activities,
            products=products,
            edges=edges,
            max_calc=max_calc,
            cutoff=cutoff,
            total_score=score,
            lca=lca,
            translate_indices=translate_indices,
            counter=counter,
        )
        edges = cls.consolidate_edges(edges)

        return {
            "products": cls.clean_small_values(products),
            "activities": cls.clean_small_values(activities),
            "edges": cls.clean_small_values(edges, kind=list),
            "counter": counter,
        }

    @classmethod
    def clean_small_values(cls, data, kind=dict, cutoff=5e-16):
        if kind == list:
            return [obj for obj in data if abs(obj['amount']) >= cutoff]
        else:
            return {k: v for k, v in data.items() if abs(v['amount']) > cutoff}

    @classmethod
    def consolidate_edges(cls, edges):
        def consolidate_edges(key, group):
            label = 'supply_chain_score' if key[2] == 'product' else 'direct_score'
            group = list(group)
            return {
                'source': key[0],
                'target': key[1],
                'type': key[2],
                'amount': sum([obj['amount'] for obj in group]),
                'exc_amount': group[0]["exc_amount"],
                label: sum([obj[label] for obj in group])
            }

        edges.sort(key=lambda x: (x['source'], x['target'], x['type']))
        return [consolidate_edges(key, group) for key, group in itertools.groupby(edges, lambda x: (x['source'], x['target'], x['type']))]


    @classmethod
    def initialize_heap(cls, lca: LCA, solver: CachingSolver, translate_indices: bool, counter: int):
        """
        Create a `priority queue <http://docs.python.org/2/library/heapq.html>`_ or ``heap`` to store inventory datasets, sorted by LCA score.

        Populates the heap with each activity in ``demand``. Initial nodes are the *functional unit*, i.e. the complete demand, and each activity in the *functional unit*. Initial edges are inputs from each activity into the *functional unit*.

        The *functional unit* is an abstract dataset (as it doesn't exist in the matrix), and is assigned the index ``-1``.

        """
        heap, edges = [], []
        products = {}
        activities = {
            -1: {
                "amount": 1,
                "direct_score": 0,
            }
        }
        for product_index, amount in enumerate(lca.demand_array):
            if amount == 0:
                continue
            counter += 1
            cumulative_score = (
                lca.characterization_matrix
                * lca.biosphere_matrix
                * solver(product_index)
            ).sum() * amount
            heappush(heap, (abs(1 / cumulative_score), product_index, amount))
            products[lca.dicts.product.reversed[product_index] if translate_indices else product_index] = {
                "amount": amount,
                "supply_chain_score": cumulative_score,
            }
            edges.append(
                {
                    "target": lca.dicts.product.reversed[product_index] if translate_indices else product_index,
                    "source": -1,
                    "type": "product",
                    "amount": amount,
                    "exc_amount": amount,
                    "supply_chain_score": cumulative_score,
                }
            )
        return heap, activities, products, edges, counter

    @classmethod
    def traverse(
        cls,
        heap: list,
        solver: CachingSolver,
        activities: dict,
        products: dict,
        edges: list,
        max_calc: int,
        cutoff: float,
        total_score: float,
        lca: LCA,
        translate_indices: bool,
        counter: int,
    ):
        """
        Build a directed graph by traversing the supply chain.

        Node ids are actually technosphere row/col indices, which makes lookup easier.

        Returns:
            (nodes, edges, number of calculations)

        """
        cutoff_score = abs(cutoff * total_score)

        characterized_biosphere = lca.characterization_matrix * lca.biosphere_matrix

        while heap:
            if counter >= max_calc:
                warnings.warn("Stopping traversal due to calculation count.")
                break
            _, product_index, product_amount = heappop(heap)

            # Need to find all actual activities which produce this product.
            supply = solver(product_index) * product_amount
            supply[
                ~lca.technosphere_matrix[product_index, :]
                .toarray()
                .astype(bool)
                .ravel()
            ] = 0

            for producing_activity_index in supply.nonzero()[0]:
                producing_activity_index = int(producing_activity_index)

                maybe_mapped_activity_index = lca.dicts.activity.reversed[producing_activity_index] if translate_indices else producing_activity_index

                edges.append(
                    {
                        "target": maybe_mapped_activity_index,
                        "source": lca.dicts.product.reversed[product_index] if translate_indices else product_index,
                        "type": "activity",
                        "amount": product_amount,
                        "exc_amount": lca.technosphere_matrix[
                            product_index, producing_activity_index
                        ],
                        # Direct score attributable to this edge
                        "direct_score": characterized_biosphere[
                            :, producing_activity_index
                        ].sum()
                        * supply[producing_activity_index],
                    }
                )
                # Want multiple edges, but not multiple activity nodes
                if maybe_mapped_activity_index not in activities:
                    counter = cls.visit_activity(
                        heap=heap,
                        activity_index=producing_activity_index,
                        counter=counter,
                        activities=activities,
                        products=products,
                        edges=edges,
                        lca=lca,
                        characterized_biosphere=characterized_biosphere,
                        solver=solver,
                        cutoff_score=cutoff_score,
                        origin_product_index=product_index,
                        translate_indices=translate_indices,
                    )

        return activities, products, edges, counter

    @classmethod
    def visit_activity(
        cls,
        heap: list,
        activity_index: int,
        counter: int,
        activities: dict,
        products: dict,
        edges: list,
        lca: LCA,
        characterized_biosphere: sparse.csr_matrix,
        solver: CachingSolver,
        cutoff_score: float,
        origin_product_index: int,
        translate_indices: bool,
    ):
        activities[lca.dicts.activity.reversed[activity_index] if translate_indices else activity_index] = {
            "amount": lca.supply_array[activity_index],
            # Total direct score over all edges
            "direct_score": characterized_biosphere[:, activity_index].sum()
            * lca.supply_array[activity_index],
        }

        tm_coo = lca.technosphere_matrix[:, activity_index].tocoo()

        # We will get the activity amounts based on their total over the functional unit. We only visit each activity once.
        scale = -1 * lca.supply_array[activity_index]

        for product_index, product_amount in zip(tm_coo.row, tm_coo.data):
            if product_index == origin_product_index:
                continue

            # Amount in technosphere matrix has sign flipped, and is in relation to the activity production amount.
            # Normalize to a positive number with production amount of 1
            product_amount *= scale

            counter += 1
            supply = solver(product_index) * product_amount
            cumulative_score = (characterized_biosphere * supply).sum()

            if abs(cumulative_score) < cutoff_score:
                continue

            try:
                products[lca.dicts.product.reversed[product_index] if translate_indices else product_index]["amount"] += product_amount
                products[lca.dicts.product.reversed[product_index] if translate_indices else product_index]["supply_chain_score"] += cumulative_score
            except KeyError:
                products[lca.dicts.product.reversed[product_index] if translate_indices else product_index] = {
                    "amount": product_amount,
                    "supply_chain_score": cumulative_score,
                }
            edges.append(
                {
                    "target": lca.dicts.product.reversed[product_index] if translate_indices else product_index,
                    "source": lca.dicts.activity.reversed[activity_index] if translate_indices else activity_index,
                    "type": "product",
                    "amount": product_amount,
                    "exc_amount": lca.technosphere_matrix[
                        product_index, activity_index
                    ],
                    "supply_chain_score": cumulative_score,
                }
            )
            heappush(heap, (abs(1 / cumulative_score), product_index, product_amount))

        return counter
