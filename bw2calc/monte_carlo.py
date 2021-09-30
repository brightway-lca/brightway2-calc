from . import spsolve, prepare_lca_inputs
from .lca import LCA
from scipy.sparse.linalg import iterative
from stats_arrays.random import MCRandomNumberGenerator
import multiprocessing


class MonteCarloLCA(LCA):
    """Normal ``LCA`` class now supports Monte Carlo and iterative use. You normally want to use it instead."""
    def __init__(self, *args, **kwargs):
        if len(args) >= 9:
            args[9] = True
        else:
            kwargs['use_distributions'] = True
        super().__init__(*args, **kwargs)


class IterativeMonteCarlo(MonteCarloLCA):
    """Base class to use iterative techniques instead of `LU factorization <http://en.wikipedia.org/wiki/LU_decomposition>`_ in Monte Carlo."""

    def __init__(self, *args, iter_solver=iterative.cgs, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_solver = iter_solver
        self.guess = None

    def solve_linear_system(self):
        if not self.iter_solver or self.guess is None:
            self.guess = spsolve(self.technosphere_matrix, self.demand_array)
            if not self.guess.shape:
                self.guess = self.guess.reshape((1,))
            return self.guess
        else:
            solution, status = self.iter_solver(
                self.technosphere_matrix,
                self.demand_array,
                x0=self.guess,
                atol="legacy",
                maxiter=1000,
            )
            if status != 0:
                return spsolve(self.technosphere_matrix, self.demand_array)
            return solution


class ComparativeMonteCarlo(IterativeMonteCarlo):
    """First draft approach at comparative LCA"""

    def __init__(self, demands, *args, **kwargs):
        self.demands = demands
        # Get all possibilities for database retrieval
        demand_all = {key: 1 for d in demands for key in d}
        super().__init__(demand_all, *args, **kwargs)

    def load_data(self):
        if not getattr(self, "method"):
            raise ValueError("Must specify an LCIA method")

        self.load_lci_data()
        self.load_lcia_data()
        self.tech_rng = MCRandomNumberGenerator(self.tech_params, seed=self.seed)
        self.bio_rng = MCRandomNumberGenerator(self.bio_params, seed=self.seed)
        self.cf_rng = MCRandomNumberGenerator(self.cf_params, seed=self.seed)

    def __next__(self):
        if not hasattr(self, "tech_rng"):
            self.load_data()
        self.rebuild_technosphere_matrix(self.tech_rng.next())
        self.rebuild_biosphere_matrix(self.bio_rng.next())
        self.rebuild_characterization_matrix(self.cf_rng.next())

        if self.presamples:
            self.presamples.update_matrices()

        results = []
        for demand in self.demands:
            self.build_demand_array(demand)
            self.lci_calculation()
            self.lcia_calculation()
            results.append(self.score)
        return results


def single_worker(args):
    demand, data_objs, iterations = args
    mc = MonteCarloLCA(demand=demand, data_objs=data_objs)
    return [next(mc) for x in range(iterations)]


def direct_solving_worker(args):
    demand, data_objs, iterations = args
    mc = DirectSolvingMonteCarloLCA(demand=demand, data_objs=data_objs)
    return [next(mc) for x in range(iterations)]


class ParallelMonteCarlo:
    """Split a Monte Carlo calculation into parallel jobs"""

    def __init__(
        self,
        demand,
        method=None,
        data_objs=None,
        iterations=1000,
        chunk_size=None,
        cpus=None,
        log_config=None,
    ):
        if data_objs is None:
            if not prepare_lca_inputs:
                raise ImportError("bw2data version >= 4 not found")
            demand, data_objs, _ = prepare_lca_inputs(
                demand=demand, method=method, remapping=False
            )

        self.demand = demand
        self.packages = data_objs
        self.cpus = cpus or multiprocessing.cpu_count()
        if chunk_size:
            self.chunk_size = chunk_size
            self.num_jobs = iterations // chunk_size
            if iterations % self.chunk_size:
                self.num_jobs += 1
        else:
            self.num_jobs = self.cpus
            self.chunk_size = (iterations // self.num_jobs) + 1

    def calculate(self, worker=single_worker):
        with multiprocessing.Pool(processes=self.cpus) as pool:
            results = pool.map(
                worker,
                [
                    (self.demand, self.packages, self.chunk_size)
                    for _ in range(self.num_jobs)
                ],
            )
        return [x for lst in results for x in lst]


def multi_worker(args):
    """Calculate a single Monte Carlo iteration for many demands.

    ``args`` are in order:
        * ``project``: Name of project
        * ``demands``: List of demand dictionaries
        * ``method``: LCIA method

    Returns a list of results: ``[(demand dictionary, result)]``

    """
    demands, data_objs = args
    mc = MonteCarloLCA(demands[0], data_objs=data_objs)
    next(mc)
    results = []
    for demand in demands:
        mc.redo_lcia(demand)
        results.append((demand, mc.score))
    return results


class MultiMonteCarlo:
    """
This is a class for the efficient calculation of *many* demand vectors from
each Monte Carlo iteration.

Args:
    * ``args`` is a list of demand dictionaries
    * ``method`` is a LCIA method
    * ``iterations`` is the number of Monte Carlo iterations desired
    * ``cpus`` is the (optional) number of CPUs to use

The input list can have complex demands, so ``[{('foo', 'bar'): 1, ('foo', 'baz'): 1}, {('foo', 'another'): 1}]`` is OK.

Call ``.calculate()`` to generate results.

    """

    def __init__(self, demands, method=None, data_objs=None, iterations=100, cpus=None):
        # Convert from activity proxies if necessary
        if data_objs is None:
            if not prepare_lca_inputs:
                raise ImportError("bw2data version >= 4 not found")
            demands, data_objs, _ = prepare_lca_inputs(
                demands=demands, method=method, remapping=False
            )

        self.demands = demands
        self.packages = data_objs
        self.iterations = iterations
        self.cpus = cpus or multiprocessing.cpu_count()

    def merge_results(self, objs):
        """Merge the results from each ``multi_worker`` worker.

        ``[('a', [0,1]), ('a', [2,3])]`` becomes ``[('a', [0,1,2,3)]``.

        """
        r = {}
        for obj in objs:
            for key, value in obj:
                r.setdefault(frozenset(key.items()), []).append(value)
        return [(dict(x), y) for x, y in r.items()]

    def calculate(self, worker=multi_worker):
        """Calculate Monte Carlo results for many demand vectors.

        Returns a list of results with the format::

            [(demand dictionary, [lca scores])]

        There is no guarantee that the results are returned in the same order as the ``demand`` input variable.

        """
        with multiprocessing.Pool(processes=self.cpus) as pool:
            results = pool.map(
                worker, [(self.demands, self.packages) for _ in range(self.iterations)],
            )
        return self.merge_results(results)
