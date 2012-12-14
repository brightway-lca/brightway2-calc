# -*- coding: utf-8 -*
import itertools
from lca import LCA
from scipy.sparse.linalg import iterative, spsolve
from stats_toolkit.random import MCRandomNumberGenerator
import multiprocessing
import numpy as np


class MonteCarloLCA(LCA):
    def __init__(self, demand, method=None, iter_solver=iterative.cgs,
            seed=None, *args, **kwargs):
        super(MonteCarloLCA, self).__init__(demand, method=method, *args,
            **kwargs)
        self.seed = seed
        self.iter_solver = iter_solver
        self.guess = None
        self.load_databases()
        self.load_method()
        self.tech_rng = MCRandomNumberGenerator(self.tech_params, seed=seed)
        self.bio_rng = MCRandomNumberGenerator(self.bio_params, seed=seed)
        self.cf_rng = MCRandomNumberGenerator(self.cf_params, seed=seed)

    def next(self):
        self.build_technosphere_matrix(self.tech_rng.next())
        self.build_biosphere_matrix(self.bio_rng.next())
        self.build_characterization_matrix(self.cf_rng.next())

        if not hasattr(self, "demand_array"):
            self.build_demand_array()

        self.lci_calculation()
        self.lcia_calculation()

    def solve_linear_system(self):
        if not self.iter_solver or self.guess == None:
            self.guess = spsolve(
                self.technosphere_matrix.data,
                self.demand_array.data)
            return self.guess
        else:
            solution, status = self.iter_solver(
                self.technosphere_matrix.data,
                self.demand_array.data,
                x0=self.guess)
            if status != 0:
                raise
            return solution

    def iterate(self):
        # TODO: use yield or __iter__? What is most pythonic?
        self.next()
        return self.score


class ComparativeMonteCarlo(LCA):
    """First draft approach at comparative LCA"""
    def __init__(self, demands, method=None, iter_solver=iterative.cgs,
            seed=None, *args, **kwargs):
        self.demands = demands
        # Get all possibilities for database retrieval
        demand_all = demands[0].copy()
        for other in demands[1:]:
            demand_all.update(other)
        super(ComparativeMonteCarlo, self).__init__(demand_all, method)
        self.seed = seed
        self.iter_solver = iter_solver
        self.guess = None
        self.load_databases()
        self.load_method()
        self.tech_rng = MCRandomNumberGenerator(self.tech_params, seed=seed)
        self.bio_rng = MCRandomNumberGenerator(self.bio_params, seed=seed)
        self.cf_rng = MCRandomNumberGenerator(self.cf_params, seed=seed)

    def next(self):
        self.build_technosphere_matrix(self.tech_rng.next())
        self.build_biosphere_matrix(self.bio_rng.next())
        self.build_characterization_matrix(self.cf_rng.next())
        results = []
        for demand in self.demands:
            self.build_demand_array(demand)
            self.lci_calculation()
            self.lcia_calculation()
            results.append(self.score)
        return results

    def solve_linear_system(self):
        if not self.iter_solver or self.guess == None:
            self.guess = spsolve(
                self.technosphere_matrix.data,
                self.demand_array.data)
            return self.guess
        else:
            solution, status = self.iter_solver(
                self.technosphere_matrix.data,
                self.demand_array.data,
                x0=self.guess)
            if status != 0:
                raise
            return solution

    def iterate(self):
        raise NotImplemented


class ParallelMonteCarlo(object):
    """Split a Monte Carlo calculation into parallel jobs"""
    def __init__(self, demand, method, iterations=1000, chunk_size=None):
        self.demand = demand
        self.method = method
        if chunk_size:
            self.chunk_size = chunk_size
            self.num_jobs = iterations // chunk_size
            if iterations % self.chunk_size:
                self.num_jobs += 1
        else:
            self.num_jobs = max(multiprocessing.cpu_count() - 1, 1)
            self.chunk_size = (iterations // self.num_jobs) + 1

    def calculate(self):
        pool = multiprocessing.Pool(processes=max(multiprocessing.cpu_count() - 1, 1))
        results = [pool.apply_async(single_worker, (self.demand, self.method,
            self.chunk_size)) for x in xrange(self.num_jobs)]
        pool.close()
        pool.join()  # Blocks until calculation is finished
        return list(itertools.chain(*[x.get() for x in results]))


class MultiMonteCarlo(object):
    """
This is a class for the efficient calculation of multiple demand vectors from
each Monte Carlo iteration.
    """
    def __init__(self, demands, method, iterations):
        self.demands = demands
        self.method = method
        self.iterations = iterations

    def merge_dictionaries(self, *dicts):
        r = {}
        for dic in dicts:
            for k, v in dic.iteritems():
                r.setdefault(k, []).append(v)
        return r

    def calculate(self):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        results = [pool.apply_async(multi_worker, (self.demands, self.method)
            ) for x in xrange(self.iterations)]
        pool.close()
        pool.join()  # Blocks until calculation is finished
        return self.merge_dictionaries(*[x.get() for x in results])


def single_worker(demand, method, iterations):
    # demand, method, iterations = args
    mc = MonteCarloLCA(demand=demand, method=method)
    return [mc.iterate() for x in range(iterations)]


def multi_worker(demands, method):
    lca = LCA(demands[0], method)
    lca.load_databases()
    lca.load_method()
    # Create new matrices
    lca.build_technosphere_matrix(MCRandomNumberGenerator(lca.tech_params).next())
    lca.build_biosphere_matrix(MCRandomNumberGenerator(lca.bio_params).next())
    lca.build_characterization_matrix(MCRandomNumberGenerator(lca.cf_params).next())
    lca.decompose_technosphere()
    lca.lci()
    lca.lcia()
    results = {}
    for demand in demands:
        lca.redo_lcia(demand)
        results[str(demand)] = lca.score
    return results
