# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division
from eight import *
from future.utils import implements_iterator

from .lca import LCA
from .utils import clean_databases
from bw2data import projects
from scipy.sparse.linalg import iterative, spsolve
from stats_arrays.random import MCRandomNumberGenerator
import itertools
import multiprocessing


@implements_iterator
class IterativeMonteCarlo(LCA):
    """Base class to use iterative techniques instead of `LU factorization <http://en.wikipedia.org/wiki/LU_decomposition>`_ in Monte Carlo."""
    def __init__(self, demand, method=None, iter_solver=iterative.cgs,
                 seed=None, *args, **kwargs):
        super(IterativeMonteCarlo, self).__init__(demand, method=method, *args,
                                                  **kwargs)
        self.seed = seed
        self.iter_solver = iter_solver
        self.guess = None
        self.lcia = method is not None

    def __iter__(self):
        return self

    def __call__(self):
        return self.next()

    def __next__(self):
        raise NotImplemented

    def solve_linear_system(self):
        if not self.iter_solver or self.guess is None:
            self.guess = spsolve(
                self.technosphere_matrix,
                self.demand_array)
            return self.guess
        else:
            solution, status = self.iter_solver(
                self.technosphere_matrix,
                self.demand_array,
                x0=self.guess,
                maxiter=1000)
            if status != 0:
                return spsolve(
                    self.technosphere_matrix,
                    self.demand_array
                )
            return solution


class MonteCarloLCA(IterativeMonteCarlo):
    """Monte Carlo uncertainty analysis with separate `random number generators <http://en.wikipedia.org/wiki/Random_number_generation>`_ (RNGs) for each set of parameters."""
    def load_data(self):
        self.load_lci_data()
        self.tech_rng = MCRandomNumberGenerator(self.tech_params, seed=self.seed)
        self.bio_rng = MCRandomNumberGenerator(self.bio_params, seed=self.seed)
        if self.lcia:
            self.load_lcia_data()
            self.cf_rng = MCRandomNumberGenerator(self.cf_params, seed=self.seed)
        if self.weighting:
            self.load_weighting_data()
            self.weighting_rng = MCRandomNumberGenerator(self.weighting_params, seed=self.seed)

    def __next__(self):
        if not hasattr(self, "tech_rng"):
            self.load_data()
        self.rebuild_technosphere_matrix(self.tech_rng.next())
        self.rebuild_biosphere_matrix(self.bio_rng.next())
        if self.lcia:
            self.rebuild_characterization_matrix(self.cf_rng.next())
        if self.weighting:
            self.weighting_value = self.weighting_rng.next()

        if not hasattr(self, "demand_array"):
            self.build_demand_array()

        self.lci_calculation()
        if self.lcia:
            self.lcia_calculation()
            if self.weighting:
                self.weighting_calculation()
            return self.score
        else:
            return self.supply_array


class ComparativeMonteCarlo(IterativeMonteCarlo):
    """First draft approach at comparative LCA"""
    def __init__(self, demands, *args, **kwargs):
        self.demands = demands
        # Get all possibilities for database retrieval
        demand_all = demands[0].copy()
        for other in demands[1:]:
            demand_all.update(other)
        super(ComparativeMonteCarlo, self).__init__(demand_all, *args, **kwargs)

    def load_data(self):
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

        results = []
        for demand in self.demands:
            self.build_demand_array(demand)
            self.lci_calculation()
            self.lcia_calculation()
            results.append(self.score)
        return results


def single_worker(project, demand, method, iterations):
    projects.set_current(project, writable=False)
    mc = MonteCarloLCA(demand=demand, method=method)
    return [mc.next() for x in range(iterations)]


class ParallelMonteCarlo(object):
    """Split a Monte Carlo calculation into parallel jobs"""
    def __init__(self, demand, method, iterations=1000, chunk_size=None,
                 cpus=None):
        clean_databases()
        self.demand = demand
        self.method = method
        self.cpus = cpus
        if chunk_size:
            self.chunk_size = chunk_size
            self.num_jobs = iterations // chunk_size
            if iterations % self.chunk_size:
                self.num_jobs += 1
        else:
            self.num_jobs = self.cpus or multiprocessing.cpu_count()
            self.chunk_size = (iterations // self.num_jobs) + 1

    def calculate(self, worker=single_worker):
        pool = multiprocessing.Pool(
            processes=min(self.num_jobs, multiprocessing.cpu_count())
        )
        results = [pool.apply_async(
            worker,
            (projects.current, self.demand, self.method, self.chunk_size)
        ) for x in range(self.num_jobs)]
        pool.close()
        pool.join()  # Blocks until calculation is finished
        results_list = list(itertools.chain(*[x.get() for x in results]))
        # Have to terminate pool or get "OSError: Too many open files"
        pool.terminate()
        return results_list


class MultiMonteCarlo(object):
    """
This is a class for the efficient calculation of multiple demand vectors from
each Monte Carlo iteration.
    """
    def __init__(self, demands, method, iterations):
        clean_databases()
        self.demands = demands
        self.method = method
        self.iterations = iterations

    def merge_dictionaries(self, *dicts):
        r = {}
        for dic in dicts:
            for k, v in dic.items():
                r.setdefault(k, []).append(v)
        return r

    def calculate(self):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        results = [pool.apply_async(
            multi_worker,
            (projects.current, self.demands, self.method)
        ) for x in range(self.iterations)]
        pool.close()
        pool.join()  # Blocks until calculation is finished
        results_dict = self.merge_dictionaries(*[x.get() for x in results])
        # Have to terminate pool or get "OSError: Too many open files"
        pool.terminate()
        return results_dict


def multi_worker(project, demands, method):
    projects.set_current(project, writable=False)
    lca = LCA(demands[0], method)
    lca.load_lci_data()
    lca.load_lcia_data()
    # Create new matrices
    lca.rebuild_technosphere_matrix(
        MCRandomNumberGenerator(lca.tech_params).next())
    lca.rebuild_biosphere_matrix(
        MCRandomNumberGenerator(lca.bio_params).next())
    lca.rebuild_characterization_matrix(
        MCRandomNumberGenerator(lca.cf_params).next())
    lca.decompose_technosphere()
    lca.lci()
    lca.lcia()
    results = {}
    for demand in demands:
        lca.redo_lcia(demand)
        results[str(demand)] = lca.score
    return results
