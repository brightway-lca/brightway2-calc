from lca import LCA
from scipy.sparse.linalg import iterative, spsolve
from stats_toolkit.random import MCRandomNumberGenerator


class MonteCarloLCA(LCA):
    def __init__(self, activity, method=None, iter_solver=iterative.cgs,
            seed=None, *args, **kwargs):
        super(MonteCarloLCA, self).__init__(activity, method=method, *args,
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

    def __iter__(self):
        self.next()
        return self.score
