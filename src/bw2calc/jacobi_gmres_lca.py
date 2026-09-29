from typing import Optional

from scipy.sparse.linalg import gmres

from bw2calc.iterative_lca import IterativeLCA


class JacobiGMRESLCA(IterativeLCA):
    """Solve ``Ax=b`` with GMRES using a Jacobi preconditioner.

    The preconditioner is the inverse of the technosphere diagonal, i.e. ``D^-1``.
    Unlike :class:`bw2calc.iterative_lca.IterativeLCA`, the first solve is also iterative.

    :class:`bw2calc.iterative_lca.IterativeLCA`, which uses BiCGSTAB, was about 3x faster
    per solve in Monte Carlo benchmarks; see its docstring for the timings, the other
    parameters, the correctness check, and the fallback to the direct solver.

    :param restart: Number of iterations between GMRES restarts. ``None`` uses SciPy defaults.
    :type restart: int or None
    """

    def __init__(
        self,
        *args,
        restart: Optional[int] = 50,
        maxiter: Optional[int] = 300,
        direct_first_solve: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("iter_solver", gmres)
        super().__init__(
            *args, maxiter=maxiter, direct_first_solve=direct_first_solve, **kwargs
        )
        self.restart = restart

    def solver_kwargs(self) -> dict:
        return {"restart": self.restart}
