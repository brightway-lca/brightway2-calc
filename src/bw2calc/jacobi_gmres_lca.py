from typing import Optional

from scipy.sparse.linalg import gmres

from bw2calc.iterative_lca import IterativeLCA


class JacobiGMRESLCA(IterativeLCA):
    """Solve ``Ax=b`` with GMRES using a Jacobi preconditioner.

    The preconditioner is the inverse of the technosphere diagonal, i.e. ``D^-1``.
    Unlike :class:`bw2calc.iterative_lca.IterativeLCA`, the first solve is also iterative.

    .. warning::

        With the default ``direct_first_solve=False``, there is no reference
        factorization, so solutions are only checked by their residual, and failures go
        straight to the direct solver, without a fallback stage or time limit. For
        demands with strong waste or recycling loops, the technosphere matrix is
        ill-conditioned, and a small residual doesn't guarantee a correct solution: GMRES
        returned a solution 20% off with a relative residual of 1e-8
        (``market for waste polystyrene``, ecoinvent 3.12). Pass
        ``direct_first_solve=True`` to check solutions by their estimated forward error
        instead.

    :class:`bw2calc.iterative_lca.IterativeLCA`, which uses BiCGSTAB, was about 3x faster
    per solve in Monte Carlo benchmarks; see its docstring for the timings, the other
    parameters, the correctness checks, and the fallback stages.

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
        # "pr_norm" calls the callback, and so checks the time limit, on every inner
        # iteration, and avoids SciPy's warning about the default `callback_type`.
        return {"restart": self.restart, "callback_type": "pr_norm"}
