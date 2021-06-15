from .errors import EfficiencyWarning, NoSolutionFound
from .lca import LCA
from scipy.sparse.linalg import lsmr
import warnings


class LeastSquaresLCA(LCA):
    """Solve overdetermined technosphere matrix with more products than activities using least-squares approximation.

    See also:

    * `Multioutput processes in LCA <http://chris.mutel.org/multioutput.html>`_
    * `LSMR in SciPy <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html#scipy.sparse.linalg.lsmr>`_
    * `Another least-squares algorithm in SciPy <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr>`_

    """

    def solve_linear_system(self, solver=lsmr):
        if self.technosphere_matrix.shape[0] == self.technosphere_matrix.shape[1]:
            warnings.warn(
                "Don't use LeastSquaresLCA for square matrices", EfficiencyWarning
            )
        self.solver_results = solver(self.technosphere_matrix, self.demand_array)
        if self.solver_results[1] not in {1, 2}:
            warnings.warn(
                "No suitable solution found - supply array is probably nonsense",
                NoSolutionFound,
            )
        return self.solver_results[0]

    def decompose_technosphere(self):
        raise NotImplementedError("Can't decompose rectangular technosphere")
