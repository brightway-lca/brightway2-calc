# -*- coding: utf-8 -*
from __future__ import division
from ..mc_vector import ParameterVectorLCA
from stats_arrays.random import MCRandomNumberGenerator
import multiprocessing
import numpy as np


def _saltelli_worker(args):
    kwargs, matrix, label = args
    ss = SobolSensitivity()
    model = ParameterVectorLCA(**kwargs)
    return (label, ss.evaluate_matrix(matrix, model))


def mp_calculate_sensitivity(params, kwargs=None, mask=None, N=250,
                             cpus=None, seed=None):
    """Compute sensitivity indices using *multiprocesing* and all available CPU cores.

    Inputs:
        * *params*: The

    Returns:
        * mapping of result indices to parameter indices
        * first-order sensitivity indices
        * global sensitivity indices

    """
    if kwargs is None:
        kwargs = {}
    ss = SobolSensitivity(seed)
    pool = multiprocessing.Pool(
        processes=cpus or multiprocessing.cpu_count()
    )
    matrix_a = ss.generate_matrix(params, N)
    matrix_b = ss.generate_matrix(params, N)
    jobs = [
        pool.apply_async(
            _saltelli_worker,
            (kwargs, matrix_a.copy(), "a")
        ),
        pool.apply_async(
            _saltelli_worker,
            (kwargs, matrix_b.copy(), "b")
        )
    ]
    for index in xrange(params.shape[0]):
        if mask is not None and not mask[index]:
            continue
        jobs.append(pool.apply_async(
            _saltelli_worker,
            (kwargs, ss.generate_c(matrix_a, matrix_b, index), index)
        ))
    pool.close()
    pool.join()
    values = [x.get() for x in jobs]
    values.sort()
    results = np.empty((len(values) - 2, N))
    mapping = {}
    reference_a = filter(lambda x: x[0] == "a", values)
    reference_b = filter(lambda x: x[0] == "b", values)
    for index in xrange(len(values) - 2):
        assert isinstance(values[index][0], int)
        mapping[index] = values[index][0]
        results[index, :] = values[index][1]
    return (
        mapping,
        ss.compute_fo_sensitivity_indices(reference_a, results),
        ss.compute_total_sensitivity_indices(reference_a, reference_b, results)
    )


class SobolSensitivity(object):
    """Compute variance-based sensitivity indices using Saltelli's variation of Sobol's method.

    See more infomation in the following sources:

    Takes an optional *seed* argument for the random number generation.

    """
    def __init__(self, seed=None):
        self.seed = seed

    def generate_matrix(self, params, N):
        """Evaluate the ``model`` for each row of ``matrix``.

        TODO: Consider using space-filling curves or other  sequences (e.g. `Low discrepancy sequences <http://en.wikipedia.org/wiki/Constructions_of_low-discrepancy_sequences>`_ or `Sobol sequences <http://en.wikipedia.org/wiki/Sobol_sequence>`_) which can improve random sampling.

        Inputs:
            * *matrix*: The array of input values, shape (*k*, *N*).
            * *N*: The integer number of Monte Carlo samples.

        Returns a one-dimensional array of model results.

        """
        rng = MCRandomNumberGenerator(params, seed=self.seed)
        return np.hstack([rng.next().reshape((-1, 1)) for x in xrange(N)])

    def generate_c(self, array_a, array_b, index):
        """Generate the **C** matrix, which is mostly **B** but one column of **A**.

        Inputs:
            * *array_a*: The **A** matrix.
            * *array_b*: The **B** matrix.
            * *index*: The integer column index to replace.

        Returns the **C** matrix.

        """
        array_c = array_b.copy()
        array_c[:, index] = array_a[:, index]
        return array_c

    def evaluate_matrix(self, matrix, model):
        """Evaluate the ``model`` for each row of ``matrix``.

        Inputs:
            * *matrix*: The array of input values, shape (*k*, *N*).

        Returns a one-dimensional array of model results.

        """
        return np.array([model(matrix[:, x]) for x in xrange(matrix.shape[1])])

    def compute_fo_sensitivity_indices(self, reference_a, perturbations):
        """Compute first order sensitivity indices.

        Inputs:

        ``reference_a`` is a one-dimensional array of LCA results generated from the matrix of reference values.

        ``perturbations`` is a two-dimensional array of LCA results for the perturbed parameters.

        ``reference_a`` has size *N*, where *N* is the number of iterations. ``perturbations`` has size (*k*, *N*), where *k* is the number of perturbed parameters.

        Returns:
            An array of sensitivity indices of shape *k*.
        """
        numerator = (perturbations * reference_a).sum(axis=0) -  np.average(reference_a) ** 2
        denominator = (reference_a ** 2).sum() - np.average(reference_a) ** 2
        return numerator / denominator

    def compute_total_sensitivity_indices(self, reference_a, reference_b, perturbations):
        """Compute total sensitivity indices.

        Inputs:

        ``reference_a`` is a one-dimensional array of LCA results generated from the matrix of reference values.

        ``reference_b`` is a one-dimensional array of LCA results generated from the matrix of reference values, where one column was replaced with the perturbed values.

        ``perturbations`` is a two-dimensional array of LCA results for the perturbed parameters.

        ``reference_a`` has size *N*, ``reference_b`` has size *N*, where *N* is the number of iterations. ``perturbations`` has size (*k*, *N*), where *k* is the number of perturbed parameters.

        Returns:
            An array of sensitivity indices of shape *k*.
        """
        numerator = (perturbations * reference_b).sum(axis=0) -  np.average(reference_a) ** 2
        denominator = (reference_a ** 2).sum() - np.average(reference_a) ** 2
        return 1 - numerator / denominator

    def calculate_sensitivity(self, params, model, N=250):
        """Calculate first-order and total sensitivity indices for a set of parameters and a model.

        Normally using the ``mp_calculate_sensitivity`` function`` is more efficient, as it distributes work amoung all available cores.

        Inputs:
            * *params*: a ``stats_arrays`` set of stochastic parameters.
            * *model*: a model object, i.e. something that supports the syntax ``model(input_vector)`` syntax.
            * *N*: Number of Monte Carlo iterations to calculate for each parameter. Should be several hundred to several thousand, depending on the linearity of the model.

        Returns two one-dimensional arrays, first-order and total sensitivity indices, each of length (number of parameters).

        """
        matrix_a = self.generate_matrix(params, N)
        matrix_b = self.generate_matrix(params, N)
        reference_a = self.evaluate_matrix(matrix_a, model)
        reference_b = self.evaluate_matrix(matrix_b, model)
        k = params.shape[0]
        results = np.empty((k, N))
        for index in xrange(k):
            matrix_c = self.generate_c(matrix_a, matrix_b, index)
            results[index, :] = self.evaluate_matrix(matrix_c, model)
        return (
            self.compute_fo_sensitivity_indices(reference_a, results),
            self.compute_total_sensitivity_indices(
                reference_a,
                reference_b,
                results
            )
        )


