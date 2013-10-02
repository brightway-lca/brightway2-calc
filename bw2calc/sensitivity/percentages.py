# -*- coding: utf-8 -*
from __future__ import division
from stats_arrays.random import RandomNumberGenerator
from stats_arrays import uncertainty_choices
import numpy as np
import multiprocessing


def get_percentages(percentage, pa, seed=None,
        brute_force_size=10000,
        num_cpus=multiprocessing.cpu_count()):
    """Get PPF (percentage point function) values for a parameter array.

    Fallback to brute force if PPF function not defined for a uncertainty distribution."""
    assert 0 <= percentage <= 1
    num_params = pa.shape[0]
    if num_params < num_cpus:
        args = [(percentage, pa.copy(), seed, brute_force_size, 0)]
    else:
        step_size = int(num_params / num_cpus) + 1
        args = [(
            percentage,
            pa[i * step_size:(i + 1) * step_size].copy(),
            seed,
            brute_force_size,
            i) for i in xrange(num_cpus)]
    pool = multiprocessing.Pool(num_cpus)
    results = pool.map(_percentage_worker, args)
    results.sort()
    return np.hstack([x[1] for x in results])


def _percentage_worker(args):
    percentage, pa, seed, size, number = args
    vector = np.zeros(pa.shape[0])
    rng = RandomNumberGenerator(
        uncertainty_choices[pa[0]['uncertainty_type']],
        pa[0].reshape((-1,)),
        seed=seed,
        size=size
    )
    for index, row in enumerate(pa):
        distribution = uncertainty_choices[row['uncertainty_type']]
        try:
            value = distribution.ppf(
                row.reshape((1,)),
                np.array(percentage, dtype=np.float32).reshape((1,1))
            )
        except:
            data = rng.generate_random_numbers(
                uncertainty_type=distribution,
                params=row.reshape((1,))
            )
            data.sort()
            value = data[0, int((size - 1.) * percentage)]
        vector[index] = value
    return (number, vector)
