from bw2calc.utils import get_seed
import multiprocessing
import pytest
import sys


@pytest.mark.skipif(sys.version_info < (3,0), reason="MP pool changes")
def test_get_seeds_different_under_mp_pool():
    with multiprocessing.Pool(processes=4) as pool:
        results = list(pool.map(get_seed, [None] * 10))
    assert sorted(set(results)) == sorted(results)
