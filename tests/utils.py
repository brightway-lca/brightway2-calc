from bw2calc.utils import get_seed, wrap_functional_unit
import multiprocessing
import pytest
import sys


@pytest.mark.skipif(sys.version_info < (3,0), reason="MP pool changes")
def test_get_seeds_different_under_mp_pool():
    with multiprocessing.Pool(processes=4) as pool:
        results = list(pool.map(get_seed, [None] * 10))
    assert sorted(set(results)) == sorted(results)

def test_wrap_functional_unit():
    given = {17: 42}
    expected = {'key': 17, 'amount': 42}
    assert wrap_functional_unit(given) == [expected]

    given = {('a', 'b'): 42}
    expected = {'database': 'a', 'code': 'b', 'amount': 42}
    assert wrap_functional_unit(given) == [expected]

    class Foo:
        def __getitem__(self, index):
            if index == 0:
                return 'a'
            elif index == 1:
                return 'b'

    given = {Foo(): 42}
    expected = {'database': 'a', 'code': 'b', 'amount': 42}
    assert wrap_functional_unit(given) == [expected]
