from bw2calc.lca import LCA
from pathlib import Path

fixture_dir = Path(__file__).resolve().parent / "fixtures"


def test_X_dict():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    assert lca.product_dict == {1: 0, 2: 1}
    assert lca.activity_dict == {101: 0, 102: 1}
    assert lca.biosphere_dict == {1: 0}


def test_reverse_dict():
    packages = [fixture_dir / "basic_fixture.zip"]
    lca = LCA({1: 1}, data_objs=packages)
    lca.lci()
    ra, rp, rb = lca.reverse_dict()
    assert ra == {0: 101, 1: 102}
    assert rp == {0: 1, 1: 2}
    assert rb == {0: 1}
