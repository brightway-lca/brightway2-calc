from bw2calc import LCA, IndependentLCAMixin
from bw2data.tests import BW2DataTest, bw2test
from bw2data.utils import TYPE_DICTIONARY, MAX_INT_32
from io import BytesIO
import numpy as np
import os

basedir = os.path.join(os.path.dirname(__file__), "fixtures", "independent")
inv_fp = os.path.join(basedir, "inv.npy")
ia_fp = os.path.join(basedir, "ia.npy")


@bw2test
def test_independent_lca_with_global_value(monkeypatch):
    monkeypatch.setattr(
        'bw2calc.lca.global_index',
        17
    )

    class ILCA(IndependentLCAMixin, LCA):
        pass

    lca = ILCA({15: 1}, database_filepath=[inv_fp], method=[ia_fp])
    lca.lci()
    lca.lcia()
    print(lca.score)
    assert lca.score == 8020


@bw2test
def test_independent_lca_with_no_global_value(monkeypatch):
    monkeypatch.setattr(
        'bw2calc.lca.global_index',
        None
    )

    class ILCA(IndependentLCAMixin, LCA):
        pass

    lca = ILCA({15: 1}, database_filepath=[inv_fp], method=[ia_fp])
    lca.lci()
    lca.lcia()
    print(lca.score)
    assert lca.score == 8020


@bw2test
def test_independent_lca_with_directly_passing_array(monkeypatch):
    monkeypatch.setattr(
        'bw2calc.lca.global_index',
        None
    )

    class ILCA(IndependentLCAMixin, LCA):
        pass

    ia = np.load(ia_fp, allow_pickle=True)
    lca = ILCA({15: 1}, database_filepath=[inv_fp], method=[ia])
    lca.lci()
    lca.lcia()
    assert lca.score == 8020


@bw2test
def test_independent_lca_with_passing_bytes_array(monkeypatch):
    monkeypatch.setattr(
        'bw2calc.lca.global_index',
        None
    )

    class ILCA(IndependentLCAMixin, LCA):
        pass

    with BytesIO() as buffer:
        np.save(buffer, np.load(ia_fp, allow_pickle=False))
        buffer.seek(0)
        lca = ILCA({15: 1}, database_filepath=[inv_fp], method=[buffer])
        lca.lci()
        lca.lcia()
        assert lca.score == 8020


if __name__ == '__main__':

    """

    Biosphere flows:

    10: A
    11: B

    Activities:

    12: C
    13: D
    14: E

    Products:

    15: F
    16: G
    17: H

    CFs:

        A: 1, B: 10

    Exchanges:

        F -> C: 1, G -> D: 1, H -> E: 1
        G -> C: 2
        H -> D: 4
        A -> D: 10
        B -> E: 100

    """
    inv_data = [
        (15, 12, 1,   'production'),
        (16, 13, 1,   'production'),
        (17, 14, 1,   'production'),
        (16, 12, 2,   'technosphere'),
        (17, 13, 4,   'technosphere'),
        (10, 13, 10,  'biosphere'),
        (11, 14, 100, 'biosphere'),
    ]

    dtype = [
        ('input', np.uint32),
        ('output', np.uint32),
        ('row', np.uint32),
        ('col', np.uint32),
        ('type', np.uint8),
        ('amount', np.float32),
    ]
    arr = np.zeros((len(inv_data),), dtype=dtype)

    for index, line in enumerate(inv_data):
        arr[index] = (
            line[0],
            line[1],
            MAX_INT_32,
            MAX_INT_32,
            TYPE_DICTIONARY[line[3]],
            line[2],
        )

    arr.sort(order=['input', 'output', 'amount'])
    np.save(inv_fp, arr, allow_pickle=False)

    ia_data = [
        (10, 1),
        (11, 10),
    ]

    dtype = [
        ('flow', np.uint32),
        ('row', np.uint32),
        ('col', np.uint32),
        ('geo', np.uint32),
        ('amount', np.float32),
    ]
    arr = np.zeros((len(ia_data),), dtype=dtype)

    for index, line in enumerate(ia_data):
        arr[index] = (
            line[0],
            MAX_INT_32,
            MAX_INT_32,
            17,
            line[1],
        )

    arr.sort(order=['flow', 'amount'])
    np.save(ia_fp, arr, allow_pickle=False)
