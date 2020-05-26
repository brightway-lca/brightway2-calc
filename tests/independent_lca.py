# from bw2calc import LCA
# from bw2data.tests import bw2test
# from bw_processing import create_calculation_package
# from io import BytesIO
# from pathlib import Path
# import numpy as np

# basedir = Path(__file__).resolve().parent / "fixtures" / "independent"
# inv_fp = basedir / "inventory.zip"
# ia_fp = basedir / "ia.zip"


# @bw2test
# def test_independent_lca_with_global_value(monkeypatch):
#     class ILCA(IndependentLCAMixin, LCA):
#         pass

#     lca = ILCA({15: 1}, database_filepath=[inv_fp], method=[ia_fp])
#     lca.lci()
#     lca.lcia()
#     assert lca.score == 8020


# @bw2test
# def test_independent_lca_with_no_global_value(monkeypatch):
#     class ILCA(IndependentLCAMixin, LCA):
#         pass

#     lca = ILCA({15: 1}, database_filepath=[inv_fp], method=[ia_fp])
#     lca.lci()
#     lca.lcia()
#     assert lca.score == 8020


# @bw2test
# def test_independent_lca_with_directly_passing_array(monkeypatch):
#     class ILCA(IndependentLCAMixin, LCA):
#         pass

#     ia = np.load(ia_fp, allow_pickle=True)
#     lca = ILCA({15: 1}, database_filepath=[inv_fp], method=[ia])
#     lca.lci()
#     lca.lcia()
#     assert lca.score == 8020


# @bw2test
# def test_independent_lca_with_passing_bytes_array(monkeypatch):
#     class ILCA(IndependentLCAMixin, LCA):
#         pass

#     with BytesIO() as buffer:
#         np.save(buffer, np.load(ia_fp, allow_pickle=False))
#         buffer.seek(0)
#         lca = ILCA({15: 1}, database_filepath=[inv_fp], method=[buffer])
#         lca.lci()
#         lca.lcia()
#         assert lca.score == 8020


# if __name__ == "__main__":

#     """

#     Biosphere flows:

#     10: A
#     11: B

#     Activities:

#     12: C
#     13: D
#     14: E

#     Products:

#     15: F
#     16: G
#     17: H

#     CFs:

#         A: 1, B: 10

#     Exchanges:

#         F -> C: 1, G -> D: 1, H -> E: 1
#         G -> C: 2
#         H -> D: 4
#         A -> D: 10
#         B -> E: 100

#     """
#     technosphere_data = [
#         {"row": 15, "col": 12, "amount": 1, "flip": False},
#         {"row": 16, "col": 13, "amount": 1, "flip": False},
#         {"row": 17, "col": 14, "amount": 1, "flip": False},
#         {"row": 16, "col": 12, "amount": 2, "flip": True},
#         {"row": 17, "col": 13, "amount": 4, "flip": True},
#     ]
#     biosphere_data = [
#         {"row": 10, "col": 13, "amount": 10, "flip": False},
#         {"row": 11, "col": 14, "amount": 100, "flip": False},
#     ]
#     create_calculation_package(
#         name="inventory",
#         path=basedir,
#         resources=[
#             {
#                 "name": "technosphere_matrix",
#                 "path": "technosphere_matrix.npy",
#                 "matrix": "technosphere_matrix",
#                 "data": technosphere_data,
#             },
#             {
#                 "name": "biosphere_matrix",
#                 "path": "biosphere_matrix.npy",
#                 "matrix": "biosphere_matrix",
#                 "data": biosphere_data,
#             },
#         ],
#     )

#     ia_data = [
#         {"row": 10, "amount": 1},
#         {"row": 11, "amount": 10},
#     ]
#     create_calculation_package(
#         name="ia",
#         path=basedir,
#         resources=[
#             {
#                 "name": "characterization_matrix",
#                 "path": "characterization_matrix.npy",
#                 "matrix": "characterization_matrix",
#                 "data": ia_data,
#             }
#         ],
#     )
