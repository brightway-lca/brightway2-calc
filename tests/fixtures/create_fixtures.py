from bw_processing import create_datapackage, INDICES_DTYPE
from fs.zipfs import ZipFS
from pathlib import Path
import json
import numpy as np


fixture_dir = Path(__file__).resolve().parent


def bw2io_example_database():
    try:
        import bw2data as bd
        import bw2io as bi
        from bw2data.backends.schema import ActivityDataset as AD

        if "__fixture_creation__" in bd.projects:
            bd.projects.delete_project("__fixture_creation__", delete_dir=True)

        bd.projects.set_current("__fixture_creation__")
        bi.add_example_database()
        db = bd.Database("Mobility example")
        method = bd.Method(("IPCC", "simple"))

        db.filepath_processed().rename(fixture_dir / "bw2io_example_db.zip")
        method.filepath_processed().rename(fixture_dir / "ipcc_simple.zip")
        with open(fixture_dir / "bw2io_example_db_mapping.json", "w") as f:
            json.dump([(obj.name, obj.id) for obj in AD.select()], f)

        bd.projects.delete_project(delete_dir=True)
    except ImportError:
        print("Can't import libraries for bw2io example database fixture creation")


def empty_biosphere():
    # Flow 1: The flow
    # Activity 1: The activity

    dp = create_datapackage(
        fs=ZipFS(str(fixture_dir / "empty_biosphere.zip"), write=True),
    )

    data_array = np.array([1, 2, 3])
    indices_array = np.array([(2, 1), (1, 1), (2, 2)], dtype=INDICES_DTYPE)
    flip_array = np.array([1, 0, 0], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="eb-technosphere",
        indices_array=indices_array,
        nrows=3,
        flip_array=flip_array,
    )

    data_array = np.array([1])
    indices_array = np.array([(1, 0)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="eb-characterization",
        indices_array=indices_array,
        global_index=0,
        nrows=1,
    )

    dp.finalize_serialization()


# def create_mc_basic():
#     with temporary_project_dir() as td:
#         biosphere = bw2data.Database("biosphere")
#         biosphere.write(
#             {
#                 ("biosphere", "1"): {"type": "emission"},
#                 ("biosphere", "2"): {"type": "emission"},
#             }
#         )
#         test_db = bw2data.Database("test")
#         test_db.write(
#             {
#                 ("test", "1"): {
#                     "exchanges": [
#                         {
#                             "amount": 0.5,
#                             "minimum": 0.2,
#                             "maximum": 0.8,
#                             "input": ("test", "2"),
#                             "type": "technosphere",
#                             "uncertainty type": 4,
#                         },
#                         {
#                             "amount": 1,
#                             "minimum": 0.5,
#                             "maximum": 1.5,
#                             "input": ("biosphere", "1"),
#                             "type": "biosphere",
#                             "uncertainty type": 4,
#                         },
#                     ],
#                     "type": "process",
#                 },
#                 ("test", "2"): {
#                     "exchanges": [
#                         {
#                             "amount": 0.1,
#                             "minimum": 0,
#                             "maximum": 0.2,
#                             "input": ("biosphere", "2"),
#                             "type": "biosphere",
#                             "uncertainty type": 4,
#                         }
#                     ],
#                     "type": "process",
#                     "unit": "kg",
#                 },
#             }
#         )
#         method = bw2data.Method(("a", "method"))
#         method.write(
#             [(("biosphere", "1"), 1), (("biosphere", "2"), 2),]
#         )
#         fixture_dir = this_dir / "mc_basic"
#         fixture_dir.mkdir(exist_ok=True)
#         biosphere.filepath_processed().rename(fixture_dir / "biosphere.zip")
#         test_db.filepath_processed().rename(fixture_dir / "test_db.zip")
#         method.filepath_processed().rename(fixture_dir / "method.zip")
#         with open(fixture_dir / "mapping.json", "w") as f:
#             json.dump(list(bw2data.mapping.items()), f)


# def create_mc_single_activity_only_production():
#     with temporary_project_dir() as td:
#         biosphere = bw2data.Database("biosphere")
#         biosphere.write(
#             {("biosphere", "1"): {"type": "emission"},}
#         )
#         saop = bw2data.Database("saop")
#         saop.write(
#             {
#                 ("saop", "1"): {
#                     "exchanges": [
#                         {
#                             "amount": 0.5,
#                             "minimum": 0.2,
#                             "maximum": 0.8,
#                             "input": ("biosphere", "1"),
#                             "type": "biosphere",
#                             "uncertainty type": 4,
#                         },
#                         {
#                             "amount": 1,
#                             "minimum": 0.5,
#                             "maximum": 1.5,
#                             "input": ("saop", "1"),
#                             "type": "production",
#                             "uncertainty type": 4,
#                         },
#                     ],
#                     "type": "process",
#                 },
#             }
#         )
#         fixture_dir = this_dir / "mc_saop"
#         fixture_dir.mkdir(exist_ok=True)
#         biosphere.filepath_processed().rename(fixture_dir / "biosphere.zip")
#         saop.filepath_processed().rename(fixture_dir / "saop.zip")
#         with open(fixture_dir / "mapping.json", "w") as f:
#             json.dump(list(bw2data.mapping.items()), f)


if __name__ == "__main__":
    empty_biosphere()
    bw2io_example_database()

#     create_example_database()
#     create_empty_biosphere()
#     create_mc_basic()
#     create_mc_single_activity_only_production()
