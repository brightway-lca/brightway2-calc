import json
from pathlib import Path

import numpy as np
from bw_processing import INDICES_DTYPE, UNCERTAINTY_DTYPE, create_datapackage
from fs.osfs import OSFS
from fs.zipfs import ZipFS

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


def _create_basic_fixture(fs, characterization=True, characterization_values=True):
    # Activities: 101, 102
    # Products: 1, 2
    # Biosphere flows: 1
    dp = create_datapackage(fs=fs)

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([1])
    indices_array = np.array([(1, 101)], dtype=INDICES_DTYPE)
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    data_array = np.array([1])
    indices_array = np.array([(1, 0)], dtype=INDICES_DTYPE)
    if characterization:
        if characterization_values:
            dp.add_persistent_vector(
                matrix="characterization_matrix",
                data_array=data_array,
                name="eb-characterization",
                indices_array=indices_array,
                global_index=0,
                nrows=1,
            )
        else:
            dp.add_persistent_vector(
                matrix="characterization_matrix",
                data_array=np.array([]),
                name="eb-characterization",
                indices_array=np.array([], dtype=INDICES_DTYPE),
                global_index=0,
                nrows=0,
            )

    dp.finalize_serialization()


def create_basic_fixture_zipfile():
    _create_basic_fixture(ZipFS(str(fixture_dir / "basic_fixture.zip"), write=True))


def create_missing_characterization():
    _create_basic_fixture(
        ZipFS(str(fixture_dir / "missing_characterization.zip"), write=True), characterization=False
    )


def create_empty_characterization():
    _create_basic_fixture(
        ZipFS(str(fixture_dir / "empty_characterization.zip"), write=True),
        characterization_values=False,
    )


def create_basic_fixture_directory():
    _create_basic_fixture(OSFS(str(fixture_dir / "basic_fixture"), create=True))


def create_svdm_fixtures():
    dp = create_datapackage(fs=ZipFS(str(fixture_dir / "svdm.zip"), write=True), sequential=True)

    data_array = np.array([42])
    indices_array = np.array([(1, 1)], dtype=INDICES_DTYPE)
    distributions_array = np.array(
        [
            (4, 0.5, np.NaN, np.NaN, 0.2, 0.8, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
    dp.add_persistent_vector(
        matrix="weighting_matrix",
        data_array=data_array,
        name="weighting",
        indices_array=indices_array,
        distributions_array=distributions_array,
    )

    data_array = np.array([1, 2, 3, 4, 5]).reshape((1, 5))
    indices_array = np.array([(1, 1)], dtype=INDICES_DTYPE)
    dp.add_persistent_array(
        matrix="weighting_matrix",
        data_array=data_array,
        name="weighting2",
        indices_array=indices_array,
    )

    dp.finalize_serialization()

    dp2 = create_datapackage(fs=ZipFS(str(fixture_dir / "svdm2.zip"), write=True))

    data_array = np.array([88])
    indices_array = np.array([(2, 2)], dtype=INDICES_DTYPE)
    dp2.add_persistent_vector(
        matrix="weighting_matrix",
        data_array=data_array,
        name="weighting3",
        indices_array=indices_array,
    )
    dp2.finalize_serialization()


def create_array_fixtures():
    # Activities: 101, 102
    # Products: 1, 2
    # Biosphere flows: 1
    dp = create_datapackage(
        fs=ZipFS(str(fixture_dir / "array_sequential.zip"), write=True), sequential=True
    )

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(1, 101), (2, 102), (2, 101)], dtype=INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    data_array = np.array([[1, 2, 3, 4]])
    indices_array = np.array([(1, 101)], dtype=INDICES_DTYPE)
    dp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
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


def create_mc_basic():
    # Flow 1: biosphere
    # Flow 2: biosphere
    # Flow 3: activity 1
    # Flow 4: activity 2
    # Activity 1
    # Activity 2
    dp = create_datapackage(
        fs=ZipFS(str(fixture_dir / "mc_basic.zip"), write=True),
    )

    data_array = np.array([1, 1, 0.5])
    indices_array = np.array([(3, 1), (4, 2), (4, 1)], dtype=INDICES_DTYPE)
    flip_array = np.array([0, 0, 1], dtype=bool)
    distributions_array = np.array(
        [
            (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (4, 0.5, np.NaN, np.NaN, 0.2, 0.8, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="mc-technosphere",
        indices_array=indices_array,
        distributions_array=distributions_array,
        nrows=3,
        flip_array=flip_array,
    )

    data_array = np.array([1, 0.1])
    indices_array = np.array([(1, 1), (2, 2)], dtype=INDICES_DTYPE)
    distributions_array = np.array(
        [
            (4, 1, np.NaN, np.NaN, 0.5, 1.5, False),
            (4, 0.1, np.NaN, np.NaN, 0, 0.2, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="mc-biosphere",
        indices_array=indices_array,
        distributions_array=distributions_array,
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(1, 0), (2, 0)], dtype=INDICES_DTYPE)
    distributions_array = np.array(
        [
            (4, 1, np.NaN, np.NaN, 0.5, 2, False),
            (4, 2, np.NaN, np.NaN, 1, 4, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=data_array,
        name="mc-characterization",
        indices_array=indices_array,
        distributions_array=distributions_array,
        global_index=0,
        nrows=3,
    )
    dp.finalize_serialization()


def create_multilca_simple():
    # First technosphere
    #           α: 1    β: 2    γ: 3
    #   L: 100  -0.2    -0.5    1
    #   M: 101  1
    #   N: 102  -0.5    1

    dp1 = create_datapackage(
        fs=ZipFS(str(fixture_dir / "multi_lca_simple_1.zip"), write=True),
    )
    data_array = np.array([0.2, 0.5, 1, 1, 0.5, 1])
    indices_array = np.array(
        [(100, 1), (100, 2), (100, 3), (101, 1), (102, 1), (102, 2)], dtype=INDICES_DTYPE
    )
    flip_array = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    # distributions_array = np.array(
    #     [
    #         (4, 0.1, np.NaN, np.NaN, 0.2, 0.8, False),
    #         (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
    #         (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
    #     ],
    #     dtype=UNCERTAINTY_DTYPE,
    # )
    dp1.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        # distributions_array=distributions_array,
        flip_array=flip_array,
    )

    # Second technosphere
    #           δ: 4    ε: 5
    #   L: 100  -0.1
    #   M: 101  -0.2    -0.1
    #   O: 103  -0.4    1
    #   P: 104  1

    dp2 = create_datapackage(
        fs=ZipFS(str(fixture_dir / "multi_lca_simple_2.zip"), write=True),
    )
    data_array = np.array([0.1, 0.2, 0.1, 0.4, 1, 1])
    indices_array = np.array(
        [(100, 4), (101, 4), (101, 5), (103, 4), (103, 5), (104, 4)], dtype=INDICES_DTYPE
    )
    flip_array = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    dp2.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    # Third technosphere
    #           ζ: 6
    #   L: 100  -0.1
    #   N: 102  -0.2
    #   Q: 105  1

    dp3 = create_datapackage(
        fs=ZipFS(str(fixture_dir / "multi_lca_simple_3.zip"), write=True),
    )
    data_array = np.array([0.1, 0.2, 1])
    indices_array = np.array([(100, 6), (102, 6), (105, 6)], dtype=INDICES_DTYPE)
    flip_array = np.array([1, 1, 0], dtype=bool)
    dp3.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data_array,
        name="technosphere",
        indices_array=indices_array,
        flip_array=flip_array,
    )

    # First biosphere
    #           α: 1    β: 2    γ: 3
    #   200     2       4       8
    #   201     1       2       3

    data_array = np.array([2, 4, 8, 1, 2, 3])
    indices_array = np.array(
        [(200, 1), (200, 2), (200, 3), (201, 1), (201, 2), (201, 3)], dtype=INDICES_DTYPE
    )
    # distributions_array = np.array(
    #     [
    #         (4, 0.1, np.NaN, np.NaN, 0.2, 0.8, False),
    #         (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
    #         (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
    #     ],
    #     dtype=UNCERTAINTY_DTYPE,
    # )
    dp1.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
        # distributions_array=distributions_array,
    )

    # Second biosphere
    #           δ: 4    ε: 5
    #   200     1       2
    #   202             1

    data_array = np.array([1, 2, 1])
    indices_array = np.array([(200, 4), (200, 5), (202, 5)], dtype=INDICES_DTYPE)
    dp2.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    # Third biosphere
    #           ζ: 6
    #   200     1
    #   202     2
    #   203     3

    data_array = np.array([1, 2, 3])
    indices_array = np.array([(200, 6), (202, 6), (203, 6)], dtype=INDICES_DTYPE)
    dp3.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=data_array,
        name="biosphere",
        indices_array=indices_array,
    )

    # First characterization
    # 200-206   1

    dp4 = create_datapackage(
        fs=ZipFS(str(fixture_dir / "multi_lca_simple_4.zip"), write=True),
    )
    indices_array = np.array(
        [(200, 0), (201, 0), (202, 0), (203, 0), (204, 0), (205, 0), (206, 0)], dtype=INDICES_DTYPE
    )
    distributions_array = np.array(
        [
            (4, 0.1, np.NaN, np.NaN, 0.5, 1.5, False),
            (4, 0.1, np.NaN, np.NaN, 0.5, 1.5, False),
            (4, 0.1, np.NaN, np.NaN, 0.5, 1.5, False),
            (4, 0.1, np.NaN, np.NaN, 0.5, 1.5, False),
            (4, 0.1, np.NaN, np.NaN, 0.5, 1.5, False),
            (4, 0.1, np.NaN, np.NaN, 0.5, 1.5, False),
            (4, 0.1, np.NaN, np.NaN, 0.5, 1.5, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
    dp4.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=np.ones(7),
        name="characterization-1",
        identifier=("first", "category"),
        indices_array=indices_array,
        distributions_array=distributions_array,
        global_index=0,
    )

    # Second characterization
    # 201   10
    # 203   10
    # 205   10

    dp5 = create_datapackage(
        fs=ZipFS(str(fixture_dir / "multi_lca_simple_5.zip"), write=True),
    )
    indices_array = np.array([(201, 0), (203, 0), (205, 0)], dtype=INDICES_DTYPE)
    distributions_array = np.array(
        [
            (5, 10, np.NaN, np.NaN, 8, 15, False),
            (5, 10, np.NaN, np.NaN, 8, 15, False),
            (5, 10, np.NaN, np.NaN, 8, 15, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
    dp5.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=np.ones(3) * 10,
        name="characterization-1",
        identifier=("second", "category"),
        indices_array=indices_array,
        distributions_array=distributions_array,
        global_index=0,
    )

    dp1.finalize_serialization()
    dp2.finalize_serialization()
    dp3.finalize_serialization()
    dp4.finalize_serialization()
    dp5.finalize_serialization()


def create_mc_complete():
    # Flow 1: biosphere
    # Flow 2: biosphere
    # Flow 3: activity 1
    # Flow 4: activity 2
    # Activity 1
    # Activity 2
    dp = create_datapackage(
        fs=ZipFS(str(fixture_dir / "mc_complete.zip"), write=True),
    )

    data_array = np.array([1, 2])
    indices_array = np.array([(100, 0), (200, 0)], dtype=INDICES_DTYPE)
    distributions_array = np.array(
        [
            (4, 100, np.NaN, np.NaN, 50, 200, False),
            (4, 200, np.NaN, np.NaN, 100, 400, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
    dp.add_persistent_vector(
        matrix="normalization_matrix",
        data_array=data_array,
        name="mc-normalization",
        indices_array=indices_array,
        distributions_array=distributions_array,
    )

    data_array = np.array([1])
    indices_array = np.array([(0, 0)], dtype=INDICES_DTYPE)
    distributions_array = np.array(
        [
            (4, 1, np.NaN, np.NaN, 0.5, 2, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
    dp.add_persistent_vector(
        matrix="weighting_matrix",
        data_array=data_array,
        name="mc-weighting",
        indices_array=indices_array,
        distributions_array=distributions_array,
    )
    dp.finalize_serialization()


if __name__ == "__main__":
    # empty_biosphere()
    # bw2io_example_database()
    # create_mc_basic()
    # create_mc_complete()
    # create_missing_characterization()
    # create_empty_characterization()
    # create_basic_fixture_zipfile()
    # create_basic_fixture_directory()
    # create_array_fixtures()
    # create_svdm_fixtures()
    create_multilca_simple()
