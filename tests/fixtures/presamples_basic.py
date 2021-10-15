from bw2data import Database, Method, projects
import numpy as np
import os
import uuid

basedir = os.path.dirname(os.path.abspath(__file__))

# Technosphere matrix

[[1, 2, -3], [0, 0.5, -2], [-0.1, 0, 1]]

# Biosphere matrix

[[0, 1, 2], [7, 5, 0]]

# Characterization matrix

[[4, 0], [0, -2]]

# A^{-1}

[
    [2 / 3, 4 / 15, 2 / 30],
    [-(2 + 2 / 3), 14 / 15, -4 / 15],
    [-(3 + 1 / 3), 2 + 2 / 3, 2 / 3],
]


def write_database():
    bio_data = {
        ("bio", "a"): {"exchange": [], "type": "biosphere"},
        ("bio", "b"): {"exchange": [], "type": "biosphere"},
    }
    Database("bio").write(bio_data)

    tech_data = {
        ("test", "1"): {
            "exchanges": [
                {
                    "amount": 1,
                    "type": "production",
                    "input": ("test", "1"),
                    "uncertainty type": 0,
                },
                {
                    "amount": 0.1,
                    "type": "technosphere",
                    "input": ("test", "3"),
                    "uncertainty type": 0,
                },
                {
                    "amount": 7,
                    "type": "biosphere",
                    "input": ("bio", "b"),
                    "uncertainty type": 0,
                },
            ],
        },
        ("test", "2"): {
            "exchanges": [
                {
                    "amount": 0.5,
                    "type": "production",
                    "input": ("test", "2"),
                    "uncertainty type": 0,
                },
                {
                    "amount": -2,
                    "type": "technosphere",
                    "input": ("test", "1"),
                    "uncertainty type": 0,
                },
                {
                    "amount": 1,
                    "type": "biosphere",
                    "input": ("bio", "a"),
                    "uncertainty type": 0,
                },
                {
                    "amount": 5,
                    "type": "biosphere",
                    "input": ("bio", "b"),
                    "uncertainty type": 0,
                },
            ],
        },
        ("test", "3"): {
            "exchanges": [
                {
                    "amount": 1,
                    "type": "production",
                    "input": ("test", "3"),
                    "uncertainty type": 0,
                },
                {
                    "amount": 3,
                    "type": "technosphere",
                    "input": ("test", "1"),
                    "uncertainty type": 0,
                },
                {
                    "amount": 2,
                    "type": "technosphere",
                    "input": ("test", "2"),
                    "uncertainty type": 0,
                },
                {
                    "amount": 2,
                    "type": "biosphere",
                    "input": ("bio", "a"),
                    "uncertainty type": 0,
                },
            ],
        },
    }

    Database("test").write(tech_data)

    cfs = [
        (("bio", "a"), 4),
        (("bio", "b"), -2),
    ]

    Method(("m",)).register()
    Method(("m",)).write(cfs)


def build_single_presample_array():
    from presamples import create_presamples_package

    tech_indices = [
        (("test", "1"), ("test", "2"), "technosphere"),
        (("test", "2"), ("test", "2"), "production"),
    ]

    tech_samples = np.array(
        (
            [1],
            [1],
        )
    )

    bio_indices = [
        (("bio", "a"), ("test", "2")),
        (("bio", "b"), ("test", "2")),
        (("bio", "b"), ("test", "1")),
    ]

    bio_samples = np.array(
        (
            [10],
            [1],
            [0],
        )
    )

    cf_indices = [("bio", "a")]

    cf_samples = np.array(([1],))

    create_presamples_package(
        matrix_data=[
            (tech_samples, tech_indices, "technosphere"),
            (bio_samples, bio_indices, "biosphere"),
            (cf_samples, cf_indices, "cf"),
        ],
        id_="single-sample",
        name="single-sample",
        dirpath=basedir,
        overwrite=True,
        seed=54321,
    )


def build_multi_presample_array_unseeded():
    from presamples import create_presamples_package

    tech_indices = [
        (("test", "1"), ("test", "2"), "technosphere"),
        (("test", "2"), ("test", "2"), "production"),
    ]

    tech_samples = np.array(
        (
            [1, 2, 3],
            [100, 101, 102],
        )
    )

    bio_indices = [
        (("bio", "a"), ("test", "2")),
        (("bio", "b"), ("test", "2")),
        (("bio", "b"), ("test", "1")),
    ]

    bio_samples = np.array(
        (
            [10, 11, 12],
            [1, 2, 3],
            [0, -1, -2],
        )
    )

    create_presamples_package(
        matrix_data=[
            (tech_samples, tech_indices, "technosphere"),
            (bio_samples, bio_indices, "biosphere"),
        ],
        id_="unseeded",
        name="unseeded",
        dirpath=basedir,
        overwrite=True,
    )


def build_multi_presample_array():
    from presamples import create_presamples_package

    tech_indices = [
        (("test", "1"), ("test", "2"), "technosphere"),
        (("test", "2"), ("test", "2"), "production"),
    ]

    tech_samples = np.array(
        (
            [1, 2, 3],
            [100, 101, 102],
        )
    )

    bio_indices = [
        (("bio", "a"), ("test", "2")),
        (("bio", "b"), ("test", "2")),
        (("bio", "b"), ("test", "1")),
    ]

    bio_samples = np.array(
        (
            [10, 11, 12],
            [1, 2, 3],
            [0, -1, -2],
        )
    )

    create_presamples_package(
        matrix_data=[
            (tech_samples, tech_indices, "technosphere"),
            (bio_samples, bio_indices, "biosphere"),
        ],
        id_="multi",
        name="multi",
        dirpath=basedir,
        overwrite=True,
        seed=42,
    )


def build_multi_presample_sequential_array():
    from presamples import create_presamples_package

    tech_indices = [
        (("test", "1"), ("test", "2"), "technosphere"),
        (("test", "2"), ("test", "2"), "production"),
    ]

    tech_samples = np.array(
        (
            [1, 2, 3],
            [100, 101, 102],
        )
    )

    bio_indices = [
        (("bio", "a"), ("test", "2")),
        (("bio", "b"), ("test", "2")),
        (("bio", "b"), ("test", "1")),
    ]

    bio_samples = np.array(
        (
            [10, 11, 12],
            [1, 2, 3],
            [0, -1, -2],
        )
    )

    create_presamples_package(
        matrix_data=[
            (tech_samples, tech_indices, "technosphere"),
            (bio_samples, bio_indices, "biosphere"),
        ],
        id_="seq",
        name="seq",
        dirpath=basedir,
        overwrite=True,
        seed="sequential",
    )


if __name__ == "__main__":
    name = "test-builder-{}".format(uuid.uuid4().hex)
    if name in projects:
        raise ValueError("Test project name not unique; please run again.")
    try:
        projects.set_current(name)
        write_database()
        build_single_presample_array()
        build_multi_presample_array()
        build_multi_presample_array_unseeded()
        build_multi_presample_sequential_array()
    finally:
        projects.delete_project(delete_dir=True)
