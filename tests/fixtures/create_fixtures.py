import bw2data
import bw2io
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
import json

this_dir = Path(__file__).resolve().parent


@contextmanager
def temporary_project_dir():
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        bw2data.projects._base_data_dir = temp_dir / "data"
        bw2data.projects._base_logs_dir = temp_dir / "logs"
        bw2data.projects.db.change_path(":memory:")
        bw2data.projects.set_current("default", update=False)
        assert len(bw2data.projects) == 1
        assert not bw2data.databases
        yield temp_dir


def create_example_database():
    with temporary_project_dir() as td:
        bw2io.add_example_database()
        db = bw2data.Database("Mobility example")
        method = bw2data.Method(('IPCC', 'simple'))

        fixture_dir = this_dir / "example_db"
        fixture_dir.mkdir(exist_ok=True)
        db.filepath_processed().rename(fixture_dir / "example_db.zip")
        method.filepath_processed().rename(fixture_dir / "ipcc.zip")
        with open(fixture_dir / "mapping.json", "w") as f:
            json.dump(list(bw2data.mapping.items()), f)


def create_empty_biosphere():
    with temporary_project_dir() as td:
        biosphere = bw2data.Database("biosphere")
        biosphere.write(
            {
                ("biosphere", "1"): {
                    "categories": ["things"],
                    "exchanges": [],
                    "name": "an emission",
                    "type": "emission",
                    "unit": "kg",
                }
            }
        )

        test_data = {
            ("t", "1"): {
                "exchanges": [{"amount": 1, "input": ("t", "2"), "type": "technosphere",}],
            },
            ("t", "2"): {"exchanges": []},
        }
        test_db = bw2data.Database("t")
        test_db.write(test_data)

        method = bw2data.Method(("a method",))
        method.write([(("biosphere", "1"), 42)])

        fixture_dir = this_dir / "empty_biosphere"
        fixture_dir.mkdir(exist_ok=True)
        biosphere.filepath_processed().rename(fixture_dir / "biosphere.zip")
        test_db.filepath_processed().rename(fixture_dir / "test_db.zip")
        method.filepath_processed().rename(fixture_dir / "method.zip")
        with open(fixture_dir / "mapping.json", "w") as f:
            json.dump(list(bw2data.mapping.items()), f)


def create_mc_basic():
    with temporary_project_dir() as td:
        biosphere = bw2data.Database("biosphere")
        biosphere.write(
            {
                ("biosphere", "1"): {"type": "emission"},
                ("biosphere", "2"): {"type": "emission"},
            }
        )
        test_db = bw2data.Database("test")
        test_db.write(
            {
                ("test", "1"): {
                    "exchanges": [
                        {
                            "amount": 0.5,
                            "minimum": 0.2,
                            "maximum": 0.8,
                            "input": ("test", "2"),
                            "type": "technosphere",
                            "uncertainty type": 4,
                        },
                        {
                            "amount": 1,
                            "minimum": 0.5,
                            "maximum": 1.5,
                            "input": ("biosphere", "1"),
                            "type": "biosphere",
                            "uncertainty type": 4,
                        },
                    ],
                    "type": "process",
                },
                ("test", "2"): {
                    "exchanges": [
                        {
                            "amount": 0.1,
                            "minimum": 0,
                            "maximum": 0.2,
                            "input": ("biosphere", "2"),
                            "type": "biosphere",
                            "uncertainty type": 4,
                        }
                    ],
                    "type": "process",
                    "unit": "kg",
                },
            }
        )
        method = bw2data.Method(("a", "method"))
        method.write(
            [(("biosphere", "1"), 1), (("biosphere", "2"), 2),]
        )
        fixture_dir = this_dir / "mc_basic"
        fixture_dir.mkdir(exist_ok=True)
        biosphere.filepath_processed().rename(fixture_dir / "biosphere.zip")
        test_db.filepath_processed().rename(fixture_dir / "test_db.zip")
        method.filepath_processed().rename(fixture_dir / "method.zip")
        with open(fixture_dir / "mapping.json", "w") as f:
            json.dump(list(bw2data.mapping.items()), f)


if __name__ == "__main__":
    # create_example_database()
    # create_empty_biosphere()
    create_mc_basic()
