# -*- coding: utf-8 -*-
from .errors import AllArraysEmpty, NoArrays
import numpy as np


def filter_matrix_data(packages, matrix_label, empty_ok=False):
    arrays = [
        package[resource["path"]]
        for package in packages
        for resource in package["datapackage"]["resources"]
        if resource["matrix"] == matrix_label
    ]
    if all(arr.shape[0] == 0 for arr in arrays) and not empty_ok:
        raise AllArraysEmpty
    elif not arrays:
        raise NoArrays(f"No arrays for '{matrix_label}'")
    return np.hstack(arrays)


def get_seed(seed=None):
    """Get valid Numpy random seed value"""
    # https://groups.google.com/forum/#!topic/briansupport/9ErDidIBBFM
    random = np.random.RandomState(seed)
    return random.randint(0, 2147483647)


# def save_calculation_package(name, demand, **kwargs):
#     """Save a calculation package for later use in an independent LCA.

#     Args:
#         * name (str): Name of file to create. Will have datetime appended.
#         * demand (dict): Demand dictionary.
#         * kwargs: Any additional keyword arguments, e.g. ``method``, ``iterations``...

#     Returns the filepath of the calculation package archive.

#     """
#     _ = lambda x: [os.path.basename(y) for y in x]

#     filepaths = get_filepaths(demand, "demand")
#     data = {
#         "demand": {mapping[k]: v for k, v in demand.items()},
#         "database_filepath": _(filepaths),
#         "adjust_filepaths": ["database_filepath"],
#     }
#     for key, value in kwargs.items():
#         if key in OBJECT_MAPPING:
#             data[key] = _(get_filepaths(value, key))
#             data["adjust_filepaths"].append(key)
#             filepaths.extend(get_filepaths(value, key))
#         else:
#             data[key] = value

#     config_fp = os.path.join(
#         projects.output_dir, "{}.config.json".format(safe_filename(name))
#     )
#     with open(config_fp, "w") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)
#     archive = os.path.join(
#         projects.output_dir,
#         "{}.{}.tar.gz".format(
#             safe_filename(name, False),
#             datetime.datetime.now().strftime("%d-%B-%Y-%I-%M%p"),
#         ),
#     )
#     with tarfile.open(archive, "w:gz") as tar:
#         tar.add(config_fp, arcname=os.path.basename(config_fp))
#         for filepath in filepaths:
#             tar.add(filepath, arcname=os.path.basename(filepath))

#     os.remove(config_fp)
#     return archive

# def load_calculation_package(fp):
#     """Load a calculation package created by ``save_calculation_package``.

#     NumPy arrays are saved to a temporary directory, and file paths are adjusted.

#     ``fp`` is the absolute file path of a calculation package file.

#     Returns a dictionary suitable for passing to an LCA object, e.g. ``LCA(**load_calculation_package(fp))``.

#     """
#     assert os.path.exists(fp), "Can't find file: {}".format(fp)

#     temp_dir = tempfile.mkdtemp()
#     with tarfile.open(fp, "r|gz") as tar:
#         tar.extractall(temp_dir)

#     config_fps = [x for x in os.listdir(temp_dir) if x.endswith(".config.json")]
#     assert len(config_fps) == 1, "Can't find configuration file"
#     config = json.load(open(os.path.join(temp_dir, config_fps[0])))
#     config["demand"] = {int(k): v for k, v in config["demand"].items()}

#     for field in config.pop("adjust_filepaths"):
#         config[field] = [os.path.join(temp_dir, fn) for fn in config[field]]

#     return config
def wrap_functional_unit(dct):
    """Transform functional units for effective logging.
    Turns ``Activity`` objects into their keys."""
    data = []
    for key, amount in dct.items():
        if isinstance(key, int):
            data.append({"id": key, "amount": amount})
        else:
            try:
                data.append({'database': key[0],
                             'code': key[1],
                             'amount': amount})
            except TypeError:
                data.append({'key': key,
                             'amount': amount})
    return data
