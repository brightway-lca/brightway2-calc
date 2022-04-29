# -*- coding: utf-8 -*-
from pathlib import Path
import itertools
import json
import numpy as np
import os
import random
import tarfile
import tempfile


def generate_fixture():
    dtype = [
        ("input", np.uint32),
        ("output", np.uint32),
        ("row", np.uint32),
        ("col", np.uint32),
        ("type", np.uint8),
        ("uncertainty_type", np.uint8),
        ("amount", np.float32),
        ("loc", np.float32),
        ("scale", np.float32),
        ("shape", np.float32),
        ("minimum", np.float32),
        ("maximum", np.float32),
        ("negative", bool),
    ]

    # Exchange types
    # "generic production": 11,
    # "generic consumption": 12,

    MAX_INT_32 = 4294967295
    LETTERS = "abcdefgh"
    NUMBERS = "1234"
    GREEK = "αβγδεζηθ"
    mapping = {k: i for i, k in enumerate(LETTERS + NUMBERS + GREEK)}

    a = [(a, b, random.random(), 12) for a, b in itertools.combinations(LETTERS, 2)]
    b = [(x, x, 1, 11) for x in LETTERS]
    c = [
        (random.choice(NUMBERS), random.choice(LETTERS), random.random(), 11)
        for _ in range(10)
    ]
    d = [(x, x, 1, 11) for x in NUMBERS]
    e = [(x, y, random.random(), 11) for x, y in zip(GREEK, NUMBERS)]
    f = [(x, x, 1, 11) for x in GREEK]

    data = a + b + c + d + e + f
    array = np.zeros(len(data), dtype=dtype)

    for i, (a, b, c, d) in enumerate(data):
        array[i] = (
            mapping[a],
            mapping[b],
            MAX_INT_32,
            MAX_INT_32,
            d,
            0,
            c,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            False,
        )

    with tempfile.TemporaryDirectory() as t:
        with tarfile.open(Path(t) / "sm-fixture.tar.bz2", "w:bz2") as f:
            path = os.path.join(t, "array.npy")
            np.save(path, array, allow_pickle=False)
            f.add(path, "array.npy")

            path = os.path.join(t, "row.mapping")
            with open(path, "w", encoding="utf-8") as j:
                json.dump(mapping, j, ensure_ascii=False)
            f.add(path, "row.mapping")
            f.add(path, "col.mapping")

            path = os.path.join(t, "categories.mapping")
            with open(path, "w", encoding="utf-8") as j:
                json.dump(
                    {"foo": {g: mapping[g] for g in GREEK[:5]}}, j, ensure_ascii=False
                )
            f.add(path, "categories.mapping")


generate_fixture()
