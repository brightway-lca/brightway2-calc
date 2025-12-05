import math
from collections.abc import Mapping
from typing import List

import numpy as np


class ResultCache(Mapping):
    def __init__(self, block_size: int = 100):
        """This class allows supply vector results to be cached."""
        self.next_index = 0
        self.block_size = block_size
        self.indices = dict()

    def __getitem__(self, key: int) -> np.ndarray:
        if not hasattr(self, "array"):
            raise KeyError
        return self.array[:, self.indices[key]]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __contains__(self, key: int) -> bool:
        return key in self.indices

    def add(self, indices: List[int], array: np.ndarray) -> None:
        if not hasattr(self, "array"):
            self.array = np.empty((array.shape[0], self.block_size), dtype=np.float32)

        if array.shape[0] != self.array.shape[0]:
            raise ValueError(
                f"Wrong number of rows in array ({array.shape[0]} should be {self.array.shape[0]})"
            )
        if len(array.shape) != 2:
            raise ValueError(
                f"`array` must be a numpy array with two dimensions (got {len(array.shape)})"
            )
        if len(indices) != array.shape[1]:
            raise ValueError(
                f"`indices` has different length than `array` ({len(indices)} vs. {array.shape[1]})"
            )

        if (total_columns := self.next_index + array.shape[1]) > self.array.shape[1]:
            extra_blocks = math.ceil((total_columns - self.array.shape[1]) / self.block_size)
            self.array = np.hstack(
                (self.array, np.empty((self.array.shape[0], self.block_size * extra_blocks)))
            )

        # Would be faster with numpy bool arrays
        for enum_index, data_obj_index in enumerate(indices):
            if data_obj_index not in self.indices:
                self.indices[data_obj_index] = self.next_index
                self.array[:, self.next_index] = array[:, enum_index]
                self.next_index += 1

    def reset(self) -> None:
        self.indices = dict()
        self.next_index = 0
        delattr(self, "array")
