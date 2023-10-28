from collections.abc import Mapping
from typing import Any, Sequence


class TwoDimensionalGrid(Mapping):
    def __init__(self, keys: Sequence[tuple[Any, Any]], values: Sequence[Any]):
        """Read-only dictionary wrapper for a strictly 2-dimensional grid.

        Supports a very limited type of slicing - only `foo["something", ...]`."""
        if not len(keys) == len(values):
            raise ValueError("`keys` must have same length as `values`")
        self.dict_ = {k: v for k, v in zip(keys, values)}

    def __getitem__(self, key: Any):
        if not len(key) == 2:
            raise KeyError
        first, second = key
        if first == Ellipsis and second == Ellipsis:
            raise KeyError
        elif first == Ellipsis:
            return {f: v for (f, s), v in self.dict_.items() if s == second}
        elif second == Ellipsis:
            return {s: v for (f, s), v in self.dict_.items() if f == first}
        else:
            return self.dict_[(first, second)]

    def __iter__(self):
        yield from self.dict_

    def __len__(self):
        return len(self.dict_)
