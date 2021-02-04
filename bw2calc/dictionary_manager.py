from collections.abc import Mapping
from functools import partial


def resolved(f):
    """Decorator that resolves a ``partial`` function before it can be used"""
    def wrapper(self, *args):
        if not self._resolved:
            self._dict = self._partial()
            self._resolved = True
            delattr(self, "_partial")
        return f(self, *args)
    return wrapper


class ReversibleRemappableDictionary(Mapping):
    """A dictionary that can be easily remapped or reversed.

    Perhaps  overkill, but at the time it was easier than creating many dictionaries on the LCA object itself.

    Example usage::

        In [1]: from bw2calc.dictionary_manager import ReversibleRemappableDictionary

        In [2]: d = ReversibleRemappableDictionary({1: 2})

        In [3]: d.reverse
        Out[3]: {2: 1}

        In [4]: d.remap({1: "foo"})

        In [5]: d['foo']
        Out[5]: 2

        In [6]: d.original
        Out[6]: {1: 2}

        In [7]: d.reverse
        Out[7]: {2: 'foo'}

        In [8]: d.unmap()

        In [9]: d[1]
        Out[9]: 2

    """

    def __init__(self, obj):
        if isinstance(obj, partial):
            self._resolved = False
            self._partial = obj
        elif isinstance(obj, Mapping):
            self._resolved = True
            self._dict = obj
        else:
            raise ValueError("Input must be a dict")

    @property
    @resolved
    def reversed(self):
        if not hasattr(self, "_reversed"):
            self._reversed = {v: k for k, v in self.items()}
        return self._reversed

    @property
    @resolved
    def original(self):
        if not hasattr(self, "_original"):
            return self
        return self._original

    @resolved
    def remap(self, mapping):
        """Transform the keys based on the mapping dict ``mapping``.

        ``mapping`` doesn't need to cover every key in the original.

        Example usage:

        {1: 2}.remap({1: "foo"} >> {"foo": 2}

        """
        if not isinstance(mapping, Mapping):
            raise ValueError
        if hasattr(self, "_reversed"):
            delattr(self, "_reversed")
        self._original = self._dict.copy()
        self._dict = {mapping.get(k, k): v for k, v in self.items()}

    @resolved
    def unmap(self):
        """Restore dict to original state."""
        if hasattr(self, "_reversed"):
            delattr(self, "_reversed")
        self._dict = self._original
        delattr(self, "_original")

    @resolved
    def __getitem__(self, key):
        return self._dict[key]

    @resolved
    def __iter__(self):
        return iter(self._dict)

    @resolved
    def __len__(self):
        return len(self._dict)

    @resolved
    def __str__(self):
        return self._dict.__str__()


class DictionaryManager:
    """Class that handles dictionaries which can be remapped or reverse.

    Usage::

        dm = DictionaryManager()
        dm.foo = {1: 2}
        dm.foo[1]
        >> 2

    """

    def __init__(self):
        self._dicts = {}

    def __getattr__(self, attr):
        try:
            return self._dicts[attr]
        except KeyError:
            raise ValueError("This dictionary not yet created")

    def __setattr__(self, attr, value):
        if attr == "_dicts":
            super().__setattr__(attr, value)
        else:
            self._dicts[attr] = ReversibleRemappableDictionary(value)

    def __len__(self):
        return len(self._dicts)

    def __iter__(self):
        return iter(self._dicts)

    def __str__(self):
        return "Dictionary manager with {} keys:".format(len(self))
