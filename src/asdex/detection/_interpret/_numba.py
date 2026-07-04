"""Numba-compatible mirrors of the index-set containers in ``_common``.

The pure-Python :class:`~._common.MultiIndexSet` and
:class:`~._common.MultiIndexSetBuilder` cannot be used inside ``@njit`` code
(they inherit ABCs, use ``@classmethod`` and ``np.unique``, and store Python
``set``s). The jitclasses here are lightweight, njit-only stand-ins so hot loops
that read index-set rows and accumulate new ones can be compiled.

Boundaries and constraints:

- These types are **njit-only**. Do not iterate them from Python
  (``list(nb)`` / ``for x in nb``) — that segfaults. Read their array fields
  or use them inside ``@njit`` functions.
- Finalizing a pattern (dedup + sort + CSR layout) stays in Python:
  :class:`NumbaMultiIndexSetBuilder` only accumulates raw ``(label, dep)``
  chunks; :meth:`~._common.MultiIndexSetBuilder.from_numba` hands them to the
  existing Python ``build()``.
- Backing arrays must be ``int64`` and C-contiguous.

Convert to/from the Python classes via ``MultiIndexSet.to_numba()`` /
``MultiIndexSet.from_numba()`` and the matching builder methods in ``_common``.
"""

import numba.core.types as types
import numpy as np
from numba import int64
from numba.experimental.jitclass.decorators import jitclass
from numba.typed import List


@jitclass(spec=[("indices", int64[:]), ("row_offsets", int64[:])])  # type: ignore
class NumbaMultiIndexSet:
    """njit-readable CSR view of per-element dependency sets.

    ``get(i)`` returns element ``i``'s dependency indices as an ``int64``
    array slice. Mirrors the storage of :class:`~._common.MultiIndexSet`.
    """

    def __init__(self, indices, row_offsets):
        self.indices = indices
        self.row_offsets = row_offsets

    @staticmethod
    def create(indices, row_offsets):
        """Python-side factory. Inside ``@njit`` call the constructor directly."""
        return NumbaMultiIndexSet(indices, row_offsets)

    def __len__(self):
        return self.row_offsets.size - 1

    def get(self, i):
        """Return the dependency indices of element ``i`` (an ``int64`` slice)."""
        return self.indices[self.row_offsets[i] : self.row_offsets[i + 1]]


@jitclass(spec=[("length", int64), ("chunks", types.ListType(int64[:, :]))])  # type: ignore
class NumbaMultiIndexSetBuilder:
    """njit-writable accumulator of ``(label, dep)`` pairs.

    Each ``union_into`` / ``append_chunk`` appends a ``(2, k)`` chunk whose
    first row is the element index and second row the dependency indices —
    the same layout as :class:`~._common.MultiIndexSetBuilder._index_arrays`.
    Finalization (dedup/sort) happens in Python via
    :meth:`~._common.MultiIndexSetBuilder.from_numba` + ``build()``.
    """

    def __init__(self, length):
        self.length = length
        self.chunks = List.empty_list(int64[:, :])

    @staticmethod
    def create(length):
        """Python-side factory. Inside ``@njit`` call the constructor directly."""
        return NumbaMultiIndexSetBuilder(length)

    def __len__(self):
        return self.length

    def union_into(self, i, arr):
        """Record that element ``i`` depends on every index in ``arr``."""
        labels = np.full(arr.shape[0], i, dtype=np.int64)
        self.chunks.append(np.stack((labels, arr)))

    def append_chunk(self, chunk):
        """Append a pre-stacked ``(2, k)`` ``(label, dep)`` chunk."""
        self.chunks.append(chunk)
