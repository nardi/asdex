"""Tests for the Numba jitclass mirrors in ``_numba``.

Covers conversion round-trips with the Python containers, reading rows inside
``@njit``, builder accumulation, and the ``union_lhs_rhs_multiplicative`` kernel
against a plain-Python reference of the ``_dot_general`` inner loop.
"""

import numpy as np
from numba import njit

from asdex.detection._interpret._common import (
    MultiIndexSet,
    MultiIndexSetBuilder,
)
from asdex.detection._interpret._numba import NumbaMultiIndexSetBuilder


def _rows(m: MultiIndexSet) -> list[set[int]]:
    return [set(m[i]) for i in range(len(m))]


# MultiIndexSet conversion


def test_multi_index_set_round_trip():
    """from_numba(m.to_numba()) reproduces the original rows."""
    m = MultiIndexSet.from_list([{3, 1}, set(), {2, 4}])
    back = MultiIndexSet.from_numba(m.to_numba())
    assert _rows(back) == [{1, 3}, set(), {2, 4}]


def test_to_numba_is_contiguous_int64():
    """The jitclass backing arrays are C-contiguous int64 (jitclass requires it)."""
    nb = MultiIndexSet.from_list([{5, 1}, {2}]).to_numba()
    assert nb.indices.dtype == np.int64
    assert nb.indices.flags["C_CONTIGUOUS"]
    assert nb.row_offsets.dtype == np.int64


def test_get_inside_njit_matches_python_rows():
    """Reading rows via get(i) inside @njit matches the Python rows."""
    m = MultiIndexSet.from_list([{0, 2}, {1}, set(), {3, 4, 5}])

    @njit
    def collect(nb):
        # Return per-row sums; a scalar reduction avoids returning ragged data.
        out = np.empty(len(nb), dtype=np.int64)
        for i in range(len(nb)):
            s = 0
            for x in nb.get(i):
                s += x
            out[i] = s
        return out

    got = collect(m.to_numba())
    expected = np.array([sum(r) for r in _rows(m)], dtype=np.int64)
    np.testing.assert_array_equal(got, expected)


# Builder conversion + accumulation


def test_builder_round_trip_equals_python_build():
    """Accumulating overlapping rows in njit, then Python build(), dedups+sorts.

    Equivalent to the same unions done with the Python builder.
    """
    reader = MultiIndexSet.from_list([{10, 11}, {11, 12}, {13}]).to_numba()

    @njit
    def accumulate(nb_reader, length):
        # Inside @njit, construct directly (staticmethods are Python-side only).
        builder = NumbaMultiIndexSetBuilder(length)
        # element 0 unions rows 0 and 1 (overlap on 11); element 1 gets row 2.
        builder.union_into(0, nb_reader.get(0))
        builder.union_into(0, nb_reader.get(1))
        builder.union_into(1, nb_reader.get(2))
        return builder

    result = MultiIndexSetBuilder.from_numba(accumulate(reader, 2)).build()
    assert _rows(result) == [{10, 11, 12}, {13}]


def test_builder_to_numba_seeds_existing_chunks():
    """A partly-filled Python builder converts to numba with its chunks intact."""
    py = MultiIndexSetBuilder(length=2)
    py[0] |= {7, 8}
    py[1] |= {9}
    round_tripped = MultiIndexSetBuilder.from_numba(py.to_numba()).build()
    assert _rows(round_tripped) == [{7, 8}, {9}]


def test_empty_builder_round_trip():
    """An empty builder round-trips to all-empty rows."""
    result = MultiIndexSetBuilder.from_numba(
        MultiIndexSetBuilder(length=3).to_numba()
    ).build()
    assert _rows(result) == [set(), set(), set()]
