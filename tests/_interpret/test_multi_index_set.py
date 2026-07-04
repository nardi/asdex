"""Unit tests for the compact index-set containers in ``_common``.

Covers :class:`MultiIndexSet` (the CSR-like stored form), its read-only
:class:`IndexSetView` elements, and the :class:`MultiIndexSetBuilder`
used by handlers to accumulate dependencies in bulk.
"""

import jax
import numpy as np
import pytest
from jax._src.core import Var

from asdex.detection._interpret._common import (
    IndexSetView,
    MultiIndexSet,
    MultiIndexSetBuilder,
    StateIndices,
    _empty_index_sets,
    _identity_index_sets,
)


def _rows(m: MultiIndexSet) -> list[set[int]]:
    """Materialize a MultiIndexSet as a plain list of sets for comparison."""
    return [set(m[i]) for i in range(len(m))]


# MultiIndexSet construction


def test_from_list_basic():
    """from_list groups dependencies per element and sorts within each row."""
    m = MultiIndexSet.from_list([{3, 1}, {2}, set()])
    assert len(m) == 3
    assert _rows(m) == [{1, 3}, {2}, set()]


def test_from_list_empty_sequence():
    """An empty sequence builds a length-0 MultiIndexSet."""
    m = MultiIndexSet.from_list([])
    assert len(m) == 0
    assert _rows(m) == []


def test_from_list_all_empty_sets():
    """All-empty rows do not crash (regression: empty labeled-index array)."""
    m = MultiIndexSet.from_list([set(), set(), set()])
    assert len(m) == 3
    assert _rows(m) == [set(), set(), set()]


def test_from_labeled_indices_deduplicates():
    """Repeated (element, dependency) columns collapse to a single entry.

    This is the accumulation pattern produced by builder unions; duplicates
    must not leak through to the COO output (they would double-count).
    """
    labeled = np.array([[0, 0, 1, 0], [5, 6, 7, 5]])  # row 0: {5,6,5}, row 1: {7}
    m = MultiIndexSet.from_labeled_indices(2, labeled)
    assert _rows(m) == [{5, 6}, {7}]
    # No duplicate 5 remains in the backing array.
    assert m[0] == {5, 6}
    assert len(list(m[0])) == 2


def test_from_labeled_indices_out_of_order():
    """Columns in arbitrary order are grouped correctly by element."""
    labeled = np.array([[2, 0, 1, 0], [9, 3, 4, 1]])
    m = MultiIndexSet.from_labeled_indices(3, labeled)
    assert _rows(m) == [{1, 3}, {4}, {9}]


def test_from_labeled_indices_empty():
    """A (2, 0) labeled array builds all-empty rows."""
    m = MultiIndexSet.from_labeled_indices(2, np.empty((2, 0), dtype=np.int_))
    assert _rows(m) == [set(), set()]


def test_getitem_slice_returns_sub_multi_index_set():
    """Slicing selects rows into a new MultiIndexSet."""
    m = MultiIndexSet.from_list([{0}, {1, 5}, {2}, {3}])
    sliced = m[1:3]
    assert isinstance(sliced, MultiIndexSet)
    assert _rows(sliced) == [{1, 5}, {2}]
    assert _rows(m[::2]) == [{0}, {2}]


def test_getitem_fancy_index_selects_rows():
    """An integer array selects (and reorders/repeats) rows into a MultiIndexSet."""
    m = MultiIndexSet.from_list([{0, 1}, set(), {2, 3, 4}, {5}])
    picked = m[np.array([2, 0, 2])]
    assert isinstance(picked, MultiIndexSet)
    assert _rows(picked) == [{2, 3, 4}, {0, 1}, {2, 3, 4}]
    # Empty selection yields a length-0 MultiIndexSet.
    assert _rows(m[np.array([], dtype=np.int_)]) == []


def test_getitem_returns_view():
    """Reading an element returns a read-only IndexSetView (no materialization)."""
    m = MultiIndexSet.from_list([{0, 1}, {2}])
    assert isinstance(m[0], IndexSetView)


def test_iteration_yields_each_row():
    """Iterating a MultiIndexSet yields one set-like view per element."""
    m = MultiIndexSet.from_list([{0, 1}, {2}])
    assert [set(s) for s in m] == [{0, 1}, {2}]


# IndexSetView


def test_view_is_set_like():
    """A row view supports len, membership, and iteration without duplicates."""
    view = IndexSetView(np.array([2, 5, 7]))
    assert len(view) == 3
    assert 5 in view
    assert 4 not in view
    assert set(view) == {2, 5, 7}


def test_view_union_returns_plain_set():
    """Set algebra on views produces a plain, mutable set."""
    a = IndexSetView(np.array([1, 2]))
    b = IndexSetView(np.array([2, 3]))
    union = a | b
    assert union == {1, 2, 3}
    assert isinstance(union, set)
    # Union with a plain set works from either side.
    assert ({0} | a) == {0, 1, 2}
    assert (a | {0}) == {0, 1, 2}


def test_view_copy_is_independent_mutable_set():
    """copy() yields a fresh set that can be mutated without touching the view."""
    view = IndexSetView(np.array([1, 2]))
    c = view.copy()
    assert c == {1, 2}
    c.add(9)
    assert set(view) == {1, 2}


def test_view_update_into_plain_set():
    """A plain set can be updated from a view (used across handlers)."""
    acc: set[int] = set()
    acc.update(IndexSetView(np.array([4, 5])))
    assert acc == {4, 5}


# MultiIndexSetBuilder


def test_builder_empty_build():
    """An untouched builder builds all-empty rows."""
    m = _empty_index_sets(3).build()
    assert _rows(m) == [set(), set(), set()]


def test_identity_builder():
    """identity() maps element i to the single index i (+ offset)."""
    assert _rows(_identity_index_sets(3).build()) == [{0}, {1}, {2}]
    assert _rows(MultiIndexSetBuilder.identity(length=2, offset=5).build()) == [
        {5},
        {6},
    ]


def test_builder_ior_accumulates_and_dedups():
    """``builder[i] |= deps`` accumulates; overlapping unions dedupe on build.

    This is the reduction / contraction pattern (see ``_reduce``, ``_dot_general``).
    """
    b = MultiIndexSetBuilder(length=2)
    b[0] |= {1, 2}
    b[0] |= {2, 3}  # overlaps with the previous union
    b[1] |= [4]
    assert _rows(b.build()) == [{1, 2, 3}, {4}]


def test_builder_setitem_records_dependencies():
    """Plain ``builder[i] = deps`` records the dependencies for element i."""
    b = MultiIndexSetBuilder(length=2)
    b[0] = {7, 8}
    b[1] = {9}
    assert _rows(b.build()) == [{7, 8}, {9}]


def test_builder_ior_accepts_view():
    """A builder accepts a set-like view as the union operand."""
    b = MultiIndexSetBuilder(length=1)
    b[0] |= IndexSetView(np.array([3, 4]))
    assert _rows(b.build()) == [{3, 4}]


def test_builder_array_assign():
    """``builder[array] = mis`` assigns mis[k] to element array[k]."""
    b = MultiIndexSetBuilder(length=4)
    b[np.array([2, 0])] = MultiIndexSet.from_list([{5}, {6, 7}])
    assert _rows(b.build()) == [{6, 7}, set(), {5}, set()]


def test_builder_array_assign_matches_scalar_loop():
    """The batch assign equals the explicit per-element loop."""
    targets = np.array([3, 1, 0])
    mis = MultiIndexSet.from_list([{10, 11}, {12}, set()])

    batch = MultiIndexSetBuilder(length=4)
    batch[targets] = mis

    loop = MultiIndexSetBuilder(length=4)
    for k, t in enumerate(targets):
        loop[int(t)] = mis[k]

    assert _rows(batch.build()) == _rows(loop.build())


def test_builder_array_assign_duplicate_targets_union():
    """Repeated targets in the index array union their assigned sets."""
    b = MultiIndexSetBuilder(length=2)
    b[np.array([0, 0, 1])] = MultiIndexSet.from_list([{1}, {2, 3}, {4}])
    assert _rows(b.build()) == [{1, 2, 3}, {4}]


def test_builder_array_assign_interops_with_scalar():
    """Batch assignment and scalar ``|=`` accumulate into the same builder."""
    b = MultiIndexSetBuilder(length=3)
    b[np.array([0, 2])] = MultiIndexSet.from_list([{1}, {5}])
    b[0] |= {9}
    assert _rows(b.build()) == [{1, 9}, set(), {5}]


def test_builder_array_assign_empty():
    """An empty index array is a no-op."""
    b = MultiIndexSetBuilder(length=2)
    b[np.array([], dtype=np.int_)] = MultiIndexSet.from_list([])
    assert _rows(b.build()) == [set(), set()]


def test_builder_array_assign_length_mismatch_raises():
    """Assigning a MultiIndexSet whose length differs from the index array errors."""
    b = MultiIndexSetBuilder(length=3)
    with pytest.raises(ValueError, match="length must match"):
        b[np.array([0, 1])] = MultiIndexSet.from_list([{1}])


def test_builder_len():
    """A builder reports its declared length."""
    assert len(MultiIndexSetBuilder(length=5)) == 5


# StateIndices


def _var() -> Var:
    """A fresh jaxpr variable to use as a StateIndices key."""
    return jax.make_jaxpr(lambda x: x)(np.zeros(1)).jaxpr.invars[0]


def test_state_indices_accepts_multi_index_set():
    """A MultiIndexSet is stored as-is."""
    state = StateIndices()
    v = _var()
    m = MultiIndexSet.from_list([{0}, {1}])
    state[v] = m
    assert state[v] is m


def test_state_indices_builds_builder():
    """A builder is built into a MultiIndexSet on assignment."""
    state = StateIndices()
    v = _var()
    state[v] = _identity_index_sets(2)
    assert isinstance(state[v], MultiIndexSet)
    assert _rows(state[v]) == [{0}, {1}]


def test_state_indices_converts_list():
    """A plain list of sets is converted via from_list."""
    state = StateIndices()
    v = _var()
    state[v] = [{0, 1}, set(), {2}]
    assert isinstance(state[v], MultiIndexSet)
    assert _rows(state[v]) == [{0, 1}, set(), {2}]


# Concatenation


def test_add_merges_into_new_multi_index_set():
    """``a + b`` row-concatenates two patterns into a merged MultiIndexSet."""
    a = MultiIndexSet.from_list([{0}, {1, 2}])
    b = MultiIndexSet.from_list([{3}, set(), {4}])
    merged = a + b
    assert isinstance(merged, MultiIndexSet)
    assert _rows(merged) == [{0}, {1, 2}, {3}, set(), {4}]


def test_add_empty_operands():
    """Merging with an empty-row pattern preserves the other's rows."""
    a = MultiIndexSet.from_list([{5}])
    empty = MultiIndexSet.from_list([])
    assert _rows(a + empty) == [{5}]
    assert _rows(empty + a) == [{5}]


def test_slice_indexing_on_builder_raises():
    """Builders do not support slice indexing."""
    with pytest.raises(NotImplementedError):
        _ = MultiIndexSetBuilder(length=3)[0:2]
