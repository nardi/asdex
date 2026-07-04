"""Types, constants, and utilities for dependency tracking."""

import itertools
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any, cast, overload
from typing import Self as Self

import numpy as np
from jax._src.core import Jaxpr, JaxprEqn, Literal, Var

IndexSet = set[int]
"""A single per-element dependency set.

Backed by Python's built-in set.
Benchmarked against pyroaring.BitMap and int bitmasks;
set[int] wins for the typical workload (small sparse sets, large universe).
"""


def _empty_index_set() -> IndexSet:
    """Create an empty dependency set."""
    return set()


def _singleton_index_set(i: int) -> IndexSet:
    """Create a dependency set containing a single index."""
    return {i}


IndexArray = np.ndarray  # 1-D np.ndarray of flat dependency indices


class IndexSetView(AbstractSet[int]):
    """Read-only, set-like view over one element's dependency indices.

    Wraps a slice of a :class:`MultiIndexSet`'s backing array without
    copying, so handlers can read a single element's dependencies and
    combine them with set algebra (``|``, ``&``, ``in``, iteration)
    without materializing a ``set``.

    The view is immutable; callers must never try to mutate it.
    Use :meth:`copy` to obtain a fresh, mutable ``set`` when in-place
    mutation is needed.
    """

    __slots__ = ("_arr",)

    def __init__(self, arr: IndexArray) -> None:
        self._arr = arr

    def __iter__(self) -> Iterator[int]:
        # Iterating a Python list of the (small) row is faster than boxing
        # numpy scalars one at a time, and yields plain ``int``.
        return iter(self._arr.tolist())

    def __len__(self) -> int:
        return int(self._arr.size)

    def __contains__(self, value: object) -> bool:
        return value in self._arr

    @classmethod
    def _from_iterable(cls, it: Iterable[int]) -> set[int]:
        # The AbstractSet mixin operators (|, &, -, ^) build their results
        # through this hook; return a plain, mutable set rather than a view.
        return set(it)

    def copy(self) -> set[int]:
        """Return a fresh, mutable ``set`` of these dependency indices."""
        return set(self._arr.tolist())


@dataclass(slots=True, kw_only=True, frozen=True, eq=False)
class MultiIndexSet(Sequence[IndexSetView]):
    """Per-element dependency sets stored in a compact CSR-like layout.

    ``_indices`` is a flat array holding every dependency index,
    grouped by element, and ``_row_offsets[i] : _row_offsets[i + 1]``
    is the slice of ``_indices`` belonging to element ``i``.

    Building this in bulk (via :class:`MultiIndexSetBuilder` or the
    ``from_*`` constructors) is far cheaper than a ``list[set[int]]``,
    which is why detection stores index sets this way.
    Reading a single element (``mis[i]``) returns a read-only
    :class:`IndexSetView` over that row so handlers can use ordinary set
    algebra without materializing a ``set``.
    """

    indices: IndexArray
    _row_offsets: IndexArray

    @classmethod
    def from_labeled_indices(cls, length: int, labeled_indices: np.ndarray) -> Self:
        """Build from a ``(2, k)`` array of ``(element, dependency)`` columns.

        Columns may appear in any order and ``(element, dependency)`` pairs
        may repeat; duplicates are dropped so each element's stored row is a
        genuine set, sorted and contiguous in ``_indices``.
        """
        labels = np.asarray(labeled_indices[0], dtype=np.int_)
        deps = np.asarray(labeled_indices[1], dtype=np.int_)

        # Deduplicate and lexicographically sort the (element, dependency)
        # pairs. This groups dependencies by element (so each row is a
        # contiguous slice), sorts within each row, and collapses the
        # duplicate pairs that accumulate through builder unions — without
        # which repeated dependencies would leak into the COO output.
        unique = np.unique(np.stack([labels, deps]), axis=1)
        unique_labels = unique[0]
        indices = unique[1]

        # Row offsets are the running total of per-element dependency counts.
        counts = np.bincount(unique_labels, minlength=length)
        row_offsets = np.zeros(length + 1, dtype=np.int_)
        np.cumsum(counts, out=row_offsets[1:])

        return cls(indices=indices, _row_offsets=row_offsets)

    @classmethod
    def from_list(cls, index_set_list: Sequence[Iterable[int]]) -> Self:
        """Build from a per-element sequence of dependency iterables."""

        def asarray(iter):
            return (
                np.fromiter(iter, int, len(iter))
                if not isinstance(iter, np.ndarray)
                else iter
            )

        labeled_indices = (
            np.concatenate(
                [
                    np.stack(
                        [
                            indices := asarray(s),
                            np.full_like(indices, i),
                        ][::-1],
                        axis=0,
                    )
                    for i, s in enumerate(index_set_list)
                ],
                axis=-1,
            )
            if len(index_set_list)
            else np.empty((2, 0), dtype=np.int_)
        )
        return cls.from_labeled_indices(len(index_set_list), labeled_indices)

    def __len__(self) -> int:
        return self._row_offsets.size - 1

    @overload
    def __getitem__(self, index: int) -> IndexSetView: ...

    @overload
    def __getitem__(self, index: slice | np.ndarray) -> "MultiIndexSet": ...

    def __getitem__(
        self, index: int | slice | np.ndarray
    ) -> "IndexSetView | MultiIndexSet":
        if isinstance(index, slice):
            index = np.arange(*index.indices(len(self)))

        if isinstance(index, np.ndarray):
            # Select a subset of rows into a new MultiIndexSet.
            starts = self._row_offsets[index]
            stops = self._row_offsets[index + 1]
            row_lengths = stops - starts
            row_offsets = np.zeros(row_lengths.size + 1, dtype=np.int_)
            np.cumsum(row_lengths, out=row_offsets[1:])

            # Gather each selected row's contiguous slice into one flat array.
            # For output position p in selected row k, the source index is
            # ``starts[k] + (p - row_offsets[k])``; broadcast the per-row shift
            # ``starts - row_offsets[:-1]`` across each row's positions.
            shift = np.repeat(starts - row_offsets[:-1], row_lengths)
            values = self.indices[np.arange(row_offsets[-1]) + shift]

            return MultiIndexSet(indices=values, _row_offsets=row_offsets)

        start = self._row_offsets[index]
        stop = self._row_offsets[index + 1]
        return IndexSetView(self.indices[start:stop])

    def __add__(self, other: "MultiIndexSet") -> "MultiIndexSet":
        """Row-concatenate two patterns, merging their backing arrays.

        The result has ``len(self) + len(other)`` rows: ``self``'s rows
        followed by ``other``'s. ``other``'s dependency slice is appended
        to ``_indices`` and its offsets are shifted past ``self``'s, so no
        rows are unioned and each stays deduplicated.
        """
        if not isinstance(other, MultiIndexSet):
            return NotImplemented
        indices = np.concatenate([self.indices, other.indices])
        row_offsets = np.concatenate(
            [self._row_offsets, other._row_offsets[1:] + self._row_offsets[-1]]
        )
        return MultiIndexSet(indices=indices, _row_offsets=row_offsets)

    # Numba interop

    def to_numba(self) -> Any:
        """Return an njit-readable :class:`NumbaMultiIndexSet` over the same rows.

        Backing arrays are made C-contiguous ``int64`` (copying only when
        necessary) to satisfy the jitclass field types.
        """
        from ._numba import NumbaMultiIndexSet  # noqa: PLC0415

        return NumbaMultiIndexSet.create(
            np.ascontiguousarray(self.indices, dtype=np.int64),
            np.ascontiguousarray(self._row_offsets, dtype=np.int64),
        )

    @classmethod
    def from_numba(cls, nb: Any) -> Self:
        """Rebuild a :class:`MultiIndexSet` from a :class:`NumbaMultiIndexSet`."""
        return cls(indices=nb.indices, _row_offsets=nb.row_offsets)


@dataclass(slots=True, kw_only=True, frozen=True)
class MultiIndexSetBuilder:
    """Mutable accumulator that builds a :class:`MultiIndexSet` in bulk.

    Supports the union-into-place idiom used by reductions and contractions::

        out = MultiIndexSetBuilder(length=n)
        out[i] |= deps      # record that element i depends on ``deps``
        out[j] = deps       # equivalent; also records dependencies

    Each write appends a batch of ``(element, dependency)`` pairs to
    ``_index_arrays``; :meth:`build` concatenates them into one CSR array.
    Assignment records dependencies rather than replacing them, matching
    the append-only nature of the builder.
    """

    length: int
    _index_arrays: list[IndexArray] = field(default_factory=list)

    @classmethod
    def identity(cls, *, length: int, offset: int = 0) -> Self:
        """Builder where element ``i`` depends on the single index ``i + offset``."""
        return cls(
            length=length,
            _index_arrays=[
                np.stack([np.arange(length), np.arange(length) + offset], axis=0)
            ],
        )

    def __len__(self) -> int:
        return self.length

    def build(self) -> MultiIndexSet:
        labeled_indices = (
            np.concatenate(self._index_arrays, axis=-1)
            if self._index_arrays
            else np.empty((2, 0), dtype=np.int_)
        )
        return MultiIndexSet.from_labeled_indices(self.length, labeled_indices)

    # Numba interop

    def to_numba(self) -> Any:
        """Return an njit-writable :class:`NumbaMultiIndexSetBuilder`.

        Seeds it with the ``(label, dep)`` chunks accumulated so far;
        further accumulation happens inside ``@njit`` code.
        """
        from ._numba import NumbaMultiIndexSetBuilder  # noqa: PLC0415

        builder = NumbaMultiIndexSetBuilder.create(self.length)
        for chunk in self._index_arrays:
            builder.append_chunk(np.ascontiguousarray(chunk, dtype=np.int64))
        return builder

    @classmethod
    def from_numba(cls, nb: Any) -> Self:
        """Rebuild a Python builder from a :class:`NumbaMultiIndexSetBuilder`.

        Call :meth:`build` on the result to dedup/sort into a
        :class:`MultiIndexSet`.
        """
        builder = cls(length=int(nb.length))
        builder._index_arrays.extend(np.asarray(chunk) for chunk in nb.chunks)
        return builder

    def __getitem__(self, index: int | slice) -> "MultiIndexSetBuilderIndexer":
        if isinstance(index, slice):
            raise NotImplementedError("Indexing with slice not supported")

        return MultiIndexSetBuilderIndexer(
            _index_arrays=self._index_arrays, _index=index
        )

    def __setitem__(self, index: int | np.ndarray, value: Any) -> None:
        # `builder[array] = mis`: assign a whole batch of elements at once,
        # equivalent to `for k: builder[array[k]] = mis[k]` but in one append.
        if isinstance(index, np.ndarray):
            self._setitem_batch(index, value)
            return

        # `builder[i] |= deps` desugars to
        # `builder[i] = builder[i].__ior__(deps)`. The indexer has already
        # appended the pairs, so writing the same indexer back is a no-op.
        if (
            isinstance(value, MultiIndexSetBuilderIndexer)
            and value._index == index
            and value._index_arrays is self._index_arrays
        ):
            return

        # Plain `builder[i] = deps`: record the dependencies for element i.
        self[index].__ior__(cast(Iterable[int], value))

    def _setitem_batch(self, index: np.ndarray, value: Any) -> None:
        """Assign ``value[k]``'s dependencies to element ``index[k]`` for every k.

        ``value`` is a :class:`MultiIndexSet` (or any per-element sequence,
        converted via :meth:`MultiIndexSet.from_list`) whose length must match
        ``index``. Records one labeled chunk covering all rows: since
        ``_indices`` is grouped by row in order, repeating each target element
        by its row length lines every dependency up with its destination.
        """
        mis = (
            value
            if isinstance(value, MultiIndexSet)
            else MultiIndexSet.from_list(value)
        )
        if len(mis) != index.size:
            raise ValueError(
                f"Cannot assign {len(mis)} index sets to {index.size} elements; "
                "the MultiIndexSet length must match the index array."
            )
        row_lengths = np.diff(mis._row_offsets)
        labels = np.repeat(index, row_lengths)
        self._index_arrays.append(np.stack([labels, mis.indices]))


@dataclass(slots=True, kw_only=True, frozen=True)
class MultiIndexSetBuilderIndexer:
    """Handle for a single element of a :class:`MultiIndexSetBuilder`.

    Only supports ``|=`` (append this element's dependencies).
    """

    _index_arrays: list[IndexArray]
    _index: int

    def __ior__(self, it: Iterable[int]) -> Self:
        indices = (
            np.fromiter(it, dtype=np.int_) if not isinstance(it, np.ndarray) else it
        )
        labeled_indices = np.stack(
            [np.full_like(indices, self._index), indices], axis=0
        )
        self._index_arrays.append(labeled_indices)
        return self


def _empty_index_sets(n: int) -> MultiIndexSetBuilder:
    """Create a builder for n empty dependency sets."""
    return MultiIndexSetBuilder(length=n)


def _identity_index_sets(n: int) -> MultiIndexSetBuilder:
    """Create an index set builder where element i depends on index i."""
    return MultiIndexSetBuilder.identity(length=n)


class StateIndices(dict[Var, MultiIndexSet]):
    """Maps each variable to its per-element dependency index sets.

    Accepts a :class:`MultiIndexSet`, a :class:`MultiIndexSetBuilder`
    (built on assignment), or a plain per-element sequence of dependency
    sets (converted via :meth:`MultiIndexSet.from_list`), so handlers can
    produce whichever form is most convenient.
    """

    def __setitem__(
        self,
        key: Var,
        value: MultiIndexSet | MultiIndexSetBuilder | Sequence[AbstractSet[int]],
    ) -> None:
        if isinstance(value, MultiIndexSetBuilder):
            value = value.build()
        elif not isinstance(value, MultiIndexSet):
            value = MultiIndexSet.from_list(value)

        super().__setitem__(key, value)


StateConsts = dict[Var, np.ndarray]
"""Maps variables to their concrete numpy array values (for static index tracking)."""

StateBounds = dict[Var, tuple[np.ndarray, np.ndarray]]
"""Maps variables to per-element inclusive (lo, hi) integer bounds.

Used to track bounded-but-not-constant values
(e.g. output of ``argmax`` over a small axis)
so that dynamic index handlers can enumerate all possible values
instead of falling back to conservative.
"""

Atom = Var | Literal
"""Atomic elements in jaxpressions: named intermediates (Var) or constants (Literal)."""

PropJaxprFn = Callable[
    [Jaxpr, Sequence[Sequence[AbstractSet[int]]], StateConsts | None],
    list[MultiIndexSet],
]
"""Signature of ``_prop_jaxpr``, passed as callback to break circular imports.

Inputs are per-variable index sets: a :class:`MultiIndexSet` or any plain
per-element sequence of sets. Outputs are always :class:`MultiIndexSet`.
"""


_MAX_ENUM_COMBINATIONS = 64
"""Maximum number of index combinations to enumerate for bounded dynamic indices.

When ``gather``, ``scatter``, ``dynamic_slice``, or ``dynamic_update_slice``
receive indices that are not statically known but have bounded value ranges
(e.g. from ``argmax`` over a small axis),
we enumerate all possible index arrays and union the resulting sparsity patterns.
This yields a tighter pattern than the conservative all-to-all fallback.

The cap prevents combinatorial blowup for multi-element index arrays:
an index with *k* elements where each has *r* possible values
gives *r^k* combinations.
If this exceeds the cap, the handler falls back to conservative.

The value 64 is chosen to keep enumeration fast
while covering the common cases
(e.g. one ``argmax`` index with up to 64 possible values,
or two indices each with up to 8 possible values).
"""


def _enumerate_bounded_patterns(
    ranges: Sequence[range],
    out_size: int,
    make_pattern: Callable[[tuple[int, ...]], Sequence[AbstractSet[int]] | None],
) -> list[AbstractSet[int]] | None:
    """Enumerate all candidate index combinations and union the resulting patterns.

    Used by ``gather``, ``scatter``, ``dynamic_slice``, and ``dynamic_update_slice``
    when indices are bounded but not statically known.
    Each call site builds its own ``ranges`` (from ``_atom_value_bounds``
    or ``_resolve_start_bounds``) and provides a ``make_pattern`` callback
    that computes the sparsity pattern for one concrete index combination.

    Returns ``None`` if the total number of combinations exceeds
    ``_MAX_ENUM_COMBINATIONS`` or if ``make_pattern`` returns ``None``
    (indicating an unrecognized pattern, as in scatter).
    """
    if math.prod(len(r) for r in ranges) > _MAX_ENUM_COMBINATIONS:
        return None

    accumulated: list[AbstractSet[int]] | None = None
    for candidate_values in itertools.product(*ranges):
        pattern = make_pattern(candidate_values)
        if pattern is None:
            return None
        if accumulated is None:
            accumulated = list(pattern)
        else:
            for i in range(out_size):
                accumulated[i] = accumulated[i] | pattern[i]

    return accumulated


# Shape and size


def _numel(shape: Sequence[int]) -> int:
    """Compute the total number of elements from a shape tuple."""
    return math.prod(shape) if shape else 1


def _atom_shape(atom: Atom) -> tuple[int, ...]:
    """Get the shape of a variable or literal."""
    if isinstance(atom, Literal):
        return tuple(getattr(atom.val, "shape", ()))
    return tuple(getattr(atom.aval, "shape", ()))


def _atom_numel(atom: Atom) -> int:
    """Get the total number of elements in a variable or literal."""
    if isinstance(atom, Literal):
        shape = getattr(atom.val, "shape", ())
        return _numel(tuple(shape)) if shape else 1
    shape = getattr(atom.aval, "shape", ())
    return _numel(tuple(shape)) if shape else 1


# Atom value access


def _index_sets(state_indices: StateIndices, atom: Atom) -> MultiIndexSet:
    """Get the index sets for a variable or literal."""
    if isinstance(atom, Literal):
        return _empty_index_sets(_atom_numel(atom)).build()
    return state_indices.get(atom, MultiIndexSetBuilder(length=1).build())


def _copy_index_sets(src: Sequence[AbstractSet[int]]) -> list[IndexSet]:
    """Copy per-element index sets into a fresh, mutable ``list[set]``."""
    return [set(s) for s in src]


def _atom_const_val(atom: Atom, state_consts: StateConsts) -> np.ndarray | None:
    """Get the concrete value of an atom, if statically known.

    The value is known in two cases:
    - **Literals**: constants embedded directly in the jaxpr.
    - **Tracked vars**: variables in ``state_consts``, whose values were
      computed from constants through earlier operations.

    Returns ``None`` when the value depends on runtime inputs.
    """
    if isinstance(atom, Literal):
        return np.asarray(atom.val)
    if isinstance(atom, Var) and atom in state_consts:
        return state_consts[atom]
    return None


def _atom_value_bounds(
    atom: Atom,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Get per-element inclusive (lo, hi) bounds for an atom.

    Returns exact ``(val, val)`` for constants,
    tracked bounds for bounded variables,
    or ``None`` when no information is available.
    """
    val = _atom_const_val(atom, state_consts)
    if val is not None:
        return (val, val)
    if isinstance(atom, Var) and atom in state_bounds:
        return state_bounds[atom]
    return None


def _propagate_const_unary(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    transform: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a unary op.

    If the input is statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in_val = _atom_const_val(eqn.invars[0], state_consts)
    if in_val is not None:
        state_consts[eqn.outvars[0]] = transform(in_val)


def _propagate_const_binary(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a binary op.

    If both inputs are statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in1 = _atom_const_val(eqn.invars[0], state_consts)
    in2 = _atom_const_val(eqn.invars[1], state_consts)
    if in1 is not None and in2 is not None:
        state_consts[eqn.outvars[0]] = transform(in1, in2)


# Zero-skipping


def _broadcast_to_output(
    val: np.ndarray, in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> np.ndarray:
    """Broadcast a const value from input shape to output shape, returning a flat array.

    Handles numpy-style broadcasting: left-pads with 1s then expands.
    """
    ndim = len(out_shape)
    arr = np.asarray(val).reshape(in_shape) if in_shape else np.asarray(val)
    pad = ndim - len(in_shape)
    padded_shape = (1,) * pad + in_shape
    return np.broadcast_to(arr.reshape(padded_shape), out_shape).ravel()


def _clear_where_zero(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    invar_idx: int,
) -> None:
    """Clear output index sets at positions where an input is a known constant zero.

    Used by ``mul``, ``div``, and ``integer_pow`` for zero-skipping:
    ``d(0 * y)/dy = 0``, ``d(0 / y)/dy = 0``, ``d(0^n)/dx = 0`` for ``n > 1``.
    """
    val = _atom_const_val(eqn.invars[invar_idx], state_consts)
    if val is None:
        return
    out_shape = _atom_shape(eqn.outvars[0])
    in_shape = _atom_shape(eqn.invars[invar_idx])
    flat = _broadcast_to_output(val, in_shape, out_shape)

    # Rebuild the output, dropping dependencies where the input is a known zero.
    out_indices = state_indices[eqn.outvars[0]]
    state_indices[eqn.outvars[0]] = [
        _empty_index_set() if flat[i] == 0 else out_indices[i]
        for i in range(len(out_indices))
    ]


# Index set operations


def _union_all(sets: Sequence[AbstractSet[int]]) -> IndexSet:
    """Union all sets together, returning a new set."""
    result: IndexSet = _empty_index_set()
    for s in sets:
        # ``update`` (not ``|=``) so set-like views are accepted as operands.
        result.update(s)
    return result


def _union_elementwise(
    inputs: Sequence[Sequence[AbstractSet[int]]], out_size: int
) -> list[IndexSet]:
    """Union multiple index set lists element-wise with scalar broadcasting.

    Each input list represents per-element index sets for one operand.
    Scalars (length 1) broadcast to match the output size via modular indexing.

    TODO: use in more places (e.g. _binary_elementwise, select_n).
    """
    return [_union_all([inp[i % len(inp)] for inp in inputs]) for i in range(out_size)]


def _check_no_index_sets(
    state_indices: StateIndices, atom: Atom, primitive_name: str
) -> None:
    """Verify that an atom carries no input dependencies.

    Some handlers assume that auxiliary inputs
    (index arrays, kernel weights, selectors)
    are constants with empty dependency sets.
    This function validates that assumption
    and raises an informative error when it is violated.
    """
    if any(_index_sets(state_indices, atom)):
        msg = (
            f"'{primitive_name}' handler assumes an auxiliary input "
            "has no dependency on the function's inputs, "
            "but found non-empty index sets. "
            "Please help out asdex's development by reporting this at https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)


def _conservative_indices(
    all_indices: Sequence[AbstractSet[int]], out_size: int
) -> list[IndexSet]:
    """Build conservative output index sets where every element depends on the union of all inputs."""
    combined = _union_all(all_indices)
    return [combined] * out_size


# Index clamping


def _clamp_starts(
    starts: tuple[int, ...], in_shape: Sequence[int], slice_sizes: Sequence[int]
) -> tuple[int, ...]:
    """Clamp start indices to valid bounds.

    Matches JAX's ``dynamic_slice`` and ``gather`` semantics,
    which silently clamp out-of-bounds starts
    rather than raising an error.
    """
    return tuple(
        max(0, min(s, dim - sz))
        for s, dim, sz in zip(starts, in_shape, slice_sizes, strict=True)
    )


# Position maps


def _position_map(shape: Sequence[int]) -> np.ndarray:
    """Build an array where each element holds its own flat position.

    For shape ``(2, 3)``, returns ``[[0, 1, 2], [3, 4, 5]]``.
    Applying operations (transpose, slice, etc.) to this array
    reveals which input position each output position reads from.
    """
    return np.arange(_numel(shape)).reshape(shape)


def _permute_indices(
    in_indices: Sequence[IndexSetView], flat_map: Sequence[int] | np.ndarray
) -> list[IndexSetView]:
    """Build output index sets by looking up input positions from a flat map.

    Each output element copies its index set from ``in_indices[flat_map[i]]``.
    Used by handlers that already have a precomputed flat integer map
    (broadcast, tile, gather).
    """
    return [in_indices[j] for j in flat_map]


def _transform_indices(
    in_indices: Sequence[IndexSetView],
    in_shape: Sequence[int],
    transform: Callable[[np.ndarray], np.ndarray] = lambda p: p,
) -> list[IndexSetView]:
    """Build output index sets by transforming a position map.

    Creates a position map for ``in_shape``
    (an array where element ``i`` holds value ``i``),
    applies ``transform``,
    and uses the result to look up index sets from ``in_indices``.

    Each output element copies its index set from the input position
    determined by the transformed position map.
    This is the common pattern for permutation-like ops
    (transpose, rev, slice, reshape, split, dynamic_slice)
    where each output reads exactly one input element.
    """
    flat_map = transform(_position_map(in_shape)).ravel()
    return _permute_indices(in_indices, flat_map)


# Coordinate helpers


def _row_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Compute row-major strides for multi-dimensional index tracking.

    Used to convert between flat indices and coordinates when propagating
    dependencies through slice and broadcast_in_dim.
    Each stride tells how many flat elements to skip
    when incrementing one coordinate position.

    For shape (2, 3, 4): _row_strides = (12, 4, 1) since moving one step in dim 0
    skips 3*4=12 elements, dim 1 skips 4 elements, and dim 2 skips 1 element.
    """
    result: list[int] = []
    stride = 1
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


def _flat_to_coords(flat: int, strides: tuple[int, ...]) -> list[int]:
    """Convert a flat index to multi-dimensional coordinates using row-major strides."""
    coord = []
    remaining = flat
    for s in strides:
        coord.append(remaining // s)
        remaining %= s
    return coord


# Const value propagation


def _seed_const_vals(state_consts: StateConsts, constvars, consts) -> None:
    """Populate state_consts for the captured constants of a ClosedJaxpr.

    Without this, gather/scatter inside nested jaxprs (cond branches,
    while bodies, jit-wrapped calls) cannot resolve closure-captured
    index arrays and fall back to conservative.
    """
    for var, val in zip(constvars, consts, strict=True):
        state_consts[var] = np.asarray(val)


def _forward_value_bounds(
    state_bounds: StateBounds, outer_atoms: Sequence[Atom], inner_vars
) -> None:
    """Transfer known value bounds from outer-scope atoms to inner jaxpr variables.

    Same idea as ``_forward_const_vals`` but for value bounds.
    """
    for outer, inner in zip(outer_atoms, inner_vars, strict=False):
        if isinstance(outer, Var) and outer in state_bounds:
            state_bounds[inner] = state_bounds[outer]


def _forward_const_vals(
    state_consts: StateConsts, outer_atoms: Sequence[Atom], inner_vars
) -> None:
    """Transfer known state_consts from outer-scope atoms to inner jaxpr variables.

    When entering a nested jaxpr (cond branch, while body, jit call),
    the outer equation's invars and the inner jaxpr's invars are different
    ``Var`` objects representing the same values.
    This copies any concrete values from the outer atoms
    to the corresponding inner vars so that downstream handlers
    (gather, scatter, dynamic_slice) can resolve indices precisely.
    """
    for outer, inner in zip(outer_atoms, inner_vars, strict=False):
        val = _atom_const_val(outer, state_consts)
        if val is not None:
            state_consts[inner] = val
