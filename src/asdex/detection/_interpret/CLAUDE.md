# _interpret — Custom Jaxpr Interpreter for Index Set Propagation

Implements a custom jaxpr interpreter that propagates per-element dependency index sets (`set[int]`)
through primitives to determine Jacobian sparsity patterns.

## Structure

- `__init__.py` — `_prop_jaxpr`, `_prop_dispatch`, fallback handling.
- `_common.py` — shared types (`IndexSet`, `StateIndices`, `StateConsts`) and utilities.
- Each JAX primitive has its own module: `_foo.py` contains `_prop_foo`.
  Includes `_cumsum.py` for cumulative sum.
- Handlers for external packages (Equinox, Flax, etc.) live in their own subfolders
  (e.g., `_equinox/`).

## Key Types

- `IndexSet` = `set[int]` — a single per-element dependency set
- `MultiIndexSet` — per-element dependency sets for one array, stored in a
  compact CSR-like layout (a flat index array plus row offsets).
  This is the stored form; it is much cheaper to build in bulk than a
  `list[set[int]]`. Reading one element (`mis[i]`) returns a read-only
  `IndexSetView` (set-like, no copy); slicing/iterating yields those views.
  It is immutable — never mutate an element in place.
- `MultiIndexSetBuilder` — mutable accumulator for building a `MultiIndexSet`.
  Supports `builder[i] |= deps` (and `builder[i] = deps`) to record
  dependencies per element; call `.build()` to freeze into a `MultiIndexSet`.
  Handlers that aggregate (reduce, dot_general, sort) build with this.
- `StateIndices` = `dict[Var, MultiIndexSet]` — maps jaxpr variables to their
  index sets. Assigning a `MultiIndexSetBuilder`, a `MultiIndexSet`, or a plain
  `list[IndexSet]` all work; the value is normalized to a `MultiIndexSet`.
- `StateConsts` = `dict[Var, np.ndarray]` — statically-known values for precise gather/scatter
- `StateBounds` = `dict[Var, tuple[np.ndarray, np.ndarray]]` — per-element inclusive (lo, hi) integer bounds

## Naming Conventions

**Terminology** — "indices" and "map" mean different things:
- **"indices" / "index sets"**: the per-element dependency sets used for
  sparsity tracking (a `MultiIndexSet`, or a plain `list[IndexSet]` while
  a handler is building one).
- **"map"**: numpy integer arrays that map output positions to input positions.
  Not index sets.

**Construction** — always use the factory helpers from `_common`:
- `_empty_index_set()` instead of `set()`
- `_singleton_index_set(i)` instead of `{i}`
- `_empty_index_sets(n)` — a `MultiIndexSetBuilder` of `n` empty sets
- `_identity_index_sets(n)` — a `MultiIndexSetBuilder` where element `i` depends on `i`

This ensures a future backend swap only requires changing the helpers,
not every handler.

**Variable names** — use these consistently across handlers:
- `in_indices`: input index sets (from `_index_sets(state_indices, atom)`)
- `in_shape`: input array shape (from `_atom_shape(atom)`)
- `in_val`: const value for a unary input (from `_atom_const_val(atom, state_consts)`)
- `in1_val` / `in2_val`: const values for binary inputs.
  Use descriptive prefixes when roles differ:
  `lhs_val` / `rhs_val` (dot_general), `pred_val` / `which_val` (select), etc.
- `in_bounds` / `in1_bounds` / `in2_bounds`: value bounds for inputs
  (from `_atom_value_bounds(atom, state_consts, state_bounds)`)
- `flat_map`: a flat integer array mapping output positions to input positions

**Docstrings** — avoid the term "deps"; prefer "index sets" or "input index sets".

## Common Utilities in `_common.py`

- **`_position_map(shape)`** —
  builds an array where each element holds its own flat position.
  Applying operations (transpose, slice, flip) to this array
  reveals which input position each output position reads from.
- **`_permute_indices(in_indices, flat_map)`** —
  builds output index sets by looking up ``in_indices[flat_map[i]]``
  for each output position.
  Used by handlers that already have a precomputed flat integer map
  (broadcast, tile, gather).
- **`_transform_indices(in_indices, in_shape, transform)`** —
  builds output index sets by applying ``transform`` to a position map of ``in_shape``.
  The transform function receives an ndarray and returns an ndarray;
  the result is raveled and passed to ``_permute_indices``.
  Used by handlers where each output reads exactly one input element
  (transpose, rev, slice, reshape, split, dynamic_slice).
- **`_propagate_const_unary(eqn, state_consts, transform)`** —
  propagates a const value through a unary op by applying `transform`.
  Mirrors `_propagate_const_binary` for the single-input case.
- **`_enumerate_bounded_patterns(ranges, out_size, make_pattern)`** —
  enumerates all candidate index combinations from ``ranges``
  (capped at ``_MAX_ENUM_COMBINATIONS``),
  calls ``make_pattern`` for each,
  and unions the results element-wise.
- **`_conservative_indices(all_indices, out_size)`** —
  conservative fallback where every output element depends on the union of all inputs.
- **`_atom_value_bounds(atom, state_consts, state_bounds)`** —
  returns `(lo, hi)` bounds for an atom:
  exact `(val, val)` for constants, tracked bounds for bounded variables, or `None`.
- **`_forward_value_bounds(state_bounds, outer_atoms, inner_vars)`** —
  transfers known value bounds from outer-scope atoms to inner jaxpr variables.

## Index Set Aliasing

A `MultiIndexSet` is **immutable**, and reading an element returns a read-only
`IndexSetView` over its backing array (no copy).
Handlers must therefore **never mutate** a set obtained from `state_indices` or `_index_sets()`.

To combine index sets, build new ones:
- `s1 | s2` and `_union_all(...)` return fresh `set[int]`s.
- To accumulate into a plain set, use `acc.update(s)` — **not** `acc |= s`,
  since a view is not a `set` and `|=` requires a `set` operand.
- Call `view.copy()` to get a fresh, mutable `set` when you need to mutate.
- To build per-element output in bulk, use a `MultiIndexSetBuilder`
  (`out[i] |= deps`) and store it; `StateIndices` builds it on assignment.
- `mis_a + mis_b` row-concatenates two `MultiIndexSet`s into a new one
  (merging their backing arrays), e.g. to pool two operands before a
  conservative union.

`_prop_while` copies carry rows into fresh mutable sets before the
fixed-point loop mutates them across iterations.

Read-only reads are typed `IndexSetView`; freshly built/mutable sets are
`set[int]` (aliased `IndexSet`). Use `AbstractSet[int]` for parameters that
accept either. Only genuinely-mutable accumulators should be typed
`list[IndexSet]`.

## Const Value Tracking

Handlers like `broadcast_in_dim`, `select_n`, and `propagate_const_elementwise`
propagate concrete values through `state_consts`.
This lets downstream handlers resolve static indices precisely.

**Invariant**: if a required const value is missing from `state_consts`,
the handler must assume the worst and return a conservative pattern.
This applies to `gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`,
`dot_general` (zero-skipping), and `mul` (zero-clearing).

## Value Bounds Tracking

`StateBounds` tracks per-element inclusive `(lo, hi)` integer bounds
for variables that are bounded but not statically constant
(e.g. the output of `argmax` over a small axis).

Bounds flow through three roles:
**producers** create bounds (`argmax`/`argmin`),
**propagators** forward them (`add`, `sub`, `convert_element_type`, `broadcast_in_dim`, `select_n`),
and **consumers** use them to tighten sparsity
(`gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`, comparisons).

**Invariant**: if bounds are unavailable (`_atom_value_bounds` returns `None`),
the handler must assume the worst and return a conservative pattern.

## Zero-Sized Arrays

Handlers must handle zero-sized arrays (shapes containing a 0 dimension) gracefully.
If the output has zero elements, the handler should return an empty index set list `[]`.
Add an early return before any coordinate-mapping logic
(`np.ravel_multi_index`, `np.indices`, `np.reshape` into the array shape)
that would crash on zero-sized shapes.

## Adding a New Handler

1. Write `_prop_<name>(eqn, state_indices, ...)` in the appropriate module.
2. Add a `case` branch in `_prop_dispatch`.
3. Remove from the fallback `case` group if upgrading from conservative.
4. Add tests in the corresponding `tests/_interpret/test_<module>.py` file.

For primitives from external packages (Equinox, Flax, etc.),
place the handler in a dedicated subfolder (e.g., `_equinox/_select_if_vmap.py`)
with tests in `tests/_interpret/_equinox/`.

## Tests

Each handler module `_foo.py` has a corresponding test file `tests/_interpret/test_foo.py`.

## Writing Style

Use **semantic line breaks** everywhere:
one sentence or clause per line in docstrings, comments, and markdown.
This applies to all prose, not just docstrings.

Focus comments on **why**, not what.
Explain why a branch exists, why a particular approach was chosen, or why a fallback is needed.
Don't narrate what the code already says.

### Handler Docstring Style

1. **Semantic summary**: What the operation does and how dependencies flow.
2. **Math**: The Jacobian structure in concise mathematical notation.
3. **Example**: A concrete input/output trace showing dependency sets before and after.
4. **Jaxpr**: The `eqn.invars` and `eqn.params` layout the handler reads.
5. **URL**: Link to the JAX docs for the primitive, as a bare URL on the last line.

## References

- [Understanding jaxprs](https://docs.jax.dev/en/latest/jaxpr.html)
- [Writing custom jaxpr interpreters](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)
