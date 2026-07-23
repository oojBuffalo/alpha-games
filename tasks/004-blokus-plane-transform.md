---
id: 4
title: Implement Blokus plane-side symmetry transform
status: pending
priority: high
dependencies: [3]
complexity: 3
recommended_subtasks: 0
---

## Description
Replace the raising sentinel `plane_transform_placeholder` in `games/blokus_duo/symmetry.py` with
the real Klein-4 plane transform over the 46-plane encoding, completing the adapter's declared
`symmetry_group` pairs.

## Details
- `games/blokus_duo/symmetry.py`: implement `plane_transform(name)` for `GROUP_NAMES =
  ("identity", "rot180", "diag", "antidiag")`, applying the existing `cell_map` spatially to every
  plane of the 46×14×14 nested-tuple output of `encode_state` (constant inventory/flag planes are
  invariant under the map but go through the same code path — no special-casing). Mirror
  `games/othello/symmetry.py::plane_transform`, the M1.5 template.
- `games/blokus_duo/game.py`: `symmetry_group` currently builds
  `(plane_transform_placeholder, full_permutation(g))` pairs — swap in the real transform per
  element. Delete the placeholder and its "lands with M2" comment.
- The action-permutation side (`full_permutation`, the `(g,a)→a′` table) is M1-owned and already
  correct — do not touch it.

## Test Strategy
Equivariance test in `tests/test_blokus_encoding.py` (or a new module): for each group element g
and seeded random reachable states, `plane_transform(g)(encode_state(s)) ==
encode_state(state_transform(g)(s))` — reusing the module utility
`games/blokus_duo/symmetry.state_transform` from the M1 battery. Also assert
`symmetry_group[0]` (identity) round-trips exactly.

## Complexity Analysis
Small, well-fenced change: apply the existing `cell_map` spatially across planes, mirroring the
Othello `plane_transform` template, and swap out one sentinel in `symmetry_group`. The Klein-4
elements are self-inverse, the equivariance test pattern already exists (`state_transform` is
built for exactly this), and the risky geometric work — the `(g,a)→a′` table — is M1-owned and
untouched.

**Suggested expansion approach:** none — atomic; the implementation and its equivariance test
belong in one change.
