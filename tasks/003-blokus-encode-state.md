---
id: 3
title: Implement Blokus 46-plane encode_state
status: pending
priority: high
dependencies: []
complexity: 5
recommended_subtasks: 3
---

## Description
Fill the M2 seam in the Blokus adapter: override `encode_state` and `input_planes` (currently the
base-class `NotImplementedError` stubs) with the D3 46-plane encoding from design doc §5.2.

## Details
- `games/blokus_duo/game.py`: add `encode_state(state)` and `input_planes` (= 46). Plane order is
  pinned by D3/§5.2, mover-relative (own = side to move, no side-to-move plane):
  1. own occupancy, 2. opponent occupancy,
  3–23. own inventory (one plane per piece, order fixed by `pieces.BASE_PIECES` — the file
  comments "This piece order also fixes the M2 inventory-plane order (D3)"),
  24–44. opponent inventory (same order),
  45. own monomino-last flag, 46. opponent monomino-last flag.
- Inventory and flag planes are constant planes (all 1s if the piece is still in inventory / the
  flag is set, else all 0s) — the standard AlphaZero broadcast convention.
- State tuple layout (both engines): `(occ0, occ1, inv0, inv1, mono_last0, mono_last1, to_play,
  terminal)`. Handle both occupancy representations — 196-bit ints (bitboard) and frozensets
  (oracle) — the same way `symmetry.state_transform` already does.
- Return nested tuples of `{0,1}` shaped 46×14×14, mirroring `games/othello/game.py::encode_state`
  (the M1.5 precedent). Adapters stay stdlib-pure; the training boundary converts with
  `numpy.asarray` (task 9).

## Test Strategy
New `tests/test_blokus_encoding.py` mirroring `tests/test_othello_encoding.py`: `input_planes ==
46`; initial state has empty occupancy planes, all 42 inventory planes full, flag planes zero;
mover-relativity (own/opponent planes swap after one move); inventory plane for a placed piece
zeroes out; monomino-last flag plane sets on a crafted end-state; occupancy planes match
`bitboard`/`oracle` cell sets on seeded random playouts.

## Complexity Analysis
Conceptually a pure projection of a state tuple that already carries every plane's data, with a
direct template to mirror (`games/othello/game.py::encode_state`) — that caps the score. What
raises it: 46 planes with a pinned order that downstream training silently depends on (D3),
mover-relative perspective, dual occupancy representations (196-bit ints vs frozensets), and a
six-category test module. Encoding bugs here corrupt every later training signal, so the testing
burden is deliberately heavy.

**Suggested expansion approach:** split three ways — (1) occupancy planes with dual-representation
handling and mover-relativity; (2) inventory + monomino-flag planes, `input_planes`, and the
adapter override wiring; (3) the `tests/test_blokus_encoding.py` module covering all six
categories in the test strategy.
