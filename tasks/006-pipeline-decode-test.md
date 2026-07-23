---
id: 6
title: Add the pipeline decode test
status: pending
priority: high
dependencies: [5]
complexity:
recommended_subtasks:
---

## Description
The §12 M2-named integration test: verify that the plane transform and the action permutation
implement the *same* geometry, by decoding transformed sparse policies back to cell sets. Catches
plane/channel wiring mismatches invisible to M1's table-level golden.

## Details
New `tests/test_blokus_pipeline.py`, quoting the spec: "for random `(state, sparse π)` and each
declared `g`: build `(g·state, g·π)` via plane transform + action permutation, decode every
`(action_id, count)` back to a cell set, and assert the multiset of `(cell-set, count)` equals
`g` applied to the original."

- Generate seeded random reachable states by playing random legal actions through `BlokusDuo`;
  build sparse π as random visit counts over `legal_moves`.
- Transform with `core/augment.augment_sample` (task 5).
- Decode with `games/blokus_duo/actions.py::action_cells`; map original cell sets with
  `games/blokus_duo/symmetry.cell_map(g)`.
- Assert multiset equality of `{(frozenset(cells), count)}` — multiset, not set, since distinct
  actions may carry equal counts.
- Also assert the plane side agrees: occupancy planes of the transformed encoding equal the
  cell-mapped occupancy of the original (guards the named M1 failure mode
  `g(anchor) ≠ anchor(g(cells))` from leaking into the plane wiring).
- Cover all 4 group elements × ~25 random states at several plies (opening, midgame, late) —
  fast enough to stay unmarked; add one `@pytest.mark.slow` sweep with a larger sample if runtime
  allows.

## Test Strategy
This task *is* a test. Verify it fails when sabotaged: temporarily swap `diag`/`antidiag` in the
plane transform (or transpose a plane) and confirm the test catches it, then restore.
