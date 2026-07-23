---
id: 5
title: Add training-side symmetry augmentation
status: pending
priority: medium
dependencies: [4]
complexity:
recommended_subtasks:
---

## Description
Wire the declared symmetry maps into training targets/augmentation — the M2-owned side of the
M1/M2 ownership boundary: a game-generic utility that applies one group element to a
`(state planes, sparse π)` sample.

## Details
- New `core/augment.py`: `augment_sample(game, planes, sparse_pi, g_index)` where `sparse_pi` is
  the D12-shaped list of `(action_id, visit_count)` pairs. Look up
  `plane_transform, action_permutation = game.symmetry_group[g_index]` and return
  `(plane_transform(planes), [(action_permutation[a], n) for a, n in sparse_pi])`.
- Core hardcodes no group: it consumes only the adapter-declared `symmetry_group` (§6.1 —
  `SymmetryElement = (plane_transform, action_permutation)` in `core/game.py`).
- Keep it pure stdlib (operates on nested tuples + pair lists) so it tests without torch and works
  for any adapter. Tensor conversion happens later, at the collate boundary (task 9).
- D9: four symmetries, augmentation-time; sampling strategy (which g per sample) belongs to the
  M3 training loop — this task delivers only the transform.

## Test Strategy
Unit tests: identity element is a no-op on both planes and pairs; visit-count multiset is
preserved under every g; applying g then its inverse round-trips (Klein-4 elements are
self-inverse). Run the same tests against both `BlokusDuo` and `Othello` adapters to prove
game-genericity.
