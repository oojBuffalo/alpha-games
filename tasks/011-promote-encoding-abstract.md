---
id: 11
title: Promote the encoding surface to abstract methods
status: pending
priority: low
dependencies: [3]
complexity: 4
recommended_subtasks: 0
---

## Description
Honor the seam comment in `core/game.py` — "These are promoted to abstract methods at M2, when
the network lands" — by making the encoding surface abstract, and backfill the minimal adapters
this forces.

## Details
- `core/game.py` lines ~148–173: `encode_state`, `encode_action`, `decode_action`,
  `policy_shape`, `input_planes` are concrete `NotImplementedError` stubs. Promote them to
  `@abstractmethod` (properties stay properties).
- Ripple — every concrete `Game` must now implement all five. Backfill trivially:
  - `games/tictactoe`: 2 mover-relative 3×3 occupancy planes, flat `policy_shape (9,)`, identity
    codec.
  - `games/connect4`: 2 planes 6×7, `policy_shape (7,)` (column id), identity codec.
  - `tests/fixtures/pass_game.py`: minimal 1–2 plane encoding over its tiny state space.
  - `tests/fixtures/bad_adapters.py`: these exist to violate the *envelope*, not the encoding
    surface — give them the same trivial stubs so they still instantiate far enough to hit
    `assert_v1_envelope`.
- Blokus and Othello already carry their full surfaces after task 3.
- If the fixture ripple turns out disproportionate, the fallback is a doc-first amendment
  deleting the promotion promise and keeping loud stubs — but try the promotion first; it is
  small and it hardens the M2.5/M3 contract (every game entering the training stack must encode).

## Test Strategy
Full battery green (`python3 -m pytest`). Add one negative test: a `Game` subclass missing
`encode_state` raises `TypeError` at instantiation. Spot-check TTT/Connect4 encodings with tiny
goldens (a placed mark shows up in the right plane/cell).

## Complexity Analysis
Each backfill is trivial (tiny boards, identity codecs), but the breadth is real: one ABC change
ripples into two game adapters, two test fixtures, and potentially any test that instantiates a
partial `Game`. The risk is discovering unlisted instantiation sites mid-change — mechanical to
fix, annoying to chase. The documented fallback (doc-first amendment keeping loud stubs) bounds
the downside if the ripple is worse than mapped.

**Suggested expansion approach:** none — keep as one change so the battery is green in a single
commit; splitting per-adapter would leave the tree broken between subtasks.
