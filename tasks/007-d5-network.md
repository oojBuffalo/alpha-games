---
id: 7
title: Build the D5 residual network
status: pending
priority: high
dependencies: [2]
complexity:
recommended_subtasks:
---

## Description
The D5 network as an explicit deliverable: 8×128 residual trunk with three heads (policy 91
spatial channels, scalar tanh value, normalized score-diff aux), parameterized by config so M2.5's
micro-Blokus can instantiate reduced dimensions.

## Details
- New `core/network.py` (torch). `NetworkConfig` frozen dataclass: `input_planes`, `board_size`,
  `policy_channels`, `trunk_blocks=8`, `trunk_channels=128`, `num_aux=1` — plus a
  `from_game(game)` constructor reading `game.input_planes` and `game.policy_shape` (Blokus:
  46 / 14 / 91). Nothing hardcodes 14×14×91 — §12 M2.5 requires a config-parameterized net whose
  dims derive from the game.
- Trunk: 3×3 conv stem to `trunk_channels`, then `trunk_blocks` residual blocks
  (conv-BN-ReLU ×2 with skip), AlphaZero-style.
- Policy head: 1×1 conv to `policy_channels` → `(N, 91, 14, 14)`, then **permute to HWC and
  flatten** so index `(r*board+c)*policy_channels + o` matches
  `games/blokus_duo/actions.py::encode` — §5.1 pins cell-major flatten and notes "M2 pays one
  tensor `permute` before the sparse gather; there is no perf argument for channel-major."
  Output raw logits over the full head; masking/renormalization is the loss's job (task 8).
- Value head: 1×1 conv → FC → scalar with `tanh` (D1/D5).
- Aux head: 1×1 conv → FC → `num_aux` linear outputs (target `score_diff/109 ∈ [−1,1]`, trained
  by MSE; tanh is not pinned for aux — keep linear, the most direct reading of "normalized
  score-diff aux").

## Test Strategy
New `tests/test_network.py`: forward shapes `(N, 17836)`, `(N,)`, `(N, 1)` for the Blokus config;
value output within `[−1, 1]`; **flatten-order golden** — feed a state, spike one logit by
construction (or index the pre-flatten tensor at `(o, r, c)`) and assert it lands at flat index
`(r*14+c)*91+o`; `from_game(BlokusDuo())` picks up 46/14/91; a small synthetic config (e.g.
board 5, 7 planes, 3 policy channels) constructs and runs forward — proving parameterization.
CPU-only, seeded.
