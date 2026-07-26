---
id: 7
title: Build the D5 residual network
status: pending
priority: high
dependencies: [2, 3]
complexity: 6
recommended_subtasks: 4
---

## Description
The D5 network as an explicit deliverable: 8×128 residual trunk with three heads (policy 91
spatial channels, scalar tanh value, normalized score-diff aux), parameterized by config so M2.5's
micro-Blokus can instantiate reduced dimensions.

## Details
- New `core/network.py` (torch). `NetworkConfig` frozen dataclass: `input_planes`, `board_size`,
  `policy_shape` (the adapter's declared tuple — spatial `(H, W, C)` like Blokus's `(14, 14, 91)`
  or flat `(K,)` like Othello's `(65,)`), `trunk_blocks=8`, `trunk_channels=128`, `num_aux` — plus
  a `from_game(game)` constructor reading `game.input_planes`, `game.policy_shape`, and
  `num_aux = len(game.value_targets.aux_names)` (Blokus → 46 / 14 / (14,14,91) / 1; Othello →
  2 / 8 / (65,) / 0). Nothing hardcodes 14×14×91 — §12 M2.5 requires a config-parameterized net
  whose dims derive from the game, and M3's zero-`core/`-diff re-check drives Othello through
  this same class, so the full declared contract (flat heads, zero aux) is in scope *now*, not a
  rejection branch. Depends on task 3 for `BlokusDuo.input_planes` (currently
  `NotImplementedError` in `core/game.py`); Othello's encoding surface already exists (M1.5).
- Trunk: 3×3 conv stem to `trunk_channels`, then `trunk_blocks` residual blocks
  (conv-BN-ReLU ×2 with skip), AlphaZero-style.
- Policy head, spatial shape (`(H, W, C)`): 1×1 conv to `C` channels → `(N, 91, 14, 14)`, then
  **permute to HWC and flatten** so index `(r*board+c)*C + o` matches
  `games/blokus_duo/actions.py::encode` — §5.1 pins cell-major flatten and notes "M2 pays one
  tensor `permute` before the sparse gather; there is no perf argument for channel-major."
  Output raw logits over the full head; masking/renormalization is the loss's job (task 8).
- Policy head, flat shape (`(K,)`, Othello's 64+1 head): 1×1 conv reduction → FC → `K` raw
  logits — no permute (a flat head has no spatial factorization to preserve; the pass id lives
  outside the board grid). Both branches end in the same `(N, num_actions)` logits contract.
- Value head: 1×1 conv → FC → scalar with `tanh` (D1/D5).
- Aux head: built only when `num_aux > 0` (Othello declares none — §12 M1.5 pins "no aux head"):
  1×1 conv → FC → `num_aux` linear outputs (target `score_diff/109 ∈ [−1,1]`, trained by MSE;
  tanh is not pinned for aux — keep linear, the most direct reading of "normalized score-diff
  aux"). Forward returns `None` (or an empty tensor — pick one and document it) when absent.

## Test Strategy
New `tests/test_network.py`: forward shapes `(N, 17836)`, `(N,)`, `(N, 1)` for the Blokus config;
value output within `[−1, 1]`; **flatten-order golden** — feed a state, spike one logit by
construction (or index the pre-flatten tensor at `(o, r, c)`) and assert it lands at flat index
`(r*14+c)*91+o`; `from_game(BlokusDuo())` picks up 46/14/(14,14,91)/1 (needs task 3's
`input_planes` — hence the dependency); `from_game(Othello())` picks up 2/8/(65,)/0 and its
forward yields `(N, 65)` logits with no aux output; a small synthetic config (e.g. board 5,
7 planes, 3 policy channels) constructs and runs forward — proving parameterization. CPU-only,
seeded.

## Complexity Analysis
The largest single build in M2: first torch module in the repo, a config dataclass with a
`from_game` bridge, an 8-block residual trunk, and three heads with different output contracts.
The highest-risk element is the policy-head flatten: `(N, 91, 14, 14)` must permute to HWC before
flattening so flat indices match `actions.encode` — a silent mismatch here corrupts every
training target while all shapes still check out, which is why the flatten-order golden is
non-negotiable. Parameterization for M2.5 (nothing hardcodes 46/14/91) adds a second axis every
piece must respect.

**Suggested expansion approach:** split four ways — (1) `NetworkConfig` + `from_game` reading
`input_planes`/`policy_shape`; (2) stem + residual trunk; (3) the three heads including the
HWC permute/flatten pinned to `actions.encode`; (4) `tests/test_network.py` with the
flatten-order golden and the micro-config parameterization proof.

## Subtasks
### 7.1 Define NetworkConfig and the from_game bridge — status: pending
Frozen dataclass carrying every dimension the net needs; nothing downstream hardcodes Blokus
numbers. **Details:** `core/network.py`: `NetworkConfig(input_planes, board_size, policy_shape,
trunk_blocks=8, trunk_channels=128, num_aux)` + classmethod `from_game(game)` reading
`game.input_planes`, `game.policy_shape`, and `num_aux = len(game.value_targets.aux_names)`.
Accept both declared head shapes — spatial 3-tuples `(H, W, C)` (validate `H == W == board`) and
flat 1-tuples `(K,)` (Othello's `(65,)`) — and reject anything else loudly; M3's
zero-`core/`-diff Othello re-check forecloses a spatial-only `from_game`. **Test:**
`from_game(BlokusDuo())` golden (46/14/(14,14,91)/1) and `from_game(Othello())` golden
(2/8/(65,)/0); a hand-built micro config constructs. **Depends on:** —

### 7.2 Build the conv stem and residual trunk — status: pending
The D5 body: 3×3 conv stem to `trunk_channels`, then `trunk_blocks` residual blocks
(conv-BN-ReLU ×2 with skip), AlphaZero-style. **Details:** plain `nn.Module`s parameterized only
by `NetworkConfig`; no pooling, board size preserved throughout; seeded init. **Test:** forward
shape `(N, trunk_channels, board, board)` for Blokus and a micro config; gradients flow.
**Depends on:** 7.1

### 7.3 Add the three heads with the pinned HWC flatten — status: pending
Policy, value, aux heads with the §5.1 flatten contract — the load-bearing subtask. **Details:**
policy, spatial shape: 1×1 conv to `C` channels → `(N, C, H, W)`, then `permute(0, 2, 3, 1)` and
flatten so flat index `(r*board+c)*C + o` matches `games/blokus_duo/actions.py::encode` ("M2 pays
one tensor permute before the sparse gather"); flat shape `(K,)`: 1×1 conv reduction → FC → `K`
logits, no permute; both branches emit raw `(N, num_actions)` logits, no masking. Value: 1×1 conv
→ FC → scalar `tanh` (D1). Aux: constructed only when `num_aux > 0`; 1×1 conv → FC → `num_aux`
linear outputs. **Test:** flatten-order spot-check by indexing the pre-flatten tensor at
`(o, r, c)`; flat config emits `(N, K)`; `num_aux=0` config has no aux parameters; value range.
**Depends on:** 7.2

### 7.4 Write tests/test_network.py — status: pending
The module-level battery proving shapes, the flatten golden, and parameterization. **Details:**
forward shapes `(N, 17836)`/`(N,)`/`(N, 1)` for `from_game(BlokusDuo())`; value ∈ [−1, 1];
the flatten-order golden against `actions.encode` over a sample of `(r, c, o)` triples;
`from_game(Othello())` → `(N, 65)` logits, no aux output; a synthetic micro config (board 5,
7 planes, 3 channels) runs forward — CPU-only, seeded. **Test:**
`python3 -m pytest tests/test_network.py`. **Depends on:** 7.3
