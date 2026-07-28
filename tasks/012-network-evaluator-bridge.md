---
id: 12
title: Bridge the network into MCTS leaf evaluation
status: pending
priority: high
dependencies: [3, 7]
complexity: 4
recommended_subtasks: 0
---

## Description
The callable that makes the design-doc sentence "the leaf value is `terminal_utility` at
terminals and (from M2) the network" (§12 M0) true: turn `(game, state)` into a mover-relative
value plus per-legal-action priors through the existing `MCTS.evaluate` seam. Without this,
tasks 7–9 produce a trained network that search never consumes.

## Details
- `make_network_evaluator(net, game, device="cpu")` in `core/network.py` (keeping torch confined
  to the three modules named in the `pyproject.toml` comment), returning a callable matching the
  M0 seam `Evaluator = Callable[[Game, State], tuple[float, dict[Action, float] | None]]`
  (`core/mcts.py:29-30`) — the seam whose docstring already says "The network plugs in here at
  M2 — the same seam, no new abstraction." No `core/mcts.py` changes.
- Per call: `game.encode_state(state)` → batch-1 float32 tensor on `device`; forward under
  `torch.inference_mode()` with the net in `eval()` mode.
- **Value:** the scalar `tanh` head as a float. It is mover-relative by construction — training
  targets are stored from the mover's perspective over the mover-relative §5.2 encoding — which
  is exactly what the seam contract requires (`value_from_movers_perspective`); the player-aware
  backup does the rest. Say this in the docstring; it is the sign convention a bug would corrupt.
- **Priors:** `legal_moves` already returns flat action ids (`core/game.py` — `encode_action`
  maps *moves* to ids and has no business here), so index the flat `(num_actions,)` logits
  vector directly with each legal id and return `{action_id: logit}` — **raw logits, not
  probabilities**. The seam's existing `MCTS._priors` performs the single legal-subset softmax
  (`core/mcts.py`); normalizing in the bridge as well would softmax the policy twice and skew
  every prior. Emit an entry for every legal id — `_priors` silently defaults missing ids to
  logit `0.0`.
- **Batch-1 per-leaf inference is the M2/M3 functional path** — batched/asynchronous inference
  is explicitly M5 scope; do not build queueing here.
- Works unchanged for any conforming adapter (Othello's flat head included) — nothing in the
  bridge may reference Blokus dimensions.

## Test Strategy
New `tests/test_network_evaluator.py` (CPU-only, seeded, small sim counts): (1) integration —
`MCTS(BlokusDuo(), evaluate=make_network_evaluator(...))` with a seeded random-init `from_game`
net runs a few dozen sims from the opening and `best_action` returns a legal action; (2) priors
golden — root `P` (as produced by `MCTS._priors` from the bridge's returned logits) equals a
directly-computed legal-subset softmax of the net's logits and sums to 1 — falsifies exactly the
double-softmax bug, since `softmax(softmax(x)) != softmax(x)`; (3) value flows — with a nonzero-value net, root child Q values are not all zero
(distinguishing the bridge from M0's zero-value default); (4) `uniform_prior=True` keeps the
net's value but overrides priors uniform (ladder rung 6 compatibility); (5) the same bridge
drives Othello through MCTS with zero `core/` changes.

## Complexity Analysis
Thin glue (~30 lines) with three hazards that each silently corrupt search rather than crash:
the mover-relative value convention through the player-aware backup, the action-id↔logit-index
correspondence (which task 7's flatten golden pins on the network side but must be re-asserted
end-to-end here — the legal ids *are* the logit indices; nothing re-encodes), and single-softmax
ownership (the seam normalizes; the bridge must return raw logits). The integration test is most
of the effort.

**Suggested expansion approach:** none — atomic; the bridge and its test land together.
