# CLAUDE.md — AlphaZero × Blokus Duo

Operational digest for working in this repo. The design doc is the source of truth; this file is the compressed version you read first.

## Project

AlphaZero replication on a **single consumer GPU (RTX 4060 Ti 16 GB)**, built as a **game-agnostic engine** with thin per-game adapters behind a stable `Game` interface. Proof-of-concept game: **Blokus Duo** (14×14, 2-player). The goal is to demonstrate the core learning dynamic — network-guided MCTS bootstrapping from random to strong play — not a superhuman engine. The binding constraint is **self-play throughput**, not network size.

- **Design doc:** `metadocs/blokus-duo-az-design-v0_5.md` — v0.5, milestone plan re-scoped; decisions **D1–D12 pinned** (unchanged). If code needs to contradict it, update the doc first, then the code.
- **Current phase:** implementation, starting at **M0**. Deliberation is finished; do not silently reopen settled decisions — flag explicitly if one looks wrong.

## Scope (asserted in code, not just prose)

v1 engine envelope: **2-player, zero-sum, perfect-information, deterministic.** Adapters declare capabilities (`num_players`, `is_stochastic`, `is_perfect_information`, `symmetry_group`, `value_targets`); `core/` asserts the envelope and fails loudly outside it. Core contract: `current_player`, `legal_moves`, `apply`, `terminal_utility(state, player_id)`, `encode_state`, `encode_action`/`decode_action`, `policy_shape`, `input_planes`.

Seams are documented but **not built**: N-player (value head + backup, M7) and stochastic transitions (`apply`/node types). Imperfect information is permanently out of scope for this codebase (future sibling engine, different solver family).

## Repo layout

`core/` (Game ABC, MCTS, replay, self-play loop, training, eval) · `games/` (one package per adapter: `blokus_duo/`, `tictactoe/`, `connect4/`, `othello/`) · `configs/` · `tests/` · `docs/`.

**Rule: adding a game touches only `games/` + `configs/`.** M1.5 (Othello) enforces this with a zero-`core/`-diff acceptance test.

## Invariants — never violate

1. **Pass invariant.** At every nonterminal state, `current_player(state)` returns a player with ≥1 legal action. Adapters realize forced passes either as an explicit pass action (Othello: 64+1 head) or by skipping inactive players (Blokus: no pass action in 14×14×91). `core/` assumes *only* this invariant — never strict alternation, never monotone blocking. (Blokus blocking is monotone — a blocked player never regains a move; Othello passing is not.)
2. **Player-aware backup.** `edge_value = leaf_value if edge.parent_player == leaf_value_player else -leaf_value`. Q is stored in the parent-mover's perspective, so PUCT selection needs no negamax flip.
3. **Sparse everywhere.** Legal-action index lists; per-node `{N,W,Q,P}` over legal actions only; replay policy targets as `(action_id, visit_count)` pairs. Dense over 17,836 actions is infeasible (~285 KB/node; tens of GB over the replay window).
4. **Orientation-ID determinism.** IDs assigned by sorting canonical cell tuples — never set-iteration order. Serialize an orientation-table hash into every checkpoint and replay dataset.
5. **Oracle first.** Slow cell-grid reference engine before the bitboard engine, independently implemented; differential-test one against the other.

## Golden constants (hardcode tests against these; independently verified)

- 21 free polyominoes, orders 1–5 (1+1+2+5+12); **89 squares** per set.
- Fixed orientations per order {1, 2, 6, 19, 63} = **91**; policy head `14×14×91` = **17,836** raw actions.
- **13,729** in-bounds placements; **414** covering each start square → **828** legal opening actions.
- Start squares **(5,5)** and **(10,10)**, 1-indexed; opening: P1 covers either, P2 covers the other.
- Input: **46 planes** = 2 occupancy + 21 own inventory + 21 opponent inventory + 2 monomino-last completion flags; **T=1**.
- Scoring: −1 per unplaced square; +15 all placed; +5 if monomino placed last (needs the explicit flag — *not* recoverable from occupancy+inventory). Score/player ∈ [−89, +20]; diff ∈ [−109, +109]; draws exist (z = 0).
- Symmetry: **Klein four-group** {identity, 180°, main diagonal, anti-diagonal} — the set-stabilizer of the start squares, no own/opponent relabeling. Full D4 deferred: 90°-class images are rule-consistent but off-support (no start square covered).

## Pinned decisions digest (rationale: design doc §10)

- **D1** value target `z = sign(score_diff)`; aux head predicts `score_diff/109`.
- **D2** action anchor = bounding-box top-left of the oriented piece.
- **D3** 46 input planes. **D4** T=1.
- **D5** trunk 8 residual blocks × 128 ch; batch 256, AMP; SGD mom 0.9, wd 1e-4, LR ≈0.02 warmup+cosine; replay 250k positions (≈25–30k games), 2–4 samples/stored position.
- **D6** validate loop @128 sims; then plies 0–1 always @512 (stored); plies ≥2: 25% @256 (stored) / 75% @64.
- **D7** root Dirichlet ε=0.25, α = 10.8/#legal (≈0.013 at the 828-wide root); root-only, self-play only.
- **D8** playout-cap randomization + score-diff aux in v1 (flagged); KataGo aux bundle deferred.
- **D9** 4 symmetries for augmentation; 8-fold is an M6 experiment.
- **D10** move selection: sample ∝ N (τ=1, no exponentiation) for plies 0–7, argmax N after; `k_temp` in config; π_train ∝ N at *all stored* plies.
- **D11** PUCT `c(s) = 1.25 + log((ΣN + 19653)/19652)` (≈1.25 at v1 budgets); unvisited Q = 0. FPU *reduction* is the first M6 lever — caveat: standard FPU negates parent Q assuming alternation; here it must route through the player-aware perspective map.
- **D12** fast-search (64-sim) positions are **dropped entirely** (KataGo convention) — no value-only storage; every stored sample carries sparse π + z + aux.

## Milestones (design doc §12; v0.5 re-scope)

**M0** Game ABC + TTT + Connect 4 (solved-position values) + **core PUCT MCTS engine** (sparse, player-aware; uniform-prior flag for ladder rung 6 — no leaf-evaluator abstraction) + **synthetic pass-game fixture** + **MCTS-vs-minimax oracle** (+ subtree-reuse/virtual-loss test) + **envelope-rejection negative test** + test-runner/CI entrypoint →
**M1** Blokus: oracle, then bitboards; full battery + **value_targets golden** (z=sign incl. sign(0)=0, aux=/109, |diff|≤109) + fixture-gen + orientation-hash serialize (write-side). M1 owns action encode/decode + the (g,a)→a′ table →
**M1.5** Othello (parallel with M1; depends only on the M0 interface): zero `core/` diffs; explicit-pass; **pass-regain** (non-monotone) test; consecutive-mover backup; D4 group + pass-id fixed-point; carries its own Othello encoding →
**M1.6** (NEW) network-free ladder rungs 1–3 + paired/mirrored game runner + anchored-Elo scaffolding — absolute-strength anchor for M2.5/M3 (rung 4 → M3) →
**M2** encoding (state planes; symmetry maps) + **D5 network build + train step** + sparse policy loss + overfit-one-batch + pipeline decode test →
**M2.5** micro-Blokus (reduced config — **undefined, pin doc-first**) + config-parameterized correctness net + **falsifiable exit test** + **early throughput go/no-go** →
**M3** self-play baseline (functional/correctness; **fixed 128 sims**, not PCR): storage schema (D12 drop-policy → M5), D10, rung 4, **observability** (net-eval + GPU-hour + throughput counters), actor–learner IPC, replay schema, **checkpoint schema + orientation-hash validate-on-load**, seeding, second zero-`core/`-diff Othello re-check through the full stack →
**M4** eval harness (network rungs 5–8; §1 protocol; per-checkpoint CIs) — **live/concurrent with the run**; **define the 'profiled plateau' rule**; assert hash before rung-8; bootstrap seed →
**M5** batched inference + PCR at the D6 schedule + **D12 drop-policy** + **numeric throughput target** + virtual-loss correctness →
**M5.5** (NEW) production run to K checkpoints + compute the §1 Δ verdict (M6 levers interleave on plateaus) →
**M6** ordered levers, each gated on the M4-defined plateau: (1) FPU reduction, (2) 8-fold augmentation (rung-7 Elo), (3) value-only fast-position storage iff value-limited, (4) global-pooling, (5) KataGo aux bundle →
**M7** (future) N-player engine for 4p Blokus: vector value head, max-n/paranoid backup, tournament eval.

Unpinned scalars flagged doc-first: aux loss weight λ_aux (M2/§7), weight-publish interval / checkpoint count K (M3/§6.2), mirrored pairs per (checkpoint,rung) cell (M4/§9).

## Test battery highlights (M1)

- Golden counts above; encode/decode bijection over all 13,729 in-bounds actions.
- Differential fuzzing bitboard vs. oracle (move-gen, apply, terminal detection, scoring).
- **Symmetry joint-permutation golden:** every g × every in-bounds action (≈55k checks): decode → transform cells → re-encode → match the precomputed `(g,a)→a′` table; table closed + bijective. Named failure mode: `g(anchor) ≠ anchor(g(cells))`.
- **Perft(2)-by-opening:** oracle reply counts for all 828 openings frozen as a hash; counts must be constant on Klein-4 orbits; totals give perft(2).
- Monomino-last scoring cases; blocked-stays-blocked (Blokus adapter property, not a core assumption); forced-pass + scoring-flag normalization.
- **value_targets golden (D1):** `z = sign(score_diff)` incl. `sign(0)=0` (draws ≠ losses; §4 has `z=0`), `aux = score_diff/109`, and the `|score_diff| ≤ 109` range assertion — the thin score→z→aux mapping that produces every training target.

## Working principles

- **Deliberate before implementing.** Design-doc changes precede divergent code; keep settled-vs-open explicit.
- **Verify load-bearing claims independently** (enumeration scripts, small proofs) before hardcoding numbers or accepting reviewer/tool feedback; record accept vs. pushback with reasons.
- **Rules-engine correctness is the top project risk.** The M1 battery is not optional and not deferrable; a subtle legality/scoring bug corrupts every downstream signal.
- **Don't build extensions early.** Every M6 lever is gated on a profiled plateau; design the extension points, nothing more.
- Architect for an actor–learner split even though v1 runs single-machine.

## Environment & commands

Python 3.11+ (dev on 3.12). `core/` is **pure-stdlib through M0**; NumPy/torch arrive with encoding/training (M2). Single machine, one GPU (RTX 4060 Ti 16 GB — fits D5 comfortably; benchmark batch 128/256/512).

- **Setup (dev):** `python3 -m pip install -e ".[dev]"` (pytest + ruff). Optional — `pyproject` sets `pythonpath = ["."]`, so the battery also runs against a bare interpreter with pytest/ruff available.
- **Tests:** `python3 -m pytest` from the repo root (full battery). Fast subset: `python3 -m pytest -m "not slow"` (the `slow` marker tags high-sim search sweeps).
- **Lint / format:** `python3 -m ruff check .` · `python3 -m ruff format .` (check-only: add `--check`).
- **CI:** `.github/workflows/ci.yml` runs lint + format-check + full battery on push/PR.
- **Fixture generation (M1):** `python3 scripts/gen_blokus_symmetry_table.py` (seconds) · `python3 scripts/gen_blokus_perft.py` (~3 min; perft(3) is bitboard-generated, Klein-4 orbit-reduced). Both write `tests/fixtures/blokus/*.json` with the orientation hash + encoding conventions embedded; regeneration on unchanged code must be **byte-identical**.

*(Fill in as tooling lands: training entrypoints, eval harness.)*

## References

- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* (AlphaZero), Science 2018; preprint arXiv:1712.01815. Keep the preprint/outline under `docs/` if useful.
- KataGo (Wu, 2020) for playout-cap randomization and the deferred aux-target bundle.
