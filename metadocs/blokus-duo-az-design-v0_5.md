# AlphaZero × Blokus Duo — Design Doc

**Status:** v0.4 — decision-complete. All config-level decisions are pinned (D1–D12), the success criterion is pre-registered, and §11 holds only an optional stretch goal. Buildable as written.
**Changes from v0.3:** pinned the last three config decisions — **D10** (self-play move selection), **D11** (PUCT constant + first-play urgency), **D12** (PCR storage semantics) — and firmed D6's ply-0–1 clause. Formalized §1's success criterion as a pre-registered paired-bootstrap contrast (Elo anchor now defined). Added the pass-convention invariant to the `Game` contract (§6.1) and promoted the Othello abstraction test to firm milestone **M1.5** with expanded acceptance checks (the invariant's tests presuppose it). Documented the off-support caveat behind deferring 8-fold symmetry (§8). Expanded test batteries: symmetry joint-permutation golden and perft(2)-by-opening with orbit cross-check (M1); pipeline decode test (M2). Ordered the M6 efficiency levers. General tightening.

---

## 1. Objective & success criterion

Replicate AlphaZero on **Blokus Duo** as a proof of concept on a **single consumer GPU** (RTX 4060 Ti 16 GB), inside a **unified engine for a variety of board games**. The deliverable demonstrates the core learning dynamic — network-guided MCTS bootstrapping from random to strong play — not a superhuman engine.

**Primary criterion (pre-registered).** Let `Elo_1 … Elo_K` be the anchored Elo of successive checkpoints against the frozen ladder (§9), plotted against cumulative network evaluations and GPU-hours (anchor: rung 1, uniform random, fixed at Elo 0). Define the contrast

> **Δ = mean(Elo over the final ⌈K/3⌉ checkpoints) − mean(Elo over the first ⌈K/3⌉ checkpoints).**

Success = the 95% paired-bootstrap confidence interval on Δ lies strictly above 0: resample, with replacement, mirrored game *pairs* within each (checkpoint, rung) cell; refit the anchored Elo curve per replicate (draws scored 0.5); B ≈ 2,000 replicates. A Mann–Kendall trend test on the checkpoint Elo sequence is reported as a secondary, functional-form-free check (reported, not gating). Per-checkpoint monotonicity is **not** required.

**Secondary (qualitative):** emergence of known strategy — central expansion, corner contention/blocking, small-piece hoarding, monomino saved for last.

Strength claim is scoped to **"strong relative to fixed algorithmic baselines"**; external human/engine calibration is an optional stretch (§11).

---

## 2. Scope & the target-family boundary

The engine targets one game family in v1, defined by three independent axes. The governing principle: **widen to the boundary of what the AlphaZero algorithm natively handles, and stop there.** Multiplayer and stochasticity are native (with caveats); imperfect information is a different algorithm.

| Axis | v1 envelope | Disposition | Where it changes |
|------|-------------|-------------|------------------|
| **Players** | 2-player | **Interface N-player-ready; multiplayer *engine* deferred** | value head + MCTS backup |
| **Determinism** | deterministic | **Transition *seam* left open; chance nodes deferred** | `apply` / MCTS node types |
| **Information** | perfect | **Walled off; imperfect-info is a future *sibling engine*** | the solver itself (not this codebase) |

(Zero-sum vs. general-sum is a fourth, mostly-moot axis: targets are zero/constant-sum, and a mild general-sum flavor is inherited for free the moment the engine goes N-player — alliances are a general-sum phenomenon.) Full pros/cons per axis: **Appendix A**.

**v1 target family = 2-player, zero-sum, perfect-information, deterministic.** This is exactly TTT, Connect 4, Othello, Hex, Gomoku, Breakthrough, Amazons, chess, Go, Blokus Duo — a large family. It is what the scalar `terminal_utility ∈ {−1,0,+1}` and single value head commit us to.

**Non-goals for v1 (with the seam that keeps each reachable):**
- **4-player Blokus / N-player games.** The *interface* already carries per-player utility and a documented path to a vector value head, so these are reachable (future milestone M7). What's deferred is the multiplayer *solver*: max-n vs. paranoid backup, a value vector, and tournament-based self-play/eval. Not built in v1.
- **Stochastic games (backgammon-class).** The transition seam is marked so `apply` is the deterministic special case of a possibly-stochastic transition. Chance nodes / expectimax are not built until a stochastic game lands on the roadmap.
- **Imperfect-information games (poker, Stratego).** Explicitly a **future sibling engine, not an extension.** MCTS-over-states is invalid there (nodes become information sets over a belief distribution); it needs a CFR / Deep CFR / ReBeL-family solver and exploitability-based evaluation. Kept out so this codebase stays unified around one solver.
- **Distributed/multi-machine training.** Architect for an actor–learner split; v1 runs single-machine.

---

## 3. Why Blokus Duo fits vanilla AlphaZero

- **Throughput is the binding constraint** on one GPU (self-play generation, not net size). Games are **≤42 plies** → many games per GPU-hour.
- **True 2-player zero-sum** → unmodified scalar value head (interface is N-player-ready, §6.1).
- **Grid game** → CNN body + spatial-plane encoding transfer.
- **Board has dihedral symmetry** (unlike chess/shogi) → symmetry augmentation is reintroducible (§8).

---

## 4. Game model

- **Pieces:** the 21 free polyominoes of order 1–5 (1+1+2+5+12), one set per player; 89 squares per set.
- **Opening:** Player 1 places a piece covering **either** of the two starting points; Player 2 must cover the **other**. Start points are near-central, at (5,5) and (10,10) 1-indexed — *not* corners.
- **Placement (post-opening):** a new piece must touch ≥1 **corner** of the player's own pieces, must **never share an edge** with the player's own pieces, and may freely touch opponent pieces.
- **Passing & termination:** a player with no legal placement passes. **Monotonicity:** once blocked, a player can never regain a move (own pieces/inventory frozen; opponent only adds occupancy, which can only remove options). So a passed player is permanently inactive; a lone active player makes **consecutive** placements. Game ends when neither is active. (This monotonicity is a *Blokus adapter* fact, exploitable there for caching; `core/` never assumes it — see §6.1 and M1.5.)
- **Scoring (official):** −1 per remaining unplaced square; **+15** if all 21 placed; **+5** more if the *last* piece placed was the monomino (given all placed). Per-player score ∈ **[−89, +20]**; score difference ∈ **[−109, +109]**. Draws (equal scores) are possible → `z = 0` exists.

> **Scoring-state caveat (drives §5):** (occupancy, inventory) cannot distinguish "all placed, monomino last" (+5) from "all placed, other last." The +5 must be tracked explicitly or terminal value is wrong.

---

## 5. State & action representation *(Blokus-instance; the engine is generic)*

### 5.1 Action head — `14×14×91` (verified)
The 21 pieces have **91 fixed orientations** (rotations + reflections): 1+2+6+19+63 for monomino→pentomino. Policy head = `14×14×91`: channel = one fixed-orientation polyomino, cell = **bounding-box top-left anchor**. Raw actions = 17,836; nearly all illegal → masking + renormalization over legal logits is load-bearing. There is **no pass action** in this head; forced passes are realized by the adapter per the §6.1 invariant.

- **Anchor = bounding-box top-left** (settled): unique encoding, nonnegative local coords, trivial bounds checks, precomputable masks.
- **Orientation-ID determinism:** assign IDs by sorting canonical cell tuples — **never** Python set-iteration order. Serialize an orientation-table hash into every checkpoint and replay dataset.

> **Verified counts (independently reproduced) — hardened as golden tests in M1:** orientations per size {1,2,6,19,63} = **91**; in-bounds placements on 14×14 = **13,729**; placements covering each start square = **414** (→ **828** legal initial actions).

### 5.2 Input planes — 46, `T=1`
2 occupancy (own, opponent) + 21 own inventory + 21 opponent inventory + 2 completion-bonus flags (own/opponent "monomino-was-last-when-completed"). `T=1` is sufficient once the flags are added (dynamics/legality are Markov in (occupancy, inventory); the only non-recoverable terminal quantity, +5, is now carried by a bit). Dropped: side-to-move plane (constant under own/opponent relabeling) and ply-count plane (inventory exposes phase).

---

## 6. Engine architecture *(game-generic)*

### 6.1 `Game` interface — capabilities, seams, and the pass invariant
Adapters **declare** their capabilities; core asserts the v1 envelope and fails loudly if a game exceeds it. This encodes the scope boundary in code, not just prose.

```
# --- declared per-game properties ---
num_players(game)             -> int    # 2 in v1; ASSERTED == 2 by the v1 engine
is_stochastic(game)           -> bool   # False in v1; ASSERTED False
is_perfect_information(game)  -> bool   # True; invariant for THIS engine (else: sibling engine)
symmetry_group(game)          -> [(plane_transform, action_permutation), ...]   # declared, not hardcoded
value_targets(game)           -> primary z-map + optional aux heads (tensor + loss weight)

# --- core contract ---
current_player(state)              -> player_id
    # PASS INVARIANT: at every nonterminal state, returns a player with >= 1 legal
    # action. Adapters realize forced passes either as (i) an explicit pass action
    # in the action space (Othello: 64+1 head) or (ii) skipping inactive players
    # here / in apply() (Blokus: no pass action in 14x14x91). core/ assumes ONLY
    # this invariant -- never strict alternation, never monotone blocking.
legal_moves(state)                 -> sparse legal-action-index list (+ masks)   # nonempty at nonterminal states
apply(state, action)               -> state
    # v1: deterministic. STOCHASTIC SEAM -- may generalize to a distribution over
    # next states / explicit chance nodes; apply() is then the point-mass special case.
terminal_utility(state, player_id) -> R
    # v1: {-1,0,+1}. N-PLAYER SEAM -- already per-player; generalizes to a utility vector.
encode_state(state)                -> plane tensor
encode_action / decode_action      -> action_id <-> move
policy_shape ; input_planes
```

The **network predicts** an expected outcome from the mover's perspective; a **terminal state has an exact utility** — hence `terminal_utility` is deterministic and player-parameterized, not an "expected outcome." Value head is **scalar `tanh` in v1**, with a documented path to a **per-player vector head** for the N-player engine.

*Design pattern:* symmetry, value/aux targets, the three capability flags, and the pass realization are all **declared or chosen by the adapter**; `core/` contains no game-specific logic for any of them. Adding a game touches only `games/` and `configs/`.

### 6.2 MCTS (PUCT) — sparse, player-aware
- **Selection (D11):** at node `s`, choose `argmax_a [ Q(a) + U(s,a) ]` with `U(s,a) = c(s) · P(a) · √(Σ_b N(b)) / (1 + N(a))` and the AZ growth schedule `c(s) = c_init + log((Σ_b N(b) + c_base + 1) / c_base)`, `c_init = 1.25`, `c_base = 19,652`. At v1 budgets the log term is ≤ ~0.026 (at 512 sims), so `c ≈ 1.25`; the formula is kept for AZ fidelity at zero cost. `Q(a)` is stored in the parent-mover's perspective by the backup rule below, so selection needs no negamax flip.
- **First-play urgency (D11):** unvisited edges use `Q = 0` (AZ convention), i.e. the prior-value of an untried move is "draw." At 64-sim fast searches this makes search strongly prior-driven — acceptable for that tier by construction (D6/D12). **FPU *reduction*** (LC0/KataGo-style `Q_init = Q_parent − c_fpu·√(Σ P(visited))`) is the first M6 lever; caveat recorded now: standard implementations negate parent Q assuming alternation, so under player-aware backup it must route through the same parent/child perspective map as the backup rule.
- **Sparse edges:** at expansion, gather only legal logits, softmax over legal, store `{N,W,Q,P}` of length = #legal; lazily allocate children. Replay policy targets stored sparsely as `(action_id, visit_count)`. (Dense is infeasible: dense targets over a 250k buffer ≈ tens of GB; per-node dense `{N,W,Q,P}` ≈ 285 KB on ~99% illegal actions.)
- **Player-aware backup** (mandatory — no strict alternation once one player is blocked):
  ```
  edge_value = leaf_value if edge.parent_player == leaf_value_player else -leaf_value
  ```
  This is the **2-player zero-sum specialization**. *N-player seam:* the generalization is a **vector-valued backup** where each edge takes the parent-player's utility component, under a declared opponent model (max-n vs. paranoid). Localized to backup + value head; deferred with M7.
- **Deterministic-transition assumption** (v1). *Stochastic seam:* chance nodes + expectimax averaging would slot in at node expansion; not built.
- **Batched parallel selection:** virtual loss / in-flight-leaf marker so concurrent traversals don't re-select the same unevaluated leaf.
- **Version pinning:** publish weights every fixed number of learner steps; pin one model version per self-play game; never swap mid-search; record `model_version` per replay sample; reuse the chosen subtree between moves.

### 6.3 Rules engine — precomputed bitboards *(Blokus-instance)*
196-bit board (one Python int for dev; 4×uint64 for a compiled hot path). Per in-bounds action precompute `piece_id, orientation_id, anchor, placement_bb, orthogonal_halo_bb, diagonal_halo_bb`. Post-opening legality:
```
piece available
placement_bb & occupied       == 0
orthogonal_halo_bb & own      == 0
diagonal_halo_bb  & own       != 0
```
The first move substitutes the **start-point condition** (placement covers the player's required start square) for the diagonal-contact condition; the other three conditions hold unchanged (the own-halo tests are vacuous on an empty own board).

---

## 7. Training & config

- **GPU: RTX 4060 Ti 16 GB (confirmed).**
- **Loss:** `l = (z − v)² − πᵀ log p + (aux term) + c‖θ‖²`.
- **Value target (D1):** `z = sign(score_diff)`; **auxiliary head** predicts normalized score difference `score_diff / 109`. (Blokus is 2-player, so z is scalar; the N-player engine swaps this for a vector target.)
- **Network (D5):** trunk 8 residual blocks × 128 channels; policy 91 spatial channels; value scalar `tanh`; aux normalized score-diff. Batch 256 (benchmark 128/256/512); mixed precision; SGD momentum 0.9; weight decay 1e-4; LR ≈ 0.02 warmup + cosine; replay ≈ 250k positions; replay ratio 2–4 samples per stored position. A KataGo-style global-pooling path may help later (inventory/score are global features); does not block the baseline (M6 lever).
- **Search budgets & storage (D6 + D12):** validate the loop at a fixed **128** sims first. Then:
  - **Plies 0–1: always full at 512 sims, targets stored** — boosted for the 828-wide root.
  - **Plies ≥ 2: 25% full @ 256 (targets stored) / 75% fast @ 64 (position dropped entirely — KataGo convention).**
  Every stored sample carries **both** targets (sparse π, z, aux). Accounting: at ~30–40 plies/game this stores ≈ 9–10 positions/game, so the 250k window ≈ 25–30k games. Value-only storage of fast positions is **not** used: z is constant within a game, so the extra samples are within-game correlated (far less than 4× independent value signal) while costing per-sample loss masks and stratified batching; the aux target inherits the same correlation. Revisit only on a demonstrated value-limited plateau (M6).
- **Move selection (D10):** play `a ~ π_play` with `π_play(a) ∝ N(a)` (τ = 1; sample the visit counts directly, no exponentiation) for the first **k_temp = 8** plies (0-indexed plies 0–7), **argmax N** thereafter; `k_temp` is a config field. The training target is `π_train(a) = N(a)/ΣN` at *all stored* plies regardless of selection mode. The sampling window covers the sim-boosted plies 0–1, concentrating exploration where visit distributions are highest-quality. Rejected: τ = 1 throughout (in a 30–40-ply game, late sampled blunders flip z and the score-diff aux on decided games); annealed τ (an extra schedule and exponentiation for marginal benefit at this game length).
- **Dirichlet (D7):** `ε = 0.25`; `α = 10.8 / #legal_actions` (≈ 0.013 at the 828-action root); applied at the search root only, self-play only.

---

## 8. Symmetry augmentation (D9) *(declared per §6.1)*

Blokus Duo declares a **Klein four-group**: identity, 180° rotation, main-diagonal reflection, anti-diagonal reflection — the set-stabilizer of the two diagonal start squares. Valid everywhere with **no own/opponent relabeling** (because either square is a legal opening, the two square-swapping transforms map legal→legal cleanly). **Use these four for v1.** Other games declare their own group (Go: 8; chess: identity only) via `symmetry_group`.

**Why 8-fold stays deferred:** all eight D4 transforms are valid automorphisms of the *post-opening rules*, but the four 90°-class transforms map reachable states to rule-consistent yet **unreachable** ones (no start square covered). Augmented targets remain *valid* — post-opening dynamics never reference the start squares, so every subgame is fully D4-equivariant — but training would be partly off-support, shifting the input distribution (and BN statistics) toward states never seen at inference. Listed as an M6 experiment, judged by rung-7 Elo.

---

## 9. Evaluation

**Frozen baseline ladder (anchored Elo; rung 1 fixed at 0):** (1) uniform random; (2) largest-piece/coverage heuristic; (3) mobility heuristic; (4) **UCT + random rollouts** (a real value signal — replaces "uniform-prior MCTS," which without rollouts/value learns nothing); (5) network policy, no search; (6) uniform-prior MCTS **with network value**; (7) full policy-and-value MCTS; (8) historical checkpoints.

**Protocol:** disable Dirichlet; deterministic root (argmax N); alternate first player and balance the start-point choice; **paired/mirrored games** — the resampling unit for §1's bootstrap; draws scored 0.5; report per-checkpoint confidence intervals; freeze baseline versions. The pre-registered Δ contrast (§1) is the primary criterion; the Mann–Kendall test is the secondary; per-checkpoint monotonicity is not required. (For the future N-player engine, 2p Elo is replaced by tournament / per-seat scoring — see M7.)

---

## 10. Resolved decisions (D1–D12)

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | `z = sign(score_diff)` + `score_diff/109` aux | Optimize winning; predict the statistic separately. |
| D2 | Bounding-box top-left anchor | Unique, trivial bounds, precomputable, deterministic IDs. |
| D3 | 2 occ + 21+21 inventory + 2 flags = 46 planes | Minimal Markov state; drops redundant planes. |
| D4 | `T=1` | Sufficient with completion flags. |
| D5 | 8×128 trunk, batch 256, AMP, 250k replay | Firm for the confirmed 4060 Ti 16 GB; benchmark 128/256/512. |
| D6 | Validate @128; then plies 0–1 @512 (stored); plies ≥2: 25% @256 (stored) / 75% @64 | Separates outcome-generation from policy-improvement compute; boosted 828-wide root. |
| D7 | `ε=0.25`, `α = 10.8/#legal`, root-only, self-play-only | Constant total Dirichlet concentration. |
| D8 | PCR in v1 (flagged) + score-diff aux; defer KataGo bundle | Big efficiency win, low complexity; baseline first. |
| D9 | 4 symmetries everywhere; 8 post-opening deferred | Declared set-stabilizer; no relabeling; 90°-class images are off-support (§8). |
| D10 | Sample `∝ N` for plies < k_temp = 8, argmax after; `π_train ∝ N` at all stored plies | Opening diversity concentrated where sims are boosted; no z-flipping late blunders; no exponentiation. |
| D11 | `c(s) = c_init + log((N+c_base+1)/c_base)`, c_init = 1.25, c_base = 19,652; unvisited `Q = 0` | AZ-faithful; log term ≈ 0.026 @512 → effectively 1.25. FPU reduction = first M6 lever, with the perspective caveat (§6.2). |
| D12 | Fast-search positions dropped entirely (KataGo convention); no value-only storage | Every sample carries both targets; clean window accounting (≈25–30k games); value-only variant buys within-game-correlated z at the cost of masks + stratified batching — M6, iff value-limited. |

---

## 11. Open / to confirm

- **External strength calibration (optional stretch).** Matches vs. experienced players or a credible external Blokus engine, to claim human-amateur strength rather than "strong vs. fixed baselines."

No open design decisions remain. The abstraction-validation game, previously pending, is promoted to firm milestone **M1.5** (Othello): the pass-convention invariant (§6.1) and its acceptance tests presuppose it, and it is the proof of the "unified engine" claim — `core/` must survive a second, structurally different 2-player game unchanged.

---

## 12. Milestones

- **M0 — Engine validation:** `Game` interface + **Tic-Tac-Toe + Connect 4** (reference games; solved-position value tests). **Plus a synthetic "pass game" fixture** (or a Blokus position with one player blocked) — TTT/C4 strictly alternate and cannot exercise the player-aware backup, so the consecutive-move path needs a dedicated test from day one.

- **M1 — Blokus rules engine:** slow cell-grid oracle **first**, then the bitboard engine, **independently implemented**. Test battery:
  - Golden counts: orientations {1,2,6,19,63} = 91; 13,729 in-bounds; 414 per start square → 828 legal initial actions.
  - Encode/decode bijection over all in-bounds actions.
  - Differential fuzzing: bitboard vs. oracle on random playouts (move-gen, apply, terminal detection, scoring).
  - **Symmetry joint-permutation golden:** for every `g` in the Klein four-group × all 13,729 in-bounds actions (≈55k checks): decode → transform the cell set → re-encode (orientation id + bbox-top-left of the *transformed* cells) → assert equality with the precomputed `(g, a) → a′` table; assert the table is closed and bijective on the in-bounds set. Named failure mode under test: `g(anchor) ≠ anchor(g(cells))`.
  - Symmetry-equivariant move-gen on random states (property test; retained from v0.3).
  - **Perft(2)-by-opening:** oracle reply counts for all 828 openings, frozen as a golden hash (`opening_action_id → reply_count`); cross-check: counts constant on Klein-4 orbits of openings; totals give perft(2). (~11M legality checks; offline, one-time.) Plus golden perft totals at shallow depths.
  - Fast/slow scoring agreement; monomino-last cases; blocked-stays-blocked proof-by-test; forced-pass + scoring-flag normalization.

- **M1.5 — Abstraction test (firm): Othello.** Acceptance: implementing it touches only `games/` + `configs/` — **zero `core/` changes**. Beyond the diff check, it must exercise what Blokus can't:
  - **Explicit-pass convention:** flat 64+1 policy head (pass as an action id), vs. Blokus's auto-skip — both realizations of the §6.1 invariant, on real games.
  - **Pass-regain test:** a passed player later moves again — Othello passing is **non-monotone**, so this catches any `core/` code that quietly assumed blocked-stays-blocked.
  - **Consecutive-mover backup on a real game** (layered on M0's synthetic fixture, not replacing it).
  - Full **D4 symmetry group** (8 elements) exercised through `symmetry_group`; draws; strict-alternation-with-exceptions turn structure.

- **M2 — Encoding:** 46-plane state, `14×14×91` action, declared symmetry maps, sparse policy loss, overfit-one-batch test. **Plus the pipeline decode test:** for random `(state, sparse π)` and each declared `g`: build `(g·state, g·π)` via plane transform + action permutation, decode every `(action_id, count)` back to a cell set, and assert the multiset of `(cell-set, count)` equals `g` applied to the original — catches plane/channel wiring mismatches invisible to M1's table-level golden.

- **M2.5 — Micro-Blokus:** end-to-end learning on a reduced config to validate the loop cheaply.

- **M3 — Self-play baseline:** fixed-simulation actor + replay + learner on one GPU; version pinning; D10 move selection; D12 storage.

- **M4 — Evaluation:** frozen ladder + the §1 statistical protocol (paired bootstrap primary; Mann–Kendall secondary; per-checkpoint CIs).

- **M5 — Efficiency:** batched inference + playout-cap randomization at the D6 schedule.

- **M6 — Extensions (ordered levers; each gated on a profiled plateau):**
  1. FPU reduction (perspective caveat, §6.2).
  2. 8-fold post-opening augmentation (§8; judged by rung-7 Elo).
  3. Value-only storage of fast-search positions, iff the plateau is value-limited (D12).
  4. Global-pooling path for global features (§7).
  5. KataGo-style auxiliary-target bundle (D8).
  Design the extension points now; build nothing early.

- **M7 — (Future, optional) N-player engine:** exercises the Axis-1 seam for **4-player Blokus** — per-player vector value head, max-n (or paranoid) backup, tournament-based self-play and per-seat evaluation. Reachable because the interface is already N-player-ready; gated behind the proven 2-player abstraction (M1.5).

---

## 13. Risks

- **Rules-engine correctness (highest):** subtle corner/edge/pass/scoring bugs corrupt every downstream signal. Mitigation: independent oracle + the M1 battery (now including the joint-permutation golden and perft(2)-by-opening).
- **Throughput wall:** even short games may train slowly. Mitigation: PCR (D6/D12), sims tuning, batched inference (M5).
- **Evaluation fuzziness:** no perfect solver. Mitigation: multi-rung frozen ladder + paired games + the pre-registered §1 protocol.
- **Abstraction leakage:** game-specifics creeping into `core/`. Mitigation: declared-capability pattern + pass invariant (§6.1) + the M1.5 zero-`core/`-change acceptance test with non-monotone-pass coverage.
- **Plateau below target:** Mitigation: exhaust the ordered M6 levers before declaring the ceiling.

---

## Appendix A — Scope axes: pros / cons / disposition

**Principle:** widen to what AlphaZero natively handles (multiplayer, stochasticity) and stop before what needs a different algorithm (imperfect information).

**Axis 1 — Players: 2 → N.**
*Pros:* unlocks 4p Blokus, Chinese Checkers, etc.; interface cost near-zero (`current_player` returns an id; player-aware backup already generalizes alternation); 2p zero-sum is a strict special case, so nothing regresses.
*Cons:* the **solver** is where it hurts — backup needs an opponent model (max-n: honest but branching explodes, unstable equilibria; paranoid: tractable but pessimistically wrong); alliances/kingmaking aren't representable by independent per-player values (4p Blokus has strong "gang up on the leader" texture); self-play and Elo break (non-transitive cycles; needs tournaments/per-seat scoring).
*Disposition:* **interface N-player-ready now; multiplayer engine = M7.**

**Axis 2 — Determinism: deterministic → stochastic (perfect info).**
*Pros:* cheap generalization (transition → distribution / chance nodes; MCTS adds expectimax; value = expectation); reaches dice/tile games; leaving the seam open avoids a later refactor.
*Cons:* chance nodes add variance → more sims/decision, plus a pre- vs. post-chance encoding choice; most targets are deterministic, so payoff is thin unless a stochastic game is actually roadmapped (otherwise YAGNI).
*Disposition:* **seam marked at `apply` / node types; chance nodes deferred until roadmapped.**

**Axis 3 — Information: perfect → imperfect.**
*Pros:* unlocks a huge frontier class (poker, Stratego).
*Cons (disqualifying here):* it's a **different algorithm** — MCTS-over-states is invalid (nodes are information sets over a belief distribution); needs CFR / Deep CFR / ReBeL / public belief states (determinization is a hack with strategy-fusion and non-locality pathologies); it **undercuts the unified goal** (two solver families behind one interface); self-play/eval shift to exploitability at much higher compute.
*Disposition:* **walled off as a future sibling engine — not an extension of this codebase.**

**Axis 4 — Zero-sum → general-sum.** Mostly moot (targets are zero/constant-sum); a mild general-sum flavor is inherited for free with Axis 1, since alliances are general-sum. No separate v1 action.
