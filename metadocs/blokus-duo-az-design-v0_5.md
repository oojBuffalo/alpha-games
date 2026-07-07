# AlphaZero × Blokus Duo — Design Doc

**Status:** v0.5 — decision-complete; milestone plan re-scoped. The pinned decisions **D1–D12** are unchanged and the success criterion (§1) stays pre-registered; §11 holds only an optional stretch goal. Three previously-unstated config *scalars* — the aux loss weight `λ_aux` (§7), the weight-publish interval / checkpoint count *K* (§6.2/§1), and the mirrored game-pairs per (checkpoint, rung) cell (§9) — are **not** among D1–D12 and are flagged *to pin doc-first* at their milestones (M2/M3/M4). Buildable as written.
**Changes within v0.5 (M1 convention pins):** §5.1 gains a "Convention pins (M1)" block freezing the operational encoding conventions the M1 fixtures and replay keys depend on — 0-indexed coordinates, cell-major flatten `(r*14+c)*91+o`, the bbox-top-left anchor operationalized, deterministic orientation-ID assignment, and the per-piece-boundary sha256 hash serialization. §12 M1 additionally pins the adapter's `symmetry_group` plane-transform slot as a raising sentinel until M2 (M1 fully owns only the action-permutation side). No decision changes; D1–D12 untouched.
**Changes within v0.5 (M1.6 convention pins):** §12 M1.6 gains a "Convention pins (M1.6)" block freezing the operational definitions its frozen-ladder rungs, pair protocol, and Elo scaffolding build against — rung 1 uniform-random (core), rung 2 largest-piece via the adapter's `decode_action` cell count (Blokus-side, seeded ties), rung 3 as an interface-generic 1-ply mobility greedy (terminal-utility priority; opponent-reply minimization; own-reply bonus when the opponent is blocked), the mirrored-pair protocol (seats swapped per pair, per-pair seed, draws 0.5, Blokus start-square balancing via an adapter-side opening-balancer hook, one emitted record per pair as the future §1 resampling unit), and the anchored Bradley–Terry Elo fit on the 400-point scale with rung 1 pinned at 0 and one virtual draw per unordered matchup. No decision changes; D1–D12 untouched.
**Changes from v0.4:** Milestone-plan (§12) revision only — the pinned decisions D1–D12, the §1 success criterion, and the §2 scope boundary are unchanged. §12 now names the core PUCT MCTS engine as an explicit M0 deliverable (with a uniform-prior config flag serving ladder rung 6 — no leaf-evaluator abstraction) and adds to M0 an MCTS-vs-minimax correctness oracle, a subtree-reuse/virtual-loss invariant test, an envelope-rejection negative test, and the test-runner/CI entrypoint. A new M1.6 pulls the network-free ladder rungs 1–3 plus the paired/mirrored game runner forward as an early absolute-strength anchor; rung 4 (UCT + random rollouts) moves to M3, and network rungs 5–8 stay in M4. M1 gains a value_targets score→z→aux golden (incl. sign(0)=0 and the |score_diff|≤109 range) and explicit ownership of action encode/decode, the (g,a)→a′ table, fixture generation, and the orientation-hash write-side; M2 gains the explicit D5 network build + train step; M2.5 gains a doc-first reduced-config definition, a config-parameterized correctness net, a falsifiable exit test, and an early throughput go/no-go. M3 is re-scoped as a fixed-128-sim functional/correctness baseline and gains observability, the actor–learner IPC mechanism, the replay on-disk schema, the checkpoint schema with orientation-hash validate-on-load, run seeding, and a second zero-core-diff Othello re-check through the full stack; D12's fast-position drop-policy moves to M5 (with a numeric throughput target and a virtual-loss correctness test). M4 is made concurrent with the run, defines the "profiled plateau" rule, asserts the hash before rung 8, and pins a bootstrap seed. A new M5.5 is the explicit production run that accumulates the K checkpoints and computes the §1 verdict. Three previously-unstated config scalars (λ_aux → M2/§7; publish interval / K → M3/§6.2; pairs-per-cell → M4/§9) are flagged to pin doc-first; none was ever among D1–D12. M1's battery, M1.5 Othello, the M6 levers, and M7 are intact.

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
- **Orientation-ID determinism:** assign IDs by sorting canonical cell tuples — **never** Python set-iteration order. Serialize an orientation-table hash into every checkpoint and replay dataset, and **validate it on load** (recompute and fail loudly on mismatch) — write-side owned by M1, read-side by M3 (§12); serializing without checking is inert.

> **Verified counts (independently reproduced) — hardened as golden tests in M1:** orientations per size {1,2,6,19,63} = **91**; in-bounds placements on 14×14 = **13,729**; placements covering each start square = **414** (→ **828** legal initial actions).

#### Convention pins (M1) — frozen into fixtures and replay keys

- **Coordinates:** 0-indexed `(r, c)`, row-major, internally and in all fixtures; start squares **(4,4)** and **(9,9)**. (§4's (5,5)/(10,10) is the same pair in this doc's 1-indexed display convention.)
- **Flatten order:** `action_id = (r*14 + c) * 91 + o` — cell-major (HWC), matching the literal `14×14×91`; decode = `divmod(action_id, 91)` then `divmod(cell, 14)`. M2 pays one tensor `permute` before the sparse gather; there is no perf argument for channel-major.
- **Anchor (D2, operational):** the board cell where the origin-normalized orientation's bounding-box top-left lands; a placement with an `h×w`-bbox orientation is in-bounds iff `r + h ≤ 14 ∧ c + w ≤ 14`.
- **Orientation IDs:** per piece, generate the 8 D4 images, origin-normalize (translate min row / min col to 0), take canonical rep `tuple(sorted(cells))`, dedupe, sort lexicographically. Pieces are ordered by `(size, lexicographically-least canonical orientation)`; global orientation ids 0–90 are assigned in that traversal order. This piece order also fixes the M2 inventory-plane order (D3).
- **Orientation-table hash:** sha256 over a canonical serialization that preserves **per-piece boundaries** (never a flat orientation list), so piece↔orientation regroupings cannot collide. Fixture files embed `{orientation_hash, conventions: {axis_order, flatten, anchor, board_size, start_squares}}` — the hash alone does not cover the flatten convention.

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
- **Version pinning:** publish weights every fixed number of learner steps; pin one model version per self-play game; never swap mid-search; record `model_version` per replay sample; reuse the chosen subtree between moves. *(The publish interval — learner steps per publish — and the resulting checkpoint count `K` used by §1's Δ are unpinned scalars: **to pin doc-first at M3**. `model_version` provenance is distinct from the §5.1 orientation-table hash.)*

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
- **Loss:** `l = (z − v)² − πᵀ log p + λ_aux·(aux term) + c‖θ‖²`. The aux-term weight `λ_aux` is an unpinned config scalar — **to pin doc-first at M2** (§12); the KataGo lineage implies a small value, but none is committed (it is *not* an implicit 1.0, and *not* one of D1–D12).
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

**Protocol:** disable Dirichlet; deterministic root (argmax N); alternate first player and balance the start-point choice; **paired/mirrored games** — the resampling unit for §1's bootstrap (the number of mirrored pairs per (checkpoint, rung) cell governs CI width and is an unpinned scalar, distinct from §1's B ≈ 2,000 bootstrap replicates — **to pin doc-first at M4**); draws scored 0.5; report per-checkpoint confidence intervals; freeze baseline versions. The pre-registered Δ contrast (§1) is the primary criterion; the Mann–Kendall test is the secondary; per-checkpoint monotonicity is not required. (For the future N-player engine, 2p Elo is replaced by tournament / per-seat scoring — see M7.)

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

The list is a topological order, not a strict serial chain; two branches run off the critical path and are called out inline (Othello depends only on the M0 interface; the M1.6 baseline opponents depend only on the M1 rules engine). "To pin doc-first" markers name a scalar left unpinned by D1–D12 that must be fixed in the doc before the owning milestone builds against it.

- **M0 — Engine validation + core search skeleton:**
  - `Game` interface + **Tic-Tac-Toe + Connect 4** (reference games; solved-position value tests).
  - **Core PUCT MCTS engine (explicit deliverable, §6.2):** the sparse, player-aware search tree — selection/PUCT with the D11 growth schedule, first-play-urgency `Q = 0`, sparse `{N,W,Q,P}` over legal actions, player-aware backup, subtree reuse between moves. There is **no leaf-evaluator abstraction**: the leaf value is `terminal_utility` at terminals and (from M2) the network. Ladder rung 6 ("uniform-prior MCTS with network value", §9) needs only a **uniform-prior config flag** (prior source) on this engine — not a swappable evaluator; rung 4 (UCT + random rollouts) is a separate standalone opponent built at M3, not a configuration of this engine.
  - **Synthetic "pass game" fixture** (or a Blokus position with one player blocked) — TTT/C4 strictly alternate and cannot exercise the player-aware backup, so the consecutive-move path needs a dedicated test from day one.
  - **MCTS-vs-minimax correctness oracle:** on the fully-solved TTT/C4 already present, high-sim MCTS must recover the known minimax move/value — the search analogue of "oracle first" (§13), since a search sign/perspective bug corrupts every downstream signal exactly like a rules bug. Plus a **subtree-reuse / virtual-loss invariant test** (no stale or duplicated stats carried across moves).
  - **Envelope-rejection negative test (§6.1):** a deliberately mis-declared stub adapter (`num_players = 3`, or `is_stochastic = True`) must trip the v1-envelope assertion and **fail loudly** — the "asserted in code, not just prose" scope boundary needs a test that the assertion actually fires (M1.5's zero-`core/`-diff check proves the *positive* abstraction claim, a different property).
  - **Test-runner / CI entrypoint** stood up here (the M0/M1 batteries run under it); record the invocation in CLAUDE.md §Environment once it lands.

- **M1 — Blokus rules engine:** slow cell-grid oracle **first**, then the bitboard engine, **independently implemented**. Test battery:
  - Golden counts: orientations {1,2,6,19,63} = 91; 13,729 in-bounds; 414 per start square → 828 legal initial actions. **Added to the golden set (they bound the D1 targets):** 89 squares/set over orders {1,1,2,5,12}; per-player score ∈ [−89, +20]; score diff ∈ [−109, +109].
  - Encode/decode bijection over all in-bounds actions. **Ownership boundary (resolves the M1/M2 overlap):** M1 owns `encode_action`/`decode_action` **and** the geometric `(g, a) → a′` symmetry-permutation table; M2 owns the 46-plane state encoding and the training-side wiring of symmetry into targets/augmentation. Until M2 lands the plane encoding, the adapter's declared `symmetry_group` fills the plane-transform slot with a **raising sentinel** (`NotImplementedError`) rather than an engine-state transform — the action permutations are usable now, and the plane side fails loudly instead of advertising the wrong representation; engine-state-tuple transforms remain a module utility (`games/blokus_duo/symmetry.state_transform`) for the M1 equivariance tests.
  - Differential fuzzing: bitboard vs. oracle on random playouts (move-gen, apply, terminal detection, scoring).
  - **Symmetry joint-permutation golden:** for every `g` in the Klein four-group × all 13,729 in-bounds actions (≈55k checks): decode → transform the cell set → re-encode (orientation id + bbox-top-left of the *transformed* cells) → assert equality with the precomputed `(g, a) → a′` table; assert the table is closed and bijective on the in-bounds set. Named failure mode under test: `g(anchor) ≠ anchor(g(cells))`.
  - Symmetry-equivariant move-gen on random states (property test; retained from v0.3).
  - **Perft(2)-by-opening:** oracle reply counts for all 828 openings, frozen as a golden hash (`opening_action_id → reply_count`); cross-check: counts constant on Klein-4 orbits of openings; totals give perft(2). (~11M legality checks; offline, one-time.) Plus golden perft totals at shallow depths.
  - Fast/slow scoring agreement; monomino-last cases; blocked-stays-blocked proof-by-test; forced-pass + scoring-flag normalization.
  - **value_targets golden (D1) — new:** the score→z→aux mapping is thin but load-bearing (it produces every training target). Assert `z = sign(score_diff)` including **`sign(0) = 0`** — a two-way `1 if diff > 0 else −1` silently relabels every Blokus draw as a loss, and §4 says `z = 0` exists — plus `aux = score_diff / 109` and the range invariant `|score_diff| ≤ 109` over reachable terminals (so aux ∈ [−1, 1]). M0's TTT draw test covers core `z = 0` but not this adapter mapping.
  - **Fixture generation + orientation-hash (write-side, §5.1) — new:** the checked-in golden fixtures (the perft(2) `opening_action_id → reply_count` hash and the `(g, a) → a′` table) are artifacts owned here — generated by scripts, stored, and each **version-bound to the orientation-table hash** it was computed under. M1 defines the orientation table and **serializes its hash** into checkpoints and replay datasets. (Read-side validation is M3.)

- **M1.5 — Abstraction test (firm): Othello.** *(Depends only on the M0 `Game` interface, not on any Blokus code — schedule in parallel with M1 rather than strictly after it; retiring abstraction leakage early is cheaper than after M1 has hardened the interface.)* Acceptance: implementing it touches only `games/` + `configs/` — **zero `core/` changes**. Beyond the diff check, it must exercise what Blokus can't:
  - **Scope note (forward dependency, made explicit):** M1.5 stands up Othello's own encoding surface — plane transforms, `encode_action`/`decode_action`, and `policy_shape` for the flat 64+1 head — which is M2-class machinery pulled forward for the second game. Pinned here: **M1.5 carries its Othello encoding** (the alternative is to reorder M1.5 after M2).
  - **Explicit-pass convention:** flat 64+1 policy head (pass as an action id), vs. Blokus's auto-skip — both realizations of the §6.1 invariant, on real games.
  - **Pass-regain test:** a passed player later moves again — Othello passing is **non-monotone**, so this catches any `core/` code that quietly assumed blocked-stays-blocked.
  - **Consecutive-mover backup on a real game** (layered on M0's synthetic fixture, not replacing it).
  - Full **D4 symmetry group** (8 elements) exercised through `symmetry_group` — **including the pass-id fixed-point check**: pass (action id 64) must be a fixed point of every D4 action-permutation, and the table stays closed/bijective on the 65-action head. Also draws; strict-alternation-with-exceptions turn structure.
  - **Coverage limit (drives the M3 re-check):** M1.5 validates only the M0/M1-era `core/` surfaces (interface + rules-level search). The larger surfaces — encoding pipeline (M2) and self-play/replay/training (M3) — do not yet exist, so the "second game leaves `core/` unchanged" guarantee is **re-run at M3** through the full stack.

- **M1.6 — Baseline opponents + game runner (NEW; absolute-strength anchor):** the network-free ladder rungs of §9 — (1) uniform random, (2) largest-piece/coverage heuristic, (3) mobility heuristic — plus the **paired/mirrored game runner** (alternate first player, balance the start-point choice, draws scored 0.5) and the anchored-Elo scaffolding (rung 1 fixed at 0). Depends only on M1's rules engine; gives M2.5/M3 a cheap "does self-play beat random yet?" measurement before the full M4 harness exists. **Rung 4 (UCT + random rollouts) is *not* here** — it needs the M0 search engine wired to a rollout evaluator and lands with M3. Network rungs 5–8 and the §1 statistical protocol stay in M4, which extends this runner/scaffolding.
  - **Convention pins (M1.6) — the frozen operational definitions of the network-free rungs, the pair protocol, and the Elo fit:**
    - **Rung 1 (uniform random):** uniform over `legal_moves`, seeded. Interface-generic; lives in `core/`.
    - **Rung 2 (largest-piece/coverage, Blokus-specific):** argmax placed-cell count, computed through the adapter surface only (`len(decode_action(a))` — no engine internals); uniform-random among ties (seeded). Lives in `games/blokus_duo/`.
    - **Rung 3 (mobility, interface-generic):** 1-ply greedy over successors `s' = apply(s, a)`. Terminal `s'` scores by the mover's `terminal_utility` with absolute priority (a win now beats any mobility score; a loss is worst). Nonterminal `s'` scores `−|legal(s')|` when the next mover is the opponent, `+|legal(s')|` when the mover moves again (opponent blocked/skipped — only observable through the interface's `current_player`). Uniform-random among ties (seeded). Lives in `core/` (uses only the `Game` interface).
    - **Pair protocol:** a mirrored pair is two games between the same agents with **seats swapped**, driven by a per-pair seed; draws score 0.5. **Start-point balancing (Blokus):** the second game's opener is restricted to openings covering the **same start square** the first game's opener actually covered — realized as a game-specific *opening-balancer hook* on the generic runner (the runner stays game-agnostic; the balancer lives with the adapter). The runner emits one record per pair — the resampling unit §1/§9's bootstrap will consume at M4.
    - **Anchored Elo (scaffolding):** Bradley–Terry/logistic on the standard 400-point scale (`E = 1/(1+10^{−Δ/400})`), fit by coordinate ascent over per-matchup aggregate scores, with rung 1 pinned at 0. One **virtual draw per unordered matchup** keeps ratings finite on extreme small samples (a deliberate small-sample conservatism; M4's pre-registered protocol supersedes this scaffolding). Pure stdlib.

- **M2 — Encoding + network build:**
  - **Encoding:** 46-plane state, `14×14×91` action, declared symmetry maps (wiring the M1-owned `(g, a) → a′` table into the training-target/augmentation transforms — M2 owns the **state-plane** side), sparse policy loss, overfit-one-batch test. **Plus the pipeline decode test:** for random `(state, sparse π)` and each declared `g`: build `(g·state, g·π)` via plane transform + action permutation, decode every `(action_id, count)` back to a cell set, and assert the multiset of `(cell-set, count)` equals `g` applied to the original — catches plane/channel wiring mismatches invisible to M1's table-level golden.
  - **Network build (D5) — explicit deliverable (new):** the overfit-one-batch test presupposes the whole model, so name it. 8×128 residual trunk; three heads (policy 91 spatial channels, scalar `tanh` value, normalized score-diff aux); the composite loss `l` (§7) with the aux term; SGD-momentum + AMP optimizer; one train step. *(Unpinned: the aux-term weight `λ_aux` — **to pin doc-first at M2, §7**; the overfit test needs a value.)*

- **M2.5 — Micro-Blokus:** end-to-end learning on a **reduced Blokus instance** to validate the loop cheaply and as an **early throughput feasibility gate** before the expensive M3 build.
  - **Reduced config is currently undefined and must be pinned doc-first before M2.5:** none of the full-game golden constants carry over (91 orientations, 46 planes, 13,729 placements, 828 openings, 196-bit board — §5/§6.3). The micro instance needs its own board size / piece subset / start squares, with its orientation table, plane count, and `policy_shape` **derived** from that choice.
  - **Correctness safety net (new):** M1's oracle + differential battery is 14×14-specific and does not protect the reduced config — extend the oracle + differential test to be **config-parameterized** so a micro-config legality/scoring bug can't silently corrupt the loop (the §13 top risk, otherwise off the battery).
  - **Falsifiable exit test (new):** "validate the loop" gets a pass condition — micro-config policy/value loss drops below a set threshold **AND** the trained net beats uniform-random at ≥ X% over a fixed paired set (the M1.6 runner mechanism on the micro board). This milestone gates the far more expensive M3, so it must pass/fail objectively.
  - **Throughput spike + go/no-go (new):** measure end-to-end games/GPU-hour on the micro loop and record a go/no-go before committing the full M3 build — throughput is §3's binding constraint and must not be first verified only at M5.
  - M2.5's loop is a deliberate minimal precursor to M3's hardened, version-pinned baseline (it may share M3 code), not a throwaway.

- **M3 — Self-play baseline (functional / correctness; fixed-sim):** actor + replay + learner on one GPU at a **fixed 128 sims** (the D6 validate tier — *not* the PCR schedule, which is M5). Scope is correctness and end-to-end wiring, not throughput.
  - **D10 move selection** live; **storage schema** live — sparse π + z + aux per stored sample. *(D12's fast-position **drop-policy** is **not** exercisable here: at one fixed sim count there is no fast tier to drop; the drop-policy moves to M5 with PCR. M3 builds the all-stored fixed-sim case only.)*
  - **Rung 4 (UCT + random rollouts)** wired here (needs the M0 search engine + a rollout evaluator); registered as a frozen ladder opponent alongside the M1.6 rungs.
  - **Observability (acceptance criteria; captured from step zero of the run — unrecoverable post-hoc):** cumulative **network-evaluation** and **GPU-hour** counters (the §1 plot x-axes) and throughput metrics (games/hour, sims/sec, learner-steps/sec — the §13 wall). The §1 primary criterion and every M6 "profiled plateau" gate structurally depend on these being recorded live.
  - **Actor–learner concurrency / IPC (the §2 seam, made concrete):** the weight-publication path (how learner-published weights, §6.2, reach the actor), the sample-flow path (actor → replay buffer), and single-GPU sharing between the inference actor and the training learner. The doc asserts the split as a principle; M3 produces the artifact.
  - **Replay dataset artifact:** on-disk schema/format (position serialization, sharding, the 250k ring-buffer eviction policy) and the sampling machinery realizing the D5 2–4 samples/stored-position ratio — the shared artifact across the actor–learner split, not a black box.
  - **Checkpoint schema + orientation-hash validate-on-load (§5.1 read-side):** define what a checkpoint bundles (weights + orientation-table hash + `model_version` + config + optimizer state for crash-resume on a long single-GPU run) and **validate on load** — recompute the orientation-table hash and **fail loudly on mismatch** for both replay shards and checkpoints. This is the whole point of serializing the hash (M1 write-side); `model_version` provenance (§6.2) is a separate quantity.
  - **Reproducibility / seeding:** a recorded run seed plus a config field seeding every stochastic component — net init, Dirichlet (D7), ∝N sampling (D10), MCTS tie-breaks, symmetry-augmentation choice (D9), replay sampling; parallel actors are seed-decorrelated so they do not generate identical games.
  - **Second zero-`core/`-diff re-check (abstraction leakage, §13):** now that the encoding + self-play + training surfaces exist, re-run the M1.5 guarantee — drive **Othello** through the full encoding + self-play + training stack with **zero `core/` changes**. M1.5 alone could only cover the M0/M1-era surfaces.

- **M4 — Evaluation harness (stood up before the production run and run *concurrently* with it):** the frozen-ladder network rungs — (5) network policy, no search; (6) uniform-prior MCTS with network value (the M0 uniform-prior config flag); (7) full policy-and-value MCTS; (8) historical checkpoints — plus the §1 statistical protocol (paired bootstrap primary; Mann–Kendall secondary; per-checkpoint CIs), extending the M1.6 runner/scaffolding.
  - **Coupled to the run, not strictly downstream (new):** the §1 Δ contrast needs checkpoints scored (or their mirrored-pair games logged at per-(checkpoint, rung) granularity) *as they are produced*, so the harness + its paired-result logging schema must be live from the first production checkpoint — otherwise the early-third checkpoints are unscored/incomparable and Δ is unreconstructable without a re-run.
  - **Before loading historical checkpoints (rung 8):** assert the orientation-table hash matches (the M3 read-side path).
  - **Plateau-detection rule (operationalize the M6 gate; new):** define "profiled plateau" quantitatively — e.g. Mann–Kendall over the last *M* checkpoints non-significant **AND** the Δ-CI width below a threshold over a stated GPU-hour window — so every M6 lever go/no-go and the ceiling declaration (§13) is falsifiable.
  - **Bootstrap seed:** the B ≈ 2,000-replicate resample carries its own recorded seed. *(Unpinned: the number of mirrored **game pairs per (checkpoint, rung) cell** — it governs CI width and is distinct from B — **to pin doc-first at M4, §9**.)*

- **M5 — Efficiency:** batched inference + **playout-cap randomization at the D6 schedule** (512 / 256 / 64), and **D12's fast-position drop-policy** (moved here from M3 — the "drop the fast tier" rule is defined over the PCR tiers, which first exist now).
  - **Numeric throughput target (acceptance; new):** pin a games/GPU-hour (or wall-clock-to-fill-the-250k-position window) success number so M5 can pass/fail — §3's "many games per GPU-hour" feasibility claim must be verified, not assumed.
  - **Virtual-loss / in-flight-leaf correctness (§6.2):** now that batched parallel selection is active, test that concurrent traversals diverge to distinct leaves and virtual loss is fully backed out — a correctness property (it distorts N, the D10 training target), not just throughput.

- **M5.5 — Production run + primary-criterion verdict (NEW; the actual experiment):** run the M5-efficient, PCR-scheduled self-play loop to accumulate the *K* version-pinned checkpoints (§6.2), with M4's eval + logging live throughout; then compute the §1 Δ paired-bootstrap contrast and the Mann–Kendall secondary and render the **pre-registered verdict** (Δ 95% CI strictly above 0). The M6 levers interleave with this run — each pulled only on an M4-defined profiled plateau. *(Unpinned upstream: the §6.2 **weight-publish interval** (learner steps per publish) and the resulting checkpoint count *K* (§1) — **to pin doc-first at M3, §6.2**.)*

- **M6 — Extensions (ordered levers; each gated on the M4-defined profiled plateau):**
  1. FPU reduction (perspective caveat, §6.2).
  2. 8-fold post-opening augmentation (§8; judged by rung-7 Elo).
  3. Value-only storage of fast-search positions, iff the plateau is value-limited (D12).
  4. Global-pooling path for global features (§7).
  5. KataGo-style auxiliary-target bundle (D8).
  Design the extension points now; build nothing early.

- **M7 — (Future, optional) N-player engine:** exercises the Axis-1 seam for **4-player Blokus** — per-player vector value head, max-n (or paranoid) backup, tournament-based self-play and per-seat evaluation. Reachable because the interface is already N-player-ready; gated behind the proven 2-player abstraction (M1.5 + the M3 full-stack re-check).

---

## 13. Risks

- **Rules-engine correctness (highest):** subtle corner/edge/pass/scoring bugs corrupt every downstream signal. Mitigation: independent oracle + the M1 battery (now including the joint-permutation golden, perft(2)-by-opening, and the value_targets score→z→aux golden).
- **Search correctness (co-highest):** player-aware backup, virtual loss, and subtree reuse (§6.2) are sign- and race-prone and corrupt every downstream signal like a rules bug. Mitigation: the M0 MCTS-vs-minimax oracle on solved TTT/C4 + subtree-reuse/virtual-loss invariant tests (M0; virtual-loss correctness re-checked at M5).
- **Throughput wall:** even short games may train slowly, and throughput is §3's binding constraint. Mitigation: an **early feasibility gate** (M2.5 throughput spike + go/no-go) ahead of the full build, then PCR (D6/D12), sims tuning, and batched inference against a **numeric games/GPU-hour target** (M5).
- **Evaluation fuzziness:** no perfect solver. Mitigation: multi-rung frozen ladder + paired games + the pre-registered §1 protocol.
- **Abstraction leakage:** game-specifics creeping into `core/`. Mitigation: declared-capability pattern + pass invariant (§6.1) + an envelope-rejection negative test (M0) + the zero-`core/`-change acceptance test run **twice** on Othello — at M1.5 for the interface/rules surfaces and **again at M3** through the full encoding + self-play + training stack (the surfaces that do not yet exist at M1.5).
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
