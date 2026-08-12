# M2.5 exit gate — verdict record

The pre-registered falsifiable exit test for the micro-Blokus instance, evaluated by
`scripts/micro_exit_gate.py` against a completed run of `configs/blokus_micro.json`.

Every bound below was pinned in the design doc (§5.3, §12 M2.5) and in the config file
**before** any run existed — see the `M2.5: pin the micro-Blokus instance and the gate
protocol doc-first` commit, which precedes every line of loop and gate code. The gate
script reads its thresholds from the config and has no way to override them from the
command line; the loss predicates are read from the persisted run record rather than
recomputed. That ordering is the whole point: the test could have failed, and nothing
about it was adjustable after the numbers came in.

## Run

| | |
|---|---|
| config | `configs/blokus_micro.json` |
| instance | `blokus_duo` / `MICRO_CONFIG` (5×5, pieces of order ≤ 3) |
| orientation hash | `78ea621ae2d1e27e239ecffa5ff44c793ef15f2884198a0394d394083d3e37e4` |
| run seed | 2500 |
| self-play | 2,000 games @ 64 sims, D7 root noise ε=0.25, α=10.8/#legal, D10 k_temp=4 |
| learner | 2,000 steps, batch 32, replay window 3,000, LR 0.02 warmup 200 + cosine 2000, λ_aux 0.25 |
| checkpoint | final (step 2000) |
| device | CPU |

## Verdict — PASS

| predicate | observed | bound | result |
|---|---|---|---|
| strength (score rate vs rung-1 uniform random) | **0.9525** | ≥ 0.70 | PASS |
| policy loss, tail mean / head mean | **0.5653** (1.7778 → 1.0050) | ≤ 0.70 | PASS |
| value loss, tail mean / head mean | **0.3159** (0.4225 → 0.1335) | ≤ 0.80 | PASS |

Strength detail: the rung-7 MCTS policy+value agent at 64 sims, root noise **off**,
argmax-N move selection, over 100 mirrored pairs (200 games) at eval seed 97531 with
start-square balancing. Total score 190.5 / 200. Anchored Elo **+512.4** against the
rung-1 anchor at 0 — recorded as observational detail, not as a predicate.

Loss detail: disjoint 200-step head and tail windows over 2,000 recorded steps. The
windows are required to be disjoint, so a truncated record cannot self-compare into a
vacuous ratio of 1.0; a record shorter than head+tail fails loudly (exit 2) instead of
scoring.

Integrity checks passed before any predicate was evaluated and any game was played:
run-record schema, the record's embedded config against the pre-registered file, the
evaluation protocol against the pinned agent forms, and game identity re-derived from
the config matching both the record's and the checkpoint's
`(game, game_config, orientation_hash)`.

Completeness checks passed in the same pass, against the counts the config pins rather
than against the windows: 2,000 recorded learner steps with ids exactly `0..1999`,
2,000 recorded self-play games with indices exactly `0..1999` (cross-checked against
the final step's `games_played`), and an evaluated `final` checkpoint at step 2,000 in
both the record entry and the checkpoint file itself. The disjoint-window guard alone
is a weaker bar — a record truncated to, say, 400 steps clears 200+200 and would be
*scored* — so the gate refuses anything that is not the whole pinned 2,000-step /
2,000-game protocol. Every expected value is read from `configs/blokus_micro.json`;
truncated or partial evidence is could-not-evaluate (exit 2), never FAIL (exit 1),
because a run that did not happen is not a run that fell short.

## What this does and does not establish

**Does.** The core AlphaZero learning dynamic works end to end in this codebase on a
real Blokus instance: self-play → sparse policy targets + D1 value/aux targets →
training → a network whose MCTS-guided play is far stronger than random. Both loss
heads fell substantially, and the strength result is a large margin over the floor
rather than a squeak past it, on a fixed paired set the training run never saw.

**Does not.** This is a 5×5 instance with 4 pieces and ~6-ply games. It says nothing
about whether the full 14×14 game is *reachable* — that is the throughput question,
answered separately in `m2_5-throughput-gate.md`, whose signed verdict still requires a
run on the RTX 4060 Ti. Beating uniform random is a low absolute bar; it is the right
bar for M2.5 (does learning happen at all?) and the wrong one for M5.5 (is the result
strong?). The ladder rungs above 1 are M4's job.

## Reproducing

```bash
python3 scripts/run_micro.py --config configs/blokus_micro.json --run-dir runs/blokus_micro
python3 scripts/micro_exit_gate.py --config configs/blokus_micro.json --run-dir runs/blokus_micro
```

Exit codes: 0 PASS, 1 FAIL, 2 could-not-evaluate. `runs/` is gitignored; the verdict is
also written to `<run-dir>/exit_gate_verdict.json`.
