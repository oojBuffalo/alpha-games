# M2.5 throughput go/no-go — micro-Blokus self-play spike [UNOFFICIAL / PROVISIONAL]

**UNOFFICIAL / PROVISIONAL — this is not the gate verdict.** §12 M2.5 pins the measurement to the RTX 4060 Ti 16 GB with CUDA + AMP; this run was on `cpu host`, so every number below is provisional and the verdict slot stays **PENDING**.

```
python3 scripts/bench_micro_throughput.py --allow-unverified-hardware --out docs/bench/m2_5-throughput-gate.md
```

- **Device:** cpu host (`cpu`), AMP off (no-op off CUDA)
- **torch:** 2.9.0 (CUDA n/a)
- **Date:** 2026-08-12
- **Config:** `configs/blokus_micro.json` — run_seed 2500, fresh weights (net-init seed 13521923826590675288), 64 sims/move, batch-1 leaf inference.
- **Protocol:** 50 warm-up games discarded, then 200 measured games (§12 M2.5's pinned interval).
- **Game identity:** blokus_duo / MICRO_CONFIG, orientation hash `78ea621ae2d1e27e239ecffa5ff44c793ef15f2884198a0394d394083d3e37e4`.

## The pinned predicate

Pre-registered in §12 M2.5 and carried in `configs/blokus_micro.json` (`throughput`), fixed before any run:

```
games_per_hour_full = 3600 * E / (r * S * P)
GO iff games_per_hour_full >= 100
```

with `S = 128` (M3's fixed sims), `P = 35` (assumed full-game plies), `E` the measured micro net-evals/sec, and `r = t_full / t_micro` the measured batch-1 forward-time ratio of the two nets on the same device. A NO-GO routes back to the design doc — sims budget, config size, or pulling M5 levers forward — before M3 starts; it is never a reason to move the floor.

## Method

A dedicated spike, not a training run: `scripts/run_micro.py`'s exact pacing (one self-play game, then one learner step drawing from the same replay window) with that module's learner step imported rather than re-implemented, freshly initialized weights (self-play throughput is weight-independent), and nothing persisted. Net evaluations and leaf legal-set sizes are counted by a wrapper around the evaluator — no `core/mcts.py` change; M3's observability task formalizes these counters in `core/metrics.py` and these script-local ones do not satisfy it. Warm-up games are played and discarded, then the counters are reset and the clock started.

## Measured — micro loop (UNOFFICIAL / PROVISIONAL)

| quantity | value |
|---|--:|
| end-to-end games/hour | 13,210.3 |
| mean plies/game | 6.14 |
| sims/sec (end-to-end) | 1,442.0 |
| net-evals/sec (end-to-end) — **E** | 465.9 |
| net-evals/sec (self-play time only) | 646.8 |
| learner steps/sec | 3.67 |
| net-evals per sim | 0.323 |
| mean legal-set size at evaluated leaves | 6.29 |
| wall clock, measured interval (s) | 54.5 |
| — self-play share | 72.0% |
| — learner share | 27.7% |

The per-phase split is what makes a NO-GO diagnosable: a learner-dominated interval points at the M3 actor–learner split, a self-play-dominated one at batched inference (the M5 lever) or the sims budget.

## Micro:full ratio table

| quantity | micro | full 14×14 | full:micro |
|---|--:|--:|--:|
| board cells | 25 | 196 | 7.84× |
| input planes | 12 | 46 | 3.83× |
| raw actions | 225 | 17,836 | 79.27× |
| mean legal-set size (random playouts) | 14.0 | 232.1 | 16.58× |
| mean plies/game (random playouts) | 5.8 | 28.2 | 4.91× |
| batch-1 forward (ms, measured) | 1.392 | 2.621 | 1.88× |

`r = 1.883` is the last row's ratio — median of 50 timed batch-1 forwards per net (after warm-up) on the same device, both nets carrying the identical D5 8×128 trunk (§5.3 keeps the trunk unchanged so exactly this number transfers).

## Scaling model, and where it is weak

`E` is measured **end-to-end** — self-play and the interleaved learner steps — so the loop's non-network cost is already inside it. `r` then rescales the per-simulation **network** cost from the micro net's `12×5×5 → (5,5,9)` shape to the full net's `46×14×14 → (14,14,91)` shape, and `S`/`P` substitute M3's sim count and the assumed full-game length. One net evaluation per simulation is assumed (batch-1 leaf inference — the known M2.5/M3 configuration, recorded here as the M5 lever rather than optimized); the measured net-evals-per-sim above is the check on that.

Five assumptions carry the projection, and **the first four all lean optimistic** — the true full-game rate is likely below the projected number:

1. **Dividing the whole loop's cost by `r` alone.** `E` contains tree descent, move generation, `apply`, state encoding and the learner step, none of which scale like the network. The full game's non-network cost grows much faster than `r` — see the ratio table: 79× the raw actions, ~17× the mean legal-set size (828 legal openings at the root vs. 42), ~8× the board cells, ~4× the planes encoded per leaf, against an `r` of 1.88. **This is the weakest assumption in the gate.**
2. **`r` measured at batch 1 on a GPU is latency-bound.** Both forwards are dominated by kernel-launch overhead there, so the measured `r` can sit near 1 while the true compute ratio is several-fold.
3. **The learner step is not rescaled at all.** The micro step is batch 32 on 5×5; M3's is batch 256 on 14×14. Folding the micro learner into `E` and then dividing by `r` understates the real learner share.
4. **`P = 35` is an assumption, not a measurement.** The measured full-game random-playout mean plies is in the ratio table as a check; the projection scales inversely with `P`.
5. **Net-evals-per-sim differs between the two boards** — direction ambiguous, unlike 1–4. The micro tree is small enough that many simulations end in an already-expanded or terminal node and evaluate nothing, so the measured ratio above (0.323) sits below 1; the full game's tree is nowhere near exhausted at 128 sims, so its ratio is ≈1. `E` therefore carries more non-network work per evaluation than the full game will, while the full game's per-sim non-network work is itself far larger.

Consequence for reading the result: a projection comfortably **above** the floor is weaker evidence than it looks, and a projection **below** it is strong evidence — the optimistic model still failed.

## Projection

```
E = 465.9032 net-evals/s
r = 1.8834
S = 128   P = 35
games_per_hour_full = 3600 * 465.9032 / (1.8834 * 128 * 35) = 198.78
floor = 100
```

## Verdict

**NO VERDICT (UNOFFICIAL / PROVISIONAL) — the arithmetic would read GO, but this run was not on the pinned RTX 4060 Ti 16 GB: projected 198.8 full-game games/hour vs. a floor of 100 (E = 465.9 net-evals/s, r = 1.883, S = 128, P = 35).**

### RTX 4060 Ti verdict — PENDING

The signed verdict requires one run on the pinned hardware. Until then this section is a placeholder and nothing above is gate evidence.

| field | value |
|---|--:|
| device | _pending — RTX 4060 Ti 16 GB, CUDA + AMP_ |
| date | _pending_ |
| E (net-evals/s) | _pending_ |
| r (t_full / t_micro) | _pending_ |
| projected full-game games/hour | _pending_ |
| floor | 100 |
| verdict | _pending — GO / NO-GO_ |

Fill it by running, on the 4060 Ti box:

```
python3 scripts/bench_micro_throughput.py --out docs/bench/m2_5-throughput-gate.md
```

which refuses to run anywhere else, rewrites this file in place with the official heading, and exits 0 on GO / 2 on NO-GO.
