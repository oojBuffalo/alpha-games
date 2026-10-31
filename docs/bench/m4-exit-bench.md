# M4 exit bench — production-`S` eval throughput vs. publish cadence

The §9 pin 11 "M4 exit number": real seconds/game of the eval harness at the pinned
production search budget, projected onto one checkpoint's full required cell set and
compared against the watched run's own publish cadence. Measured, not asserted — and
on this first measurement it comes out **infeasible by ~20×**.

```
python3 scripts/run_eval.py <eval-config> --bench
```

- **Date:** 2026-10-22
- **Code:** `feature/m4-eval-orchestrator` @ `74dd8e4`; `protocol_version = 2`, protocol fingerprint `365cb8117cd390ed…`
- **Eval device:** `cpu` — Intel Core i7-8700K (6 cores / 12 threads); torch 2.7.0+cu126, 6 intra-op threads. The bench is a single process.
- **Eval config:** `configs/m4_eval.json` verbatim except `run_dir` pointed at the concrete run below — `pairs_per_cell` 24, `eval_sims` 512, forms `[5, 6, 7]`, `eval_seed` 271828, `rung8_lag_divisor` 4, `rung8_earliest_version` 1.
- **Game identity:** `blokus_duo` / `FULL_CONFIG`, orientation hash `4408e7d7b56b3533685ce92e88fd1bc9453ba405f4afacfa96e431974446cb35`.

## The watched run

The bench needs a **schema-v2** published member checkpoint (recorded `network_config` +
`artifact_kind`; §12 M4 provenance pin). The existing M3 acceptance run
(`blokus_duo-20260823T142412117771-…`, 30 checkpoints) is schema v1 and the loader rejects
it by design — there is deliberately no v1→v2 migration path, since the pin exists so the
architecture is *recorded*, never inferred. A fresh bounded run was therefore launched
from this branch with `configs/blokus_duo.json` (every material field identical; `run_dir`
made absolute) and `--max-learner-steps 400`:

| | |
|---|---|
| run | `runs/blokus_duo/blokus_duo-20261022T070444222635-1fa824aa4f67` |
| learner | 400 steps (cap), batch 256, 4 actors @ 128 sims, `cuda` (RTX 4060 Ti 16 GB) |
| published members | `ckpt-1` (step 200, 08:16:51 UTC), `ckpt-2` (step 400, 09:24:33 UTC) |
| publish cadence | **1.128 h** (one gap; the Aug-23 full run averaged 1.26 h over 30) |
| games to ckpt-2 | 908 self-play games; ≈ 30 plies/game from the actor counters |
| candidate | `ckpt-2` — loss_total 5.04, loss_policy 4.37, loss_value 0.67 (a weak, early net) |

Operational note, not evidence: the learner exits at its step cap but the actors have no
cap and only stop when signalled, so the run kept playing (to 5,103 games / 5.5 GB of
shards) until the actors were sent SIGTERM ~11 h later. A bounded run should pass **both**
`--max-learner-steps` and `--max-games-per-actor`. None of the extra games touch the bench:
it reads only `ckpt-*.pt` and the `checkpoint_published` markers.

## Measured

Verbatim `--bench` output (bench wall clock 20:23:28 → 21:16:43 UTC, 53 min):

```
candidate_version: 2
device: cpu  eval_sims: 512
cells_benched: 13  games_played: 26 (1 pair(s)/cell sampled)
seconds_per_game: 128.767
games_per_hour: 28.0
projected per checkpoint (production 24 pairs/cell): 624 games, 22.32h
publish_cadence_hours: 1.1283931054671605
feasible: False
```

The 13 cells are exactly what a live poll would schedule for member 2 — forms 5/6/7 × the
four frozen network-free rungs, plus one rung-8 cell (member 2 is the first challenger
with a historical member to meet):

```
rung5-v1-2|random   rung5-v1-2|largest-piece   rung5-v1-2|mobility   rung5-v1-2|uct-rollout-v1
rung6-v1-2|random   rung6-v1-2|largest-piece   rung6-v1-2|mobility   rung6-v1-2|uct-rollout-v1
rung7-v1-2|random   rung7-v1-2|largest-piece   rung7-v1-2|mobility   rung7-v1-2|uct-rollout-v1
rung7-v1-2|rung7-v1-1
```

At production load the full set is at most 15 cells/checkpoint (pin 1), i.e. 720 games →
≈ 25.8 h at this rate; the 13-cell projection above is the bench's own number and is what
is recorded.

## Where the time goes

Measured alongside, on the same checkpoint and device: one batch-1 network evaluation
(encode + forward + decode) costs **≈ 16 ms** (63 evals/s). Everything below that line is
arithmetic, not measurement:

- A form-6/7 move at S = 512 ≈ 512 × 16 ms ≈ **8 s of pure inference**; with ≈ 15 candidate
  moves per 30-ply game that is ≈ 2 min/game before any search overhead — most of the
  128.8 s/game mean.
- Form 5 (policy only, no search) is one evaluation per move and is close to free.
- The rung-4 opponent (UCT + random rollouts, pure Python) does not use the network
  search budget: `core.uct.UCTAgent` fixes `NUM_SIMULATIONS = 1000` per move (§12 M3's
  frozen rung-4 convention), independent of `eval_sims = 512`. It is itself minutes per
  game on the three `uct-rollout-v1` cells (one per form); the rung-1–3 opponents are
  negligible.
- The bench plays **one game at a time** in one process. That is not the same as one
  hardware thread: inference runs through torch with 6 intra-op threads, so how much of
  the 12-thread CPU sat idle was not measured.

## What this does and does not establish

**Does.** "Live and concurrent" (§12 M4) is falsifiable, and as configured it is
*projected* false. This is a projection from the measured rates, not an observed
liveness breach: a two-member run cannot breach pin 11's lag bound, since lag
(`core.eval_run.eval_lag`: newest published member minus the snapshot's complete-member
prefix) is at most 2 at the second publish. Projected at constant rates, one checkpoint
takes 22.32 h of eval against a 1.128 h publish cadence, so the backlog grows by
≈ 1 − 1.128/22.32 ≈ **0.95 checkpoints per publish interval**. With nothing scored yet
the lag samples run 1, 2, 3, 4, 5, 6, …; pin 11's breach is `lag > LIVENESS_MAX_LAG = 4`
at `LIVENESS_BREACH_CONSECUTIVE = 2` consecutive publish samples, so the first possible
breach is at the **sixth** publish. Over K = 30 the harness needs ≈ 30 × 22.32 ≈ 670 h
of eval **in total**. Training stops after ≈ 34 h at this cadence (≈ 38 h at the Aug-23
run's 1.26 h). After that the finite backlog does drain, but only ≈ 635 h later, which
is far outside any "live" reading. The harness code is not at fault — it played exactly
the cells a live poll would, at the pinned budget.

**Does not.** This says nothing about correctness, and nothing about the GPU: pin 11's
throughput bound (≤ 5 % learner degradation) is decided by the M5.5 pre-flight A/B, which
this bench cannot stand in for. The candidate is an early, weak net; game length and hence
seconds/game will drift as later checkpoints play longer games.

## Levers (all doc-first; none applied here)

Ordered by what they cost in pinned decisions:

1. **Parallel cell workers.** Cells are independent, so a multi-process harness is an
   implementation change touching no pin. Its speedup on this hardware is **unmeasured**:
   each process already runs torch with 6 intra-op threads, so N workers at that setting
   can oversubscribe the 12-thread CPU rather than scale. Sizing it needs a
   worker-count × intra-op-thread benchmark first.
2. **`device: cuda` for the harness.** Batch-1 inference on the 4060 Ti was ≈ 4 ms in the M2.5
   gate vs. 16 ms here — ≈ 3–4× on the inference share. Shares the GPU with the learner, so
   it is governed by pin 11's A/B, not by this bench.
3. **Batched inference across concurrent eval games** — the M5 lever, pulled forward.
4. **Re-pinning `S = 512` or 24 pairs/cell.** Both are material (pin 10): a new
   `protocol_version` and a new eval namespace. Last resort, not first.

(1) + (2) together are the candidate route to `feasible: True` (neither speedup is
measured here yet); a re-run of this bench after either change is the test.

## Reproducing

```bash
# a schema-v2 run (both caps, so it terminates on its own); ~2.5 h on the 4060 Ti
python3 scripts/run_selfplay.py configs/blokus_duo.json --max-learner-steps 400 --max-games-per-actor 250
# point configs/m4_eval.json's run_dir at the concrete run directory, then
python3 scripts/run_eval.py configs/m4_eval.json --bench
```

`--bench` writes nothing to the eval store; `runs/` is gitignored. Two published members
are needed for `publish_cadence_hours`/`feasible` to be non-null.
