---
id: 13
title: Benchmark D5 batch sizes 128/256/512 on the 4060 Ti
status: pending
priority: medium
dependencies: [9]
complexity: 2
recommended_subtasks: 0
---

## Description
D5 pins "batch 256 (benchmark 128/256/512)" on the confirmed RTX 4060 Ti 16 GB — the benchmark
is part of the pinned decision, not an optional extra. Produce the measurement artifact that
either confirms batch 256 or gives M3 the numbers to argue otherwise.

## Details
- New `scripts/bench_train_step.py` (torch; scripts/ sits outside the installed package, so the
  pyproject "confined to core/…" comment — which governs package code — is unaffected). Seeded
  synthetic Blokus-shaped batches (46×14×14 planes, realistic legal-set sizes) through task 9's
  real `train_step` with AMP on CUDA.
- For each batch size in {128, 256, 512}: warmup steps, then timed steps → report step time,
  positions/sec, and peak VRAM (`torch.cuda.max_memory_allocated`), plus GPU name and torch
  version.
- Commit the output as `docs/bench/m2-train-step.md` — the acceptance artifact. Include the
  one-line verdict: does batch 256 fit comfortably and is it the throughput-reasonable choice.
- **Manual GPU task:** CI is CPU-only and never runs this; the script degrades to a clear error
  (not a silent CPU benchmark) when CUDA is absent.

## Test Strategy
Run on the 4060 Ti box; the committed `docs/bench/m2-train-step.md` with all three batch sizes,
VRAM numbers, and the batch-256 verdict is the acceptance criterion. `ruff` clean; script exits
loudly without CUDA.

## Complexity Analysis
A timing harness around an already-built train step; the only care points are proper CUDA timing
(synchronize before reading clocks) and honest VRAM accounting. No production code paths change.

**Suggested expansion approach:** none — atomic; a script plus its committed artifact.
