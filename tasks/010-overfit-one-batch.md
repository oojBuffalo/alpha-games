---
id: 10
title: Add the overfit-one-batch test
status: pending
priority: high
dependencies: [1, 3, 9]
complexity:
recommended_subtasks:
---

## Description
The §12 M2 exit test: the full model + loss + optimizer memorizes one small real batch. "The
overfit-one-batch test presupposes the whole model" — it is the proof that every M2 piece is
wired together correctly.

## Details
- New `tests/test_overfit.py`, marked `@pytest.mark.slow` (the marker is declared in
  `pyproject.toml` `[tool.pytest.ini_options]`).
- Build a batch of ~8–16 real positions: seeded random playouts through `BlokusDuo` (apply random
  legal actions), `encode_state` each; synthetic-but-plausible targets — sparse π as visit counts
  concentrated on one legal action per position, `z`/`aux` from
  `games/blokus_duo/targets.py::value_targets` applied to a played-out result (or fixed ±1/0
  values spanning the range).
- Use the pinned `λ_aux` via `value_target_spec()` (task 1) — this test is the reason the value
  had to be pinned.
- Train a few hundred full-precision CPU steps (skip AMP here; determinism beats speed) with a
  fixed seed at a higher LR if needed for fast memorization.
- Assert: total loss falls below a small threshold; policy argmax equals the target action on
  ≥90% of the batch; value predictions land within a tolerance of their `z` targets.

## Test Strategy
This task is a test. `python3 -m pytest tests/test_overfit.py` passes deterministically (fixed
seeds, CPU); it must run in tens of seconds, not minutes — shrink the batch or steps if not.
Sanity-check it fails when sabotaged (e.g. zero the LR, or feed permuted policy targets) before
trusting the green.
