---
id: 2
title: Add NumPy and PyTorch dependencies
status: pending
priority: high
dependencies: []
complexity: 3
recommended_subtasks: 0
---

## Description
Introduce NumPy and PyTorch as project dependencies — M2 is the milestone where they arrive
(`core/` has been pure stdlib through M1.6) — and keep CI green with CPU wheels.

## Details
- `pyproject.toml`: `dependencies = []` currently carries the comment "NumPy/torch arrive with
  encoding/training (M2)". Add `numpy>=1.26` and `torch>=2.2` to `[project] dependencies`; leave
  the `dev` extra (pytest, ruff) as is.
- `.github/workflows/ci.yml`: ensure the install step pulls CPU-only torch wheels
  (`--index-url https://download.pytorch.org/whl/cpu` or the `+cpu` extra-index pattern) so CI
  stays fast and does not download CUDA wheels.
- Adapters stay stdlib-pure: `games/` modules do not import numpy/torch (task 3 returns nested
  tuples, following the Othello precedent). Torch/numpy imports are confined to the new training
  modules (`core/network.py`, `core/losses.py`, `core/train.py` — tasks 7–9).

## Test Strategy
`python3 -m pip install -e ".[dev]"` succeeds; `python3 -c "import numpy, torch"` works; the full
existing battery (`python3 -m pytest`) still passes; CI run completes on a branch push.

## Complexity Analysis
Config-only change (two `pyproject.toml` lines plus a CI install tweak), but scored above minimum
because CI wheel selection is iteration-prone: CPU-only torch indexes, wheel size/cache limits on
GitHub Actions, and the risk of the resolver pulling CUDA wheels locally vs CPU in CI. No
production code changes; the battery itself is untouched.

**Suggested expansion approach:** none — atomic; any CI fixes are part of landing this task, not
separate subtasks.
