# AlphaZero × Blokus Duo

An AlphaZero replication built as a **game-agnostic engine** with thin per-game adapters
behind a stable `Game` interface, targeting a single consumer GPU. Proof-of-concept game:
Blokus Duo (14×14, 2-player).

The design doc is the source of truth: [`metadocs/blokus-duo-az-design-v0_5.md`](metadocs/blokus-duo-az-design-v0_5.md).
`CLAUDE.md` is the compressed operational digest. The milestone plan lives in the design
doc §12; this repo is currently at **M0** (engine skeleton + correctness net).

## Layout

- `core/` — game-generic engine: the `Game` ABC + v1-envelope assertion (`core/game.py`)
  and the sparse, player-aware PUCT search (`core/mcts.py`). No game- or network-specific
  logic. Pure-stdlib through M0.
- `games/` — one package per adapter. M0 ships `tictactoe/` and `connect4/` (reference
  games); Blokus (M1) and Othello (M1.5) follow. **Adding a game touches only `games/`
  and `configs/`.**
- `tests/` — the test battery, incl. an independent max-n reference solver
  (`tests/reference/`) and the synthetic pass-game / bad-adapter fixtures (`tests/fixtures/`).

## Setup

Requires Python 3.11+ (CI runs 3.12). `core/` is pure-stdlib through M0; NumPy/torch
are needed from M2 onward for encoding/training.

```sh
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
```

`requirements.txt` / `requirements-dev.txt` mirror the dependencies declared in
`pyproject.toml` (the source of truth); an editable install works the same way:

```sh
python3 -m pip install -e ".[dev]"   # pytest + ruff (optional: pyproject sets pythonpath)
```

On a machine without a CUDA GPU (e.g. CI), install the CPU-only torch wheels first so
the editable/requirements install doesn't resolve the multi-GB CUDA build:

```sh
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements-dev.txt
```

## Develop

```sh
python3 -m pytest                     # full battery
python3 -m pytest -m "not slow"       # fast subset (skips the high-sim search sweeps)
python3 -m ruff check . && python3 -m ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs lint, format check, and the full battery on push/PR.
