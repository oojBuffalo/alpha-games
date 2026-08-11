# AlphaZero × Blokus Duo

An AlphaZero replication built as a **game-agnostic engine** with thin per-game adapters
behind a stable `Game` interface, targeting a single consumer GPU.

[![CI](https://img.shields.io/github/actions/workflow/status/oojBuffalo/alpha-games/ci.yml?label=CI)](https://github.com/oojBuffalo/alpha-games/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/oojBuffalo/alpha-games)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)

Proof-of-concept game: Blokus Duo (14×14, 2-player). The design doc is the source of truth —
[`metadocs/blokus-duo-az-design-v0_5.md`](metadocs/blokus-duo-az-design-v0_5.md); `CLAUDE.md`
is the compressed operational digest. The milestone plan lives in the design doc §12; this
repo is currently at **M0** (engine skeleton + correctness net).

> [!IMPORTANT]
> The design doc is the source of truth. Decisions D1–D12 are pinned — if the code needs to
> contradict it, update the doc first, then the code.

## Highlights

- **Game-agnostic core** — a stable `Game` ABC with thin per-game adapters; adding a game
  touches only `games/` and `configs/`.
- **Sparse, player-aware MCTS** — `{N,W,Q,P}` over legal actions only, never dense over the
  17,836-action space; Q is stored in the parent-mover's perspective, so PUCT selection needs
  no negamax flip.
- **Single-GPU target** — built for one consumer GPU (RTX 4060 Ti 16GB); self-play throughput,
  not network size, is the binding constraint.
- **Oracle-first correctness** — a slow, independently-implemented reference engine is
  differential-tested against the fast path before anything ships.

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

## Usage

```sh
python3 -m pytest                     # full battery
python3 -m pytest -m "not slow"       # fast subset (skips the high-sim search sweeps)
python3 -m ruff check . && python3 -m ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs lint, format check, and the full battery on push/PR.

Fixture generation (M1 Blokus battery) — both write to `tests/fixtures/blokus/*.json` with
the orientation hash + encoding conventions embedded; regeneration on unchanged code must be
byte-identical:

```sh
python3 scripts/gen_blokus_symmetry_table.py   # orientation/symmetry table (seconds)
python3 scripts/gen_blokus_perft.py            # perft(3), Klein-4 orbit-reduced (~3 min)
```

Optional, GPU-only, manual (RTX 4060 Ti 16GB): the D5 batch-size sweep behind the batch-256
pin, `docs/bench/m2-train-step.md` is observational only, and CI never runs it.

```sh
python3 scripts/bench_train_step.py --out docs/bench/m2-train-step.md
```

<!-- TODO: self-play / training entrypoint — not yet built (M3+) -->

## Project Layout

- `core/` — game-generic engine: the `Game` ABC + v1-envelope assertion (`core/game.py`)
  and the sparse, player-aware PUCT search (`core/mcts.py`). No game- or network-specific
  logic. Pure-stdlib through M0.
- `games/` — one package per adapter. M0 ships `tictactoe/` and `connect4/` (reference
  games); Blokus (M1) and Othello (M1.5) follow. **Adding a game touches only `games/`
  and `configs/`.**
- `tests/` — the test battery, incl. an independent max-n reference solver
  (`tests/reference/`) and the synthetic pass-game / bad-adapter fixtures (`tests/fixtures/`).
- `scripts/` — fixture generation (`gen_blokus_*.py`) and the manual GPU training-step
  benchmark (`bench_train_step.py`).
- `docs/` — AlphaZero background reading (design outline + preprint).
- `metadocs/` — the design doc; source of truth for architecture and the milestone plan.

## Contributing

Follow `CLAUDE.md` — the project's compressed operational digest (working principles,
milestone process, invariants). At minimum, both of these must pass before opening a PR — CI
re-runs the full battery (without `-m "not slow"`):

```sh
python3 -m pytest -m "not slow"
python3 -m ruff check . && python3 -m ruff format --check .
```

## License

MIT — see [LICENSE](LICENSE).

## References

- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go
  through self-play* (AlphaZero), Science 2018; preprint under
  [`docs/alphazero_preprint.pdf`](docs/alphazero_preprint.pdf) (arXiv:1712.01815).
- KataGo (Wu, 2020) — playout-cap randomization and the deferred auxiliary-target bundle.
