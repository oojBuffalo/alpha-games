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

## Develop

```sh
python3 -m pip install -e ".[dev]"   # pytest + ruff (optional: pyproject sets pythonpath)
python3 -m pytest                     # full battery
python3 -m pytest -m "not slow"       # fast subset (skips the high-sim search sweeps)
python3 -m ruff check . && python3 -m ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs lint, format check, and the full battery on push/PR.
