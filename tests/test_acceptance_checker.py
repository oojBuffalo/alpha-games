"""The M3 GPU-acceptance-run checker: ``core/acceptance.py`` (§12 M3, issue #63).

Synthetic run directories, entirely in-process (no real self-play/training,
mirroring ``tests/test_replay_window.py``'s synthetic-shard technique) --
fast, and exercises the checker's own logic in isolation from the actor/
learner/IPC stack the other M3 tests already cover exhaustively. One
passing fixture, and one perturbation per documented failure mode.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from core.acceptance import MIN_COMPLETED_GAMES, verify_acceptance
from core.metrics import EpochMetricsWriter
from core.observability import (
    CHECKPOINT_PUBLISHED_KIND,
    SERIES_GAMES_COMPLETED,
    SERIES_LEARNER_STEP,
    SERIES_POSITIONS_EVALUATED,
    SERIES_SIMS_RUN,
    delta_record,
    total_record,
)
from core.replay_shard import SampleRecord, shard_filename, write_shard
from games.tictactoe import TicTacToe

TTT = TicTacToe()


def _game_records(run_id, actor_id, game_index, movers, model_version):
    """Build one synthetic game's records, with a caller-chosen mover sequence.

    Args:
        run_id: The synthetic run id.
        actor_id: The synthetic actor id.
        game_index: The game's durable index.
        movers: The mover stamped on each successive ply -- the caller's
            direct control over consecutive-mover (blocked-skip) sequences.
        model_version: Stamped on every record in this game.

    Returns:
        A tuple of ``len(movers)`` ``SampleRecord``s.
    """
    state = TTT.initial_state()
    planes = TTT.encode_state(state)
    legal_action = next(iter(TTT.legal_moves(state)))
    return tuple(
        SampleRecord(
            planes=planes,
            sparse_pi=((legal_action, 1),),
            z=0.0,
            aux=(),
            mover=mover,
            model_version=model_version,
            ply=ply,
            game_id=(run_id, actor_id, game_index),
        )
        for ply, mover in enumerate(movers)
    )


def _write_game_shard(shard_dir, run_id, actor_id, seq, movers, model_version):
    """Publish one synthetic single-game shard."""
    records = _game_records(run_id, actor_id, seq, movers, model_version)
    shard_id = shard_filename(run_id, actor_id, seq)
    write_shard(shard_dir / shard_id, TTT, records, run_id=run_id, actor_id=actor_id, seq=seq)


def _run_config_stub(*, batch_size, publish_interval):
    """A minimal duck-typed run-config stand-in (mirrors ``tests/test_learner.py``)."""
    return SimpleNamespace(
        training=SimpleNamespace(batch_size=batch_size, publish_interval=publish_interval)
    )


def _write_checkpoints(ckpt_dir: Path, versions) -> None:
    """Touch empty ``ckpt-<v>.pt`` files -- ``list_published_versions`` only globs names."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for v in versions:
        (ckpt_dir / f"ckpt-{v}.pt").touch()


def _write_learner_metrics(run_dir: Path, *, learner_step: int, published_versions) -> None:
    writer = EpochMetricsWriter(run_dir, "learner")
    writer.append(total_record(SERIES_LEARNER_STEP, learner_step))
    for v in published_versions:
        writer.append(
            {
                "kind": CHECKPOINT_PUBLISHED_KIND,
                "model_version": v,
                "learner_step": v,
                "timestamp": time.time(),
            }
        )


def _write_actor_metrics(
    run_dir: Path,
    *,
    num_games: int,
    positions_per_game: int = 3,
    include_positions_series: bool = True,
) -> None:
    writer = EpochMetricsWriter(run_dir, "actor-0")
    for _ in range(num_games):
        writer.append(delta_record(SERIES_GAMES_COMPLETED, 1))
        writer.append(delta_record(SERIES_SIMS_RUN, 128))
        if include_positions_series:
            writer.append(delta_record(SERIES_POSITIONS_EVALUATED, positions_per_game))


# --- the passing fixture -------------------------------------------------------

_NUM_GAMES = MIN_COMPLETED_GAMES + 5  # comfortably over the floor
_PUBLISH_INTERVAL = 5
_LEARNER_STEP = 10  # >= 2 full publish intervals
_BATCH_SIZE = 4


def _build_passing_run(tmp_path: Path, run_id: str = "accept-run") -> Path:
    """Build a synthetic run directory that passes every acceptance check."""
    root = tmp_path / run_id
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)

    for i in range(_NUM_GAMES):
        if i == 0:
            movers = [0, 0, 1, 0, 1]  # consecutive-mover (blocked-skip) sequence at plies 0-1
        else:
            movers = [0, 1, 0]
        model_version = 0 if i < _NUM_GAMES // 2 else 1  # actor reload observed partway through
        _write_game_shard(shard_dir, run_id, "0", i, movers, model_version)

    _write_actor_metrics(root, num_games=_NUM_GAMES)
    _write_learner_metrics(root, learner_step=_LEARNER_STEP, published_versions=(0, 1))
    _write_checkpoints(root / "checkpoints", (0, 1))
    return root


def _run_config():
    return _run_config_stub(batch_size=_BATCH_SIZE, publish_interval=_PUBLISH_INTERVAL)


def _check(report, name):
    matches = [c for c in report.checks if c.name == name]
    assert len(matches) == 1, f"no unique check named {name!r} in {[c.name for c in report.checks]}"
    return matches[0]


def test_passing_run_reports_pass_on_every_check(tmp_path):
    root = _build_passing_run(tmp_path)
    report = verify_acceptance(root, TTT, _run_config())
    assert report.passed
    for check in report.checks:
        assert check.passed, f"{check.name}: {check.detail}"


def test_render_produces_a_pass_fail_checklist(tmp_path):
    root = _build_passing_run(tmp_path)
    report = verify_acceptance(root, TTT, _run_config())
    text = report.render()
    assert text.count("[PASS]") == len(report.checks)
    assert "[FAIL]" not in text
    assert text.strip().endswith("verdict: PASS")


# --- failure modes: one perturbation each --------------------------------------


def test_fails_when_fewer_than_the_game_floor(tmp_path):
    root = tmp_path / "run"
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    for i in range(5):  # well under MIN_COMPLETED_GAMES
        _write_game_shard(shard_dir, "run", "0", i, [0, 1, 0], model_version=0)
    _write_actor_metrics(root, num_games=5)
    _write_learner_metrics(root, learner_step=_LEARNER_STEP, published_versions=(0, 1))
    _write_checkpoints(root / "checkpoints", (0, 1))

    report = verify_acceptance(root, TTT, _run_config())
    check = _check(report, f"at least {MIN_COMPLETED_GAMES} completed games")
    assert not check.passed
    assert not report.passed


def test_fails_when_no_game_has_a_consecutive_mover_sequence(tmp_path):
    root = tmp_path / "run"
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    for i in range(_NUM_GAMES):
        _write_game_shard(
            shard_dir, "run", "0", i, [0, 1, 0, 1], model_version=0
        )  # strict alternation
    _write_actor_metrics(root, num_games=_NUM_GAMES)
    _write_learner_metrics(root, learner_step=_LEARNER_STEP, published_versions=(0, 1))
    _write_checkpoints(root / "checkpoints", (0, 1))

    report = verify_acceptance(root, TTT, _run_config())
    check = _check(report, "at least one consecutive-mover (blocked-skip) sequence")
    assert not check.passed
    assert not report.passed


def test_fails_when_no_reload_observed(tmp_path):
    root = tmp_path / "run"
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    for i in range(_NUM_GAMES):
        movers = [0, 0, 1] if i == 0 else [0, 1, 0]
        _write_game_shard(shard_dir, "run", "0", i, movers, model_version=0)  # every game at v0
    _write_actor_metrics(root, num_games=_NUM_GAMES)
    _write_learner_metrics(root, learner_step=_LEARNER_STEP, published_versions=(0, 1))
    _write_checkpoints(root / "checkpoints", (0, 1))

    report = verify_acceptance(root, TTT, _run_config())
    check = _check(report, "actor reload observed (some game played at model_version >= 1)")
    assert not check.passed
    assert not report.passed


def test_fails_when_no_samples_drawn(tmp_path):
    """Real shards exist, but the learner never trained a step -- nothing sampled."""
    root = _build_passing_run(tmp_path)
    # Overwrite the learner metrics with learner_step = 0 (no samples_drawn).
    for p in (root / "metrics").glob("learner-*.jsonl"):
        p.unlink()
    _write_learner_metrics(root, learner_step=0, published_versions=())
    _write_checkpoints(root / "checkpoints", (0,))  # only the seeded init

    report = verify_acceptance(root, TTT, _run_config())
    check = _check(report, "shards ingested and sampled")
    assert not check.passed
    assert not report.passed


def test_fails_when_no_checkpoint_published_beyond_v0(tmp_path):
    root = _build_passing_run(tmp_path)
    for p in (root / "checkpoints").glob("ckpt-*.pt"):
        p.unlink()
    _write_checkpoints(root / "checkpoints", (0,))  # only the seeded init

    report = verify_acceptance(root, TTT, _run_config())
    check = _check(report, "at least one published checkpoint beyond the seeded v0")
    assert not check.passed
    assert not report.passed


def test_fails_when_learner_step_short_of_one_publish_interval(tmp_path):
    root = _build_passing_run(tmp_path)
    huge_publish_interval_config = _run_config_stub(batch_size=_BATCH_SIZE, publish_interval=10_000)

    report = verify_acceptance(root, TTT, huge_publish_interval_config)
    check = _check(report, "at least one full publish interval of learner steps")
    assert not check.passed
    assert not report.passed


def test_fails_when_a_documented_series_is_missing(tmp_path):
    root = tmp_path / "run"
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    for i in range(_NUM_GAMES):
        movers = [0, 0, 1] if i == 0 else [0, 1, 0]
        model_version = 0 if i < _NUM_GAMES // 2 else 1
        _write_game_shard(shard_dir, "run", "0", i, movers, model_version)
    _write_actor_metrics(root, num_games=_NUM_GAMES, include_positions_series=False)
    _write_learner_metrics(root, learner_step=_LEARNER_STEP, published_versions=(0, 1))
    _write_checkpoints(root / "checkpoints", (0, 1))

    report = verify_acceptance(root, TTT, _run_config())
    check = _check(report, "every documented series present with a positive total")
    assert not check.passed
    assert SERIES_POSITIONS_EVALUATED in check.detail
    assert not report.passed


def test_fails_gracefully_on_an_empty_run_directory(tmp_path):
    root = tmp_path / "empty-run"
    root.mkdir()
    report = verify_acceptance(root, TTT, _run_config())
    assert not report.passed
    assert not _check(report, f"at least {MIN_COMPLETED_GAMES} completed games").passed
