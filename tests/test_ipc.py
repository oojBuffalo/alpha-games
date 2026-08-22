"""Actor–learner filesystem IPC: ``core/ipc.py`` (§12 M3, issue #61).

Three layers, cheapest first:

1. **The seams in isolation** -- :class:`~core.ipc.ShutdownFlag`,
   :func:`~core.ipc.build_actor_pacing`, :func:`~core.ipc.build_actor_refresh` --
   pure closures over paths, unit-tested without any driver or process.
2. **The artifact protocol, exhaustively, single-process.** Both drivers
   wired through ``core.ipc``'s real seams (never test doubles) in one
   process, interleaved by hand: publishes flow into real games, no game's
   records ever mix model versions, a restarted actor never reissues a
   durable identity, and a shutdown strictly between publish boundaries
   writes only the rolling resume snapshot -- the acceptance test's every
   invariant, proved deterministically.
3. **Process mechanics, real multiprocessing (slow).** The same protocol
   under real ``spawn``-context processes and a real ``SIGTERM``: publishes
   still flow, a killed-and-restarted actor still resumes cleanly, and every
   process exits on its own after being signaled -- proving the wiring
   survives actual process/signal boundaries, not just the artifact
   contracts layer 2 already covers exhaustively.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import time
from types import SimpleNamespace

import pytest

from core.actor import ActorDriver
from core.artifact_fingerprint import FingerprintMismatchError
from core.checkpoint import (
    list_published_versions,
    load_checkpoint,
    published_checkpoint_path,
    resume_path,
)
from core.ipc import (
    ShutdownFlag,
    build_actor_pacing,
    build_actor_refresh,
    launch_run,
    start_actor_process,
)
from core.learner import (
    PACING_GO,
    PACING_HOLD,
    LearnerDriver,
    pacing_file_path,
    write_pacing_file,
)
from core.network import NetworkConfig
from core.replay_shard import SampleRecord, read_shard, shard_filename, write_shard
from core.replay_window import ReplayWindow
from core.runconfig import TrainingConfig, load_run_config
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import MICRO_CONFIG
from games.othello import Othello
from games.tictactoe import TicTacToe

TTT = TicTacToe()
TTT_NET_CONFIG = NetworkConfig(2, (3, 3), (9,), trunk_blocks=1, trunk_channels=4)
OTHELLO = Othello()


def _ttt_self_play(**overrides):
    """A D6-validate-tier :class:`~core.runconfig.SelfPlayConfig` over TTT."""
    base = dataclasses.replace(load_run_config().self_play, sims=128)
    return dataclasses.replace(base, **overrides) if overrides else base


def _ttt_training_config(**overrides):
    """A ``TrainingConfig`` sized for fast TTT-based IPC tests."""
    base = dict(
        games=1,
        learner_steps=1,
        steps_per_game=1,
        batch_size=4,
        replay_window=10_000,
        learning_rate=1e-2,
        warmup_steps=0,
        cosine_total_steps=100,
        aux_loss_weight=0.0,
        checkpoint_selection="final",
        publish_interval=1000,
        checkpoint_count=1,
        replay_warmup_positions=1,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _stub_run_config(training, run_seed=1):
    """A minimal duck-typed run-config stand-in (mirrors ``tests/test_learner.py``)."""
    return SimpleNamespace(
        training=training,
        run_seed=run_seed,
        to_dict=lambda: {"training": dataclasses.asdict(training), "run_seed": run_seed},
    )


def _write_ttt_shard(shard_dir, run_id, actor_id, seq, num_positions):
    """Write one synthetic TTT shard of exactly ``num_positions`` records."""
    state = TTT.initial_state()
    planes = TTT.encode_state(state)
    legal_action = next(iter(TTT.legal_moves(state)))
    records = tuple(
        SampleRecord(
            planes=planes,
            sparse_pi=((legal_action, 1),),
            z=0.0,
            aux=(),
            mover=0,
            model_version=0,
            ply=ply,
            game_id=(run_id, actor_id, seq),
        )
        for ply in range(num_positions)
    )
    shard_id = shard_filename(run_id, actor_id, seq)
    write_shard(shard_dir / shard_id, TTT, records, run_id=run_id, actor_id=actor_id, seq=seq)
    return shard_id


def _make_learner(shard_dir, ckpt_dir, run_dir, *, training=None, **kwargs):
    """Build a :class:`~core.learner.LearnerDriver` over TTT with a tiny net."""
    training = training if training is not None else _ttt_training_config()
    return LearnerDriver(
        game=TTT,
        run_config=_stub_run_config(training),
        shard_dir=shard_dir,
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
        network_config=TTT_NET_CONFIG,
        **kwargs,
    )


# ==============================================================================
# 1. ShutdownFlag
# ==============================================================================


def test_shutdown_flag_starts_clear():
    assert ShutdownFlag()() is False


def test_shutdown_flag_catches_sigterm_and_sigint():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        term_flag = ShutdownFlag().install()
        os.kill(os.getpid(), signal.SIGTERM)
        assert term_flag() is True

        int_flag = ShutdownFlag().install()
        os.kill(os.getpid(), signal.SIGINT)
        assert int_flag() is True
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def test_shutdown_flag_install_returns_self_for_chaining():
    saved = signal.getsignal(signal.SIGTERM)
    try:
        flag = ShutdownFlag()
        assert flag.install() is flag
    finally:
        signal.signal(signal.SIGTERM, saved)


# ==============================================================================
# 2. build_actor_pacing
# ==============================================================================


def test_pacing_missing_file_reads_as_go(tmp_path):
    assert build_actor_pacing(tmp_path)() is False


def test_pacing_reflects_the_live_file_every_call(tmp_path):
    path = pacing_file_path(tmp_path)
    pacing = build_actor_pacing(tmp_path)

    write_pacing_file(
        path, state=PACING_HOLD, ratio=0.5, positions_stored=10, samples_drawn=5, learner_step=1
    )
    assert pacing() is True

    write_pacing_file(
        path, state=PACING_GO, ratio=3.0, positions_stored=10, samples_drawn=30, learner_step=3
    )
    assert pacing() is False  # same closure, re-reads the file each call


# ==============================================================================
# 3. build_actor_refresh
# ==============================================================================


def test_refresh_blocks_until_the_first_checkpoint_lands(tmp_path):
    shard_dir, ckpt_dir, run_dir = tmp_path / "shards", tmp_path / "ckpt", tmp_path / "run"
    calls = {"n": 0}

    def wait():
        calls["n"] += 1
        if calls["n"] == 2:
            _make_learner(shard_dir, ckpt_dir, run_dir)  # publishes v0 as a side effect

    refresh = build_actor_refresh(
        game=TTT, ckpt_dir=ckpt_dir, network_config=TTT_NET_CONFIG, wait=wait
    )
    evaluator, version = refresh()

    assert calls["n"] == 2
    assert version == 0
    assert evaluator is not None


def test_refresh_returns_growing_versions_and_caches_the_unchanged_case(tmp_path):
    shard_dir, ckpt_dir, run_dir = tmp_path / "shards", tmp_path / "ckpt", tmp_path / "run"
    _write_ttt_shard(shard_dir, "run", "a", 0, 10)
    training = _ttt_training_config(
        publish_interval=1, checkpoint_count=3, replay_warmup_positions=1
    )
    learner = _make_learner(shard_dir, ckpt_dir, run_dir, training=training)

    def unreachable():
        raise AssertionError("must not block: a checkpoint already exists")

    refresh = build_actor_refresh(
        game=TTT, ckpt_dir=ckpt_dir, network_config=TTT_NET_CONFIG, wait=unreachable
    )
    eval0, v0 = refresh()
    eval0_again, v0_again = refresh()
    assert (v0, v0_again) == (0, 0)
    assert eval0 is eval0_again  # cache hit: an unchanged `latest` never rebuilds

    learner._run_step()  # step 0 -> 1, publish_interval=1 -> publishes v1
    eval1, v1 = refresh()
    assert v1 == 1
    assert eval1 is not eval0  # cache miss: a new `latest` rebuilds


def test_refresh_raises_loudly_on_fingerprint_mismatch(tmp_path):
    shard_dir, ckpt_dir, run_dir = tmp_path / "shards", tmp_path / "ckpt", tmp_path / "run"
    _make_learner(shard_dir, ckpt_dir, run_dir)  # publishes v0 under TTT's fingerprint

    mismatched = build_actor_refresh(game=OTHELLO, ckpt_dir=ckpt_dir)
    with pytest.raises(FingerprintMismatchError):
        mismatched()


# ==============================================================================
# 4. Single-process integration: the artifact protocol, exhaustively (TTT)
# ==============================================================================


def test_publishes_flow_and_no_game_ever_mixes_model_versions(tmp_path):
    shard_dir, ckpt_dir, run_dir = tmp_path / "shards", tmp_path / "ckpt", tmp_path / "run"
    # replay_warmup_positions kept high on purpose: this test isolates publish
    # flow / version-mixing / shard-uniqueness, not D5 pacing dynamics (that's
    # test_learner.py's job) -- with only one actor and a manual step-for-step
    # interleave, real floor/ceiling enforcement would starve on so little
    # data and deadlock the very pacing wiring under test.
    training = _ttt_training_config(
        publish_interval=1, checkpoint_count=6, replay_warmup_positions=1000
    )
    learner = _make_learner(shard_dir, ckpt_dir, run_dir, training=training)  # publishes v0

    refresh = build_actor_refresh(game=TTT, ckpt_dir=ckpt_dir, network_config=TTT_NET_CONFIG)
    pacing = build_actor_pacing(run_dir)
    actor = ActorDriver(
        game=TTT,
        self_play=_ttt_self_play(),
        run_id="run-1",
        actor_id=0,
        out_dir=shard_dir,
        run_seed=7,
        refresh=refresh,
        pacing=pacing,
        max_games=1,
    )

    all_paths = []
    versions_seen = set()
    for _ in range(4):
        (path,) = actor.run()  # plays exactly one game at whatever is currently latest
        all_paths.append(path)
        game_versions = {r.model_version for r in read_shard(path, TTT).records}
        assert len(game_versions) == 1  # (b) one model_version per game_id
        versions_seen |= game_versions
        learner._run_step()  # advances one step; publish_interval=1 -> a new version

    assert len(list_published_versions(ckpt_dir)) >= 5  # (a) publishes actually flowed
    assert versions_seen >= {0, 1}  # actor played under >= 2 distinct versions

    names = [p.name for p in all_paths]
    assert len(names) == len(set(names))  # (c) shard names unique

    # The learner's own window (constructed above) is exactly "the learner's
    # window" the design doc says picks shards up on its next rescan -- a
    # second, freshly constructed ReplayWindow over the same shard_dir would
    # see them as already-ingested via the persisted manifest, not newly
    # discovered, so this checks the real window rather than a bystander one.
    learner.window.rescan()
    ingested_ids = {entry.shard_id for entry in learner.window.shard_entries}
    assert ingested_ids == set(names)  # (c) every actor shard ingested cleanly
    assert learner.window.positions_stored == sum(
        len(read_shard(p, TTT).records) for p in all_paths
    )


def test_restarted_actor_resumes_its_persisted_sequence_without_duplicate_identity(tmp_path):
    shard_dir, ckpt_dir, run_dir = tmp_path / "shards", tmp_path / "ckpt", tmp_path / "run"
    _make_learner(shard_dir, ckpt_dir, run_dir)  # publishes v0

    def make_actor():
        # Fresh closures each call -- exactly what a real process restart gets.
        refresh = build_actor_refresh(game=TTT, ckpt_dir=ckpt_dir, network_config=TTT_NET_CONFIG)
        pacing = build_actor_pacing(run_dir)
        return ActorDriver(
            game=TTT,
            self_play=_ttt_self_play(),
            run_id="run-restart",
            actor_id=1,
            out_dir=shard_dir,
            run_seed=99,
            refresh=refresh,
            pacing=pacing,
            max_games=2,
        )

    first_paths = make_actor().run()
    first_indices = [read_shard(p, TTT).records[0].game_id[2] for p in first_paths]
    assert first_indices == [0, 1]

    # "Restart": a brand-new ActorDriver (+ brand-new refresh/pacing closures)
    # over the same (run_id, actor_id, shard_dir) identity.
    second_paths = make_actor().run()
    second_indices = [read_shard(p, TTT).records[0].game_id[2] for p in second_paths]
    assert second_indices == [2, 3]

    all_indices = first_indices + second_indices
    assert len(all_indices) == len(set(all_indices))  # no duplicate (actor_id, game_index)


def test_shutdown_between_publish_boundaries_writes_a_snapshot_never_a_publish(tmp_path):
    shard_dir, ckpt_dir, run_dir = tmp_path / "shards", tmp_path / "ckpt", tmp_path / "run"
    _write_ttt_shard(shard_dir, "run", "a", 0, 20)

    training = _ttt_training_config(
        publish_interval=5, checkpoint_count=4, replay_warmup_positions=1
    )
    stop = {"n": 0}

    def should_stop():
        stop["n"] += 1
        return stop["n"] > 2  # stop after 2 trained steps -- strictly inside [0, 5)

    learner = _make_learner(
        shard_dir, ckpt_dir, run_dir, training=training, should_stop=should_stop
    )
    assert list_published_versions(ckpt_dir) == (0,)  # only the mandatory fresh-start v0
    assert not resume_path(ckpt_dir).exists()

    learner.run()

    assert learner.step == 2  # stopped strictly between publish boundaries
    assert list_published_versions(ckpt_dir) == (0,)  # shutdown never adds a ckpt-*.pt
    snapshot = load_checkpoint(resume_path(ckpt_dir), TTT)
    assert snapshot.learner_step == 2  # the rolling snapshot reflects the in-flight step


# ==============================================================================
# 5. Real multiprocessing: process mechanics + a real SIGTERM (slow)
# ==============================================================================

MICRO = BlokusDuo(config=MICRO_CONFIG)


def _build_micro_game():
    """A picklable, module-level game factory for spawn-context processes.

    Must be a real top-level function (never a lambda/closure) -- pickle
    resolves it by ``module.qualname`` in the child process (module
    docstring of ``core.ipc``).
    """
    return BlokusDuo(config=MICRO_CONFIG)


def _micro_net_config():
    """A tiny net over the micro-Blokus encoding surface -- speed, not throughput."""
    base = NetworkConfig.from_game(MICRO)
    return dataclasses.replace(base, trunk_blocks=1, trunk_channels=4)


MICRO_NET_CONFIG = _micro_net_config()


def _acceptance_run_config():
    """A tiny-but-real micro-Blokus ``RunConfig`` for the multiprocess acceptance run."""
    cfg = load_run_config()
    self_play = dataclasses.replace(cfg.self_play, sims=128)  # D6 tier (ActorDriver requires it)
    training = dataclasses.replace(
        cfg.training,
        batch_size=4,
        replay_window=2000,
        learning_rate=1e-3,
        warmup_steps=0,
        cosine_total_steps=200,
        publish_interval=1,
        # Deliberately large: the tiny net / tiny board here trains fast
        # enough that a small checkpoint_count could let the learner reach
        # its own pinned stop condition before this test gets around to
        # signaling it -- checked separately and precisely by the
        # single-process shutdown test above. This test's job is proving the
        # signal actually crosses the process boundary while the learner is
        # still genuinely mid-run.
        checkpoint_count=100_000,
        replay_warmup_positions=1,
    )
    return dataclasses.replace(cfg, self_play=self_play, training=training)


def _wait_until(predicate, timeout, interval=0.2):
    """Poll ``predicate`` until it is true or ``timeout`` seconds have passed.

    Args:
        predicate: Zero-argument callable.
        timeout: Seconds to keep polling.
        interval: Seconds between polls.

    Returns:
        The last value of ``predicate()`` (so a caller gets a truthy/falsy
        result either way, never an exception from timing out).
    """
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result or time.monotonic() >= deadline:
            return result
        time.sleep(interval)


@pytest.mark.slow
def test_multiprocess_acceptance_actors_and_learner_share_the_filesystem(tmp_path):
    """The full #61 acceptance scenario: real spawn-context processes, a real signal.

    Two actor processes + one learner process, wired entirely through
    ``core.ipc.launch_run``. Proves process *mechanics* -- the single-process
    tests above already prove the artifact protocol exhaustively (module
    docstring): publishes actually cross the process boundary, a real
    ``SIGTERM`` reaches a specific actor and it restarts cleanly under its
    old identity, and every process exits on its own once signaled.
    """
    run_config = _acceptance_run_config()
    shard_dir, ckpt_dir, run_dir = tmp_path / "shards", tmp_path / "ckpt", tmp_path / "run"
    run_id = "accept-run"

    launched = launch_run(
        game_factory=_build_micro_game,
        run_config=run_config,
        self_play=run_config.self_play,
        run_id=run_id,
        num_actors=2,
        shard_dir=shard_dir,
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
        run_seed=4242,
        device="cpu",
        network_config=MICRO_NET_CONFIG,
        refresh_poll_interval=0.2,
        pacing_poll_interval=0.2,
        ceiling_poll_interval=0.2,
    )

    try:

        def enough_progress():
            if len(list_published_versions(ckpt_dir)) < 3:  # v0 + >= 2 real publishes
                return False
            seen = set()
            for path in shard_dir.glob("shard-*.npz"):
                seen |= {r.model_version for r in read_shard(path, MICRO).records}
            return len(seen) >= 2

        assert _wait_until(enough_progress, timeout=120.0), (
            "no publishes / version diversity within the timeout"
        )

        # (d) SIGTERM one actor mid-run, then restart it under the same identity.
        killed_id = 0
        killed = launched.actors[killed_id]
        killed.terminate()  # SIGTERM -- ShutdownFlag catches it, not SIGKILL
        killed.join(60.0)
        assert not killed.is_alive()
        assert killed.exitcode == 0  # clean exit through its own should_stop path

        # actor_id is reused by the restart, so shard filenames alone cannot
        # distinguish "produced before the restart" from "produced after
        # it" -- snapshot the set only *after* the killed process's own
        # graceful shutdown (which flushes its own final in-flight shard) has
        # fully finished, then watch for a genuinely new member.
        shards_before_restart = {
            p.name for p in shard_dir.glob(f"shard-{run_id}-{killed_id}-*.npz")
        }

        restarted = start_actor_process(
            launched.ctx,
            game_factory=_build_micro_game,
            self_play=run_config.self_play,
            run_id=run_id,
            actor_id=killed_id,
            shard_dir=shard_dir,
            ckpt_dir=ckpt_dir,
            run_dir=run_dir,
            run_seed=4242,
            device="cpu",
            network_config=MICRO_NET_CONFIG,
            refresh_poll_interval=0.2,
            pacing_poll_interval=0.2,
        )
        launched.actors[killed_id] = restarted

        def restarted_produced_a_shard():
            current = {p.name for p in shard_dir.glob(f"shard-{run_id}-{killed_id}-*.npz")}
            return bool(current - shards_before_restart)

        assert _wait_until(restarted_produced_a_shard, timeout=90.0), (
            "the restarted actor never published a shard"
        )
    finally:
        launched.shutdown(timeout=60.0)

    for process in launched.all_processes():
        assert process.exitcode == 0  # every process shut down cleanly, none crashed/hung

    # --- post-shutdown: the artifact protocol held across the real run ------
    shard_paths = sorted(shard_dir.glob("shard-*.npz"))
    assert shard_paths
    names = [p.name for p in shard_paths]
    assert len(names) == len(set(names))  # (c) shard names unique across both actors

    per_actor_game_ids: set[tuple[str, int]] = set()
    versions_seen: set[int] = set()
    for path in shard_paths:
        data = read_shard(path, MICRO)
        by_game: dict[tuple, set[int]] = {}
        for record in data.records:
            by_game.setdefault(record.game_id, set()).add(record.model_version)
            versions_seen.add(record.model_version)
        for game_id, versions in by_game.items():
            assert len(versions) == 1, f"{game_id} mixed model versions {versions}"  # (b)
            key = (game_id[1], game_id[2])  # (actor_id, game_index)
            assert key not in per_actor_game_ids, f"duplicate identity {key}"  # (d)
            per_actor_game_ids.add(key)

    assert versions_seen  # (a) publishes flowed into actual play
    assert len(versions_seen) >= 2

    window = ReplayWindow(shard_dir, MICRO, capacity=100_000)
    window.rescan()
    assert {e.shard_id for e in window.shard_entries} == set(names)  # (c) all ingested cleanly

    # (e) shutdown never publishes -- only the rolling resume snapshot.
    assert resume_path(ckpt_dir).exists()
    for version in list_published_versions(ckpt_dir):
        load_checkpoint(published_checkpoint_path(ckpt_dir, version), MICRO)  # no torn write
