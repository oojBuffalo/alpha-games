"""The M3 run entrypoint: ``scripts/run_selfplay.py`` (§12 M3, issue #63).

Two layers:

1. **CLI smoke tests** (fast-ish, real ``core.ipc.launch_run`` multiprocessing
   over micro-Blokus with a tiny net, mirroring ``tests/test_ipc.py``'s own
   slow layer): launch produces a complete, provenance-recorded run; resume
   proceeds on an unchanged or non-materially-different config and refuses
   -- naming the field, no override -- on any material one; fork creates a
   new identity with recorded lineage and never touches the parent; an
   unknown config field is a loud error; two fresh single-actor runs from
   the same seed play an identical first game.
2. **The decisive test** (``slow``): a controlled kill −9 vs. an
   uninterrupted twin. A single-process, single-actor, lockstep harness
   (:func:`_lockstep_worker`) drives one real actor game and one real
   learner step per round -- reusing the exact ``ActorDriver``/
   ``LearnerDriver`` objects issue #61's own single-process integration
   layer drives by hand (free-running multiprocess actors/learner cannot be
   bit-equal across a kill -- manifest order batches by rescan timing, a
   documented property this repo already lives with in ``core.ipc``'s
   module docstring). The worker parks in an inert sleep loop after a
   chosen number of rounds fully durably complete, so the kill point is
   race-free and never lands inside the shard/checkpoint reserve-then-
   publish window (already covered by #54/#59). The resume half calls
   ``core.run_identity.resolve_resume`` for real -- the CLI's own
   config-continuity contract -- before continuing the same identity.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from core.actor import ActorDriver
from core.checkpoint import load_checkpoint, resume_path
from core.ipc import DEFAULT_SHUTDOWN_GRACE_SECONDS, build_actor_refresh
from core.learner import LearnerDriver
from core.metrics import iter_epoch_records
from core.network import NetworkConfig
from core.observability import (
    KIND_SEGMENT_END,
    KIND_SEGMENT_START,
    KIND_SPIKE_STOP,
    SERIES_GAMES_COMPLETED,
    SERIES_LEARNER_STEP,
    reduce_run,
    reduce_window,
)
from core.replay_shard import read_shard
from core.run_identity import (
    LAUNCH_SCHEMA_VERSION,
    LaunchConfig,
    MaterialConfigDiffError,
    RunRecord,
    iso_now,
    load_launch_config,
    read_run_record,
    read_stored_config,
    resolve_resume,
    run_root,
    write_provenance,
)
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import MICRO_CONFIG
from tests.test_acceptance_checker import _build_passing_run
from tests.test_run_identity import V1_CONFIG_FIXTURE

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_selfplay.py"


def _load_run_selfplay():
    """Import ``scripts/run_selfplay.py`` as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("run_selfplay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rs = _load_run_selfplay()

MICRO = BlokusDuo(config=MICRO_CONFIG)


def _micro_net_config() -> NetworkConfig:
    """A tiny net over the micro-Blokus encoding surface -- speed, not throughput."""
    base = NetworkConfig.from_game(MICRO)
    return dataclasses.replace(base, trunk_blocks=1, trunk_channels=4)


MICRO_NET_CONFIG = _micro_net_config()


def _base_raw(tmp_path: Path, **training_overrides) -> dict:
    """The pinned micro config, tiny-fied and given a valid launcher block.

    ``replay_warmup_positions`` is kept deliberately high so the D5
    replay-ratio ceiling/floor enforcement stays dormant for these tiny
    bounded runs (mirrors ``tests/test_ipc.py``'s own
    ``_acceptance_run_config``) -- without it, a handful of games can trip
    the ceiling/floor and deadlock a max-games/max-steps-bounded run well
    before it reaches its own stop condition.
    """
    raw = json.loads((ROOT / "configs" / "blokus_micro.json").read_text())
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    raw["self_play"] = {
        **raw["self_play"],
        "sims": 128,
    }  # D6 validate tier; ActorDriver requires it
    training = dict(raw["training"])
    training.update(
        publish_interval=2,
        checkpoint_count=3,
        replay_warmup_positions=10_000,
        batch_size=4,
        replay_window=2000,
        learning_rate=1e-3,
        warmup_steps=0,
        cosine_total_steps=100,
    )
    training.update(training_overrides)
    raw["training"] = training
    raw["run_dir"] = str(tmp_path / "runs")
    raw["num_actors"] = 1
    raw["device"] = "cpu"
    raw["schema_version"] = LAUNCH_SCHEMA_VERSION
    raw["runtime"] = {
        "refresh_poll_interval": 0.02,
        "pacing_poll_interval": 0.02,
        "ceiling_poll_interval": 0.02,
    }
    return raw


def _write_config(path: Path, raw: dict) -> Path:
    path.write_text(json.dumps(raw))
    return path


def _only_run_dir(tmp_path: Path) -> Path:
    dirs = list((tmp_path / "runs").iterdir())
    assert len(dirs) == 1, dirs
    return dirs[0]


def _shard_count(run_dir: Path) -> int:
    return len(list((run_dir / "shards").glob("*.npz")))


# ==============================================================================
# 1. CLI smoke tests
# ==============================================================================


def test_launch_end_to_end_produces_a_complete_provenanced_run(tmp_path):
    raw = _base_raw(tmp_path)
    cfg = _write_config(tmp_path / "cfg.json", raw)

    launched = rs.cmd_launch(
        cfg,
        max_games_per_actor=3,
        max_learner_steps=4,
        block=True,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
    )
    for p in launched.all_processes():
        assert p.exitcode == 0

    run_dir = _only_run_dir(tmp_path)
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "run_record.json").is_file()
    assert _shard_count(run_dir) == 3
    assert list((run_dir / "checkpoints").glob("ckpt-*.pt"))

    stored = read_stored_config(run_dir)
    assert stored == LaunchConfig.from_dict(raw)
    record = read_run_record(run_dir)
    assert record.lineage is None
    assert record.entry_condition["exit_test"]["issue"].endswith("/issues/50")
    assert record.entry_condition["throughput_gate"]["issue"].endswith("/issues/66")


def test_resume_with_unchanged_config_proceeds(tmp_path):
    raw = _base_raw(tmp_path)
    cfg = _write_config(tmp_path / "cfg.json", raw)
    rs.cmd_launch(
        cfg,
        max_games_per_actor=2,
        max_learner_steps=2,
        block=True,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
    )
    run_dir = _only_run_dir(tmp_path)
    before = _shard_count(run_dir)

    launched = rs.cmd_resume(
        cfg,
        run_dir,
        max_games_per_actor=2,
        max_learner_steps=2,
        block=True,
        network_config=MICRO_NET_CONFIG,
    )
    for p in launched.all_processes():
        assert p.exitcode == 0
    assert _shard_count(run_dir) == before + 2
    assert read_run_record(run_dir).run_id == run_dir.name  # identity never changed


def test_resume_with_non_material_diff_proceeds_and_logs_it(tmp_path, capsys):
    raw = _base_raw(tmp_path)
    cfg = _write_config(tmp_path / "cfg.json", raw)
    rs.cmd_launch(
        cfg,
        max_games_per_actor=1,
        max_learner_steps=1,
        block=True,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
    )
    run_dir = _only_run_dir(tmp_path)

    raw2 = dict(raw)
    raw2["runtime"] = {
        **raw["runtime"],
        "refresh_poll_interval": raw["runtime"]["refresh_poll_interval"] + 1,
    }
    cfg2 = _write_config(tmp_path / "cfg2.json", raw2)

    launched = rs.cmd_resume(
        cfg2,
        run_dir,
        max_games_per_actor=1,
        max_learner_steps=1,
        block=True,
        network_config=MICRO_NET_CONFIG,
    )
    for p in launched.all_processes():
        assert p.exitcode == 0
    out = capsys.readouterr().out
    assert "runtime.refresh_poll_interval" in out
    assert "proceeding" in out


@pytest.mark.parametrize(
    "mutate,field_name",
    [
        (lambda raw: raw.__setitem__("run_seed", raw["run_seed"] + 1), "run_seed"),
        (lambda raw: raw.__setitem__("num_actors", raw["num_actors"] + 1), "num_actors"),
        (
            lambda raw: raw["self_play"].__setitem__("k_temp", raw["self_play"]["k_temp"] + 1),
            "self_play.k_temp",
        ),
        (
            lambda raw: raw["training"].__setitem__(
                "batch_size", raw["training"]["batch_size"] + 1
            ),
            "training.batch_size",
        ),
    ],
    ids=["run_seed", "num_actors", "k_temp", "batch_size"],
)
def test_resume_refuses_on_material_diff_naming_the_exact_field(tmp_path, mutate, field_name):
    raw = _base_raw(tmp_path)
    cfg = _write_config(tmp_path / "cfg.json", raw)
    rs.cmd_launch(
        cfg,
        max_games_per_actor=1,
        max_learner_steps=1,
        block=True,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
    )
    run_dir = _only_run_dir(tmp_path)
    before = _shard_count(run_dir)

    raw2 = json.loads(json.dumps(raw))  # deep copy
    mutate(raw2)
    cfg2 = _write_config(tmp_path / "cfg2.json", raw2)

    with pytest.raises(MaterialConfigDiffError) as exc_info:
        rs.cmd_resume(cfg2, run_dir, block=False, network_config=MICRO_NET_CONFIG)
    assert field_name in exc_info.value.material
    assert field_name in str(exc_info.value)
    # No override flag exists anywhere on this surface.
    import inspect

    assert "force" not in inspect.signature(rs.cmd_resume).parameters
    assert "override" not in inspect.signature(rs.cmd_resume).parameters
    # Refusal happens before anything is (re)started -- no new shard appeared.
    assert _shard_count(run_dir) == before


def test_fork_creates_new_identity_with_lineage_and_leaves_parent_untouched(tmp_path):
    raw = _base_raw(tmp_path)
    cfg = _write_config(tmp_path / "cfg.json", raw)
    rs.cmd_launch(
        cfg,
        max_games_per_actor=1,
        max_learner_steps=1,
        block=True,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
    )
    parent_dir = _only_run_dir(tmp_path)
    parent_shards_before = _shard_count(parent_dir)
    parent_config_bytes = (parent_dir / "config.json").read_text()

    raw2 = dict(raw)
    raw2["run_seed"] = raw["run_seed"] + 12345  # forks may freely change material fields
    cfg2 = _write_config(tmp_path / "cfg2.json", raw2)

    rs.cmd_fork(
        cfg2,
        parent_dir,
        max_games_per_actor=1,
        max_learner_steps=1,
        block=True,
        now=1_700_000_500.0,
        network_config=MICRO_NET_CONFIG,
    )

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 2
    fork_dir = next(d for d in run_dirs if d != parent_dir)

    fork_record = read_run_record(fork_dir)
    assert fork_record.lineage is not None
    assert fork_record.lineage.parent_run_id == read_run_record(parent_dir).run_id
    assert fork_record.lineage.parent_run_dir == str(parent_dir)
    assert fork_record.lineage.imported_weights_version is None

    fork_stored = read_stored_config(fork_dir)
    assert fork_stored.run.run_seed == raw2["run_seed"]

    # The parent is provably untouched.
    assert (parent_dir / "config.json").read_text() == parent_config_bytes
    assert _shard_count(parent_dir) == parent_shards_before


def test_verify_acceptance_cli_reads_a_schema_v1_run_dir_end_to_end(tmp_path, capsys):
    """An M3-era run dir verifies through the CLI: the loader, not just the checker.

    The run directory is the acceptance checker's own passing fixture, written
    for the full game, under the genuine schema-v1 ``config.json`` (no
    ``self_play.leaf_batch_size``) -- which the strict launch loader rejects.
    """
    full = BlokusDuo()
    run_id = "blokus_duo-20260825T000000000000-1f18e71e6f6b"
    root = _build_passing_run(tmp_path, run_id, game=full, learner_step=400)
    (root / "config.json").write_bytes(V1_CONFIG_FIXTURE.read_bytes())
    (root / "run_record.json").write_text(
        json.dumps(
            RunRecord(
                run_id=run_id, created_at="2026-08-25T00:00:00Z", entry_condition={}
            ).to_dict()
        )
    )
    with pytest.raises(ValueError, match="schema_version mismatch"):
        load_launch_config(root / "config.json")

    assert rs.main(["--verify-acceptance", str(root)]) == 0

    out = capsys.readouterr().out
    assert out.count("[PASS]") == 9
    assert "[FAIL]" not in out
    assert "verdict: PASS" in out
    assert (root / "config.json").read_bytes() == V1_CONFIG_FIXTURE.read_bytes()


def test_unknown_config_field_is_a_loud_error(tmp_path):
    raw = _base_raw(tmp_path)
    raw["num_actorz"] = 4  # typo
    cfg = _write_config(tmp_path / "cfg.json", raw)
    with pytest.raises(ValueError, match="unknown config keys"):
        rs.cmd_launch(cfg, block=False)


def test_material_diff_error_has_no_way_to_be_overridden_via_the_cli_argparser():
    parser = rs.build_arg_parser()
    help_text = parser.format_help()
    assert "--force" not in help_text
    assert "--override" not in help_text


# ==============================================================================
# 1b. The run-wide spike stop (§12 M5): --spike-games
# ==============================================================================


def _orchestrator_records(run_dir: Path) -> list[dict]:
    return list(iter_epoch_records(run_dir, "orchestrator"))


def _wait_bounded(launched, timeout: float) -> bool:
    """Wait for a run on a helper thread, so a run that never stops fails instead of hanging.

    Args:
        launched: The live run (started with ``block=False``).
        timeout: Seconds to allow before giving up and killing every child.

    Returns:
        Whether ``wait_for_completion`` returned within ``timeout``.
    """
    thread = threading.Thread(
        target=rs.wait_for_completion, args=(launched,), kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        for process in launched.all_processes():
            process.kill()
        return False
    return True


def _assert_every_process_stopped_cleanly(launched) -> None:
    """Assert every child is gone and none crashed or had to be killed.

    A child signaled while already exiting on its own (its cap reached, its
    work complete) reports ``-SIGTERM``, not ``0``: the interpreter restores
    the default handler early in its teardown.

    Args:
        launched: The run, after its wait returned.
    """
    for process in launched.all_processes():
        assert not process.is_alive()
        assert process.exitcode in (0, -signal.SIGTERM), process


def test_spike_games_stops_the_whole_run_and_records_the_stop(tmp_path):
    # The learner's own stop is out of reach, so every process is mid-run at the stop.
    raw = _base_raw(tmp_path, publish_interval=1000, checkpoint_count=1000)
    raw["num_actors"] = 2
    cfg = _write_config(tmp_path / "cfg.json", raw)
    backstop = 40  # per actor: only reached if the spike stop never fires

    launched = rs.cmd_launch(
        cfg,
        spike_games=6,
        max_games_per_actor=backstop,
        block=False,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
    )
    assert _wait_bounded(launched, 120.0), "the run did not stop"
    _assert_every_process_stopped_cleanly(launched)
    assert launched.learner.exitcode == 0  # signaled mid-run: its own should_stop path

    run_dir = _only_run_dir(tmp_path)
    total = reduce_run(run_dir).totals[SERIES_GAMES_COMPLETED]
    records = _orchestrator_records(run_dir)
    (stop,) = [rec for rec in records if rec["kind"] == KIND_SPIKE_STOP]
    assert stop["target"] == 6
    # A signaled actor finishes its in-flight game, so the final total may exceed
    # the total observed at the stop; neither is below the target.
    assert 6 <= stop["games_completed"] <= total < 2 * backstop
    assert _shard_count(run_dir) == total  # every counted game's shard was published
    # No child had to be escalated, and the segment closed after all of them.
    assert [rec["kind"] for rec in records] == [
        KIND_SEGMENT_START,
        KIND_SPIKE_STOP,
        KIND_SEGMENT_END,
    ]
    child_timestamps = [
        rec["timestamp"]
        for proc in ("learner", "actor-0", "actor-1")
        for rec in iter_epoch_records(run_dir, proc)
    ]
    assert records[-1]["timestamp"] >= max(child_timestamps)
    assert reduce_run(run_dir).gpu_hours > 0  # the segment is complete

    # The stopped run's own records are what the windowed reducer reads: real
    # actor flushes group into games, and every validate-tier series is found.
    window = reduce_window(run_dir, warmup_games=1, measure_games=3, replay_warmup_positions=1)
    assert window.reason is None
    assert (window.start_game_ordinal, window.end_game_ordinal, window.games_in_window) == (1, 4, 3)
    assert window.stored_positions_per_game == window.played_positions_per_game > 0
    assert window.inference_batches == sum(window.inference_batch_sizes.values()) > 0
    assert window.consecutive_mover_transitions is not None
    assert (window.actor_amp_active, window.is_cuda) == (False, False)  # a CPU run
    tier_fields = ("moves_full_512", "moves_mid_256", "moves_fast_64", "mid_tier_fraction")
    assert all(name in window.unavailable for name in tier_fields)  # no schedule ran


def test_spike_games_terminates_a_run_whose_learner_has_no_stop_of_its_own(tmp_path):
    """The hang case: capped actors stop feeding shards, the learner waits at its ceiling.

    Both actors exit at their cap; the learner's pinned stop is out of reach
    and its D5 ceiling is live, so once it has caught up with the shards it
    waits for more forever. Only the run-wide stop can end this run.
    """
    raw = _base_raw(
        tmp_path, replay_warmup_positions=1, publish_interval=1000, checkpoint_count=1000
    )
    raw["num_actors"] = 2
    cfg = _write_config(tmp_path / "cfg.json", raw)

    launched = rs.cmd_launch(
        cfg,
        spike_games=6,
        max_games_per_actor=3,
        block=False,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
    )
    assert _wait_bounded(launched, 120.0), "the run hung with its learner at the ceiling"
    _assert_every_process_stopped_cleanly(launched)
    assert launched.learner.exitcode == 0  # it left its ceiling wait on the signal

    run_dir = _only_run_dir(tmp_path)
    reduced = reduce_run(run_dir)
    assert reduced.totals[SERIES_GAMES_COMPLETED] == 6  # exactly the caps: 2 actors x 3
    assert 0 < reduced.totals[SERIES_LEARNER_STEP] < 1000 * 1000  # stopped, not finished
    (stop,) = [rec for rec in _orchestrator_records(run_dir) if rec["kind"] == KIND_SPIKE_STOP]
    assert (stop["target"], stop["games_completed"]) == (6, 6)


def test_spike_games_ends_a_run_whose_actors_all_exit_below_the_target(tmp_path):
    """The target is out of reach: both actors are capped below it and the learner waits.

    Nothing can complete another game, so the run is shut down rather than
    polled forever -- and no ``spike_stop`` is recorded, since none happened.
    """
    raw = _base_raw(
        tmp_path, replay_warmup_positions=1, publish_interval=1000, checkpoint_count=1000
    )
    raw["num_actors"] = 2
    cfg = _write_config(tmp_path / "cfg.json", raw)

    launched = rs.cmd_launch(
        cfg,
        spike_games=50,
        max_games_per_actor=1,
        block=False,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
    )
    assert _wait_bounded(launched, 120.0), "the run polled on for a target it could not reach"
    _assert_every_process_stopped_cleanly(launched)

    run_dir = _only_run_dir(tmp_path)
    assert reduce_run(run_dir).totals[SERIES_GAMES_COMPLETED] == 2  # the caps: 2 actors x 1
    kinds = [rec["kind"] for rec in _orchestrator_records(run_dir)]
    assert kinds == [KIND_SEGMENT_START, KIND_SEGMENT_END]


def test_a_run_without_a_spike_target_records_only_its_gpu_segment(tmp_path):
    cfg = _write_config(tmp_path / "cfg.json", _base_raw(tmp_path))
    rs.cmd_launch(
        cfg,
        max_games_per_actor=2,
        max_learner_steps=2,
        block=True,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
        shutdown_grace_seconds=120.0,
    )
    kinds = [rec["kind"] for rec in _orchestrator_records(_only_run_dir(tmp_path))]
    assert kinds == [KIND_SEGMENT_START, KIND_SEGMENT_END]


@pytest.mark.parametrize(
    "mode_args,command",
    [([], "cmd_launch"), (["--resume", "run"], "cmd_resume"), (["--fork-from", "run"], "cmd_fork")],
    ids=["launch", "resume", "fork"],
)
def test_the_cli_threads_the_stop_arguments_into_every_launch_mode(monkeypatch, mode_args, command):
    seen = {}
    monkeypatch.setattr(rs, command, lambda *args, **kwargs: seen.update(kwargs))

    flags = ["--spike-games", "310", "--shutdown-grace-seconds", "900"]
    assert rs.main(["cfg.json", *mode_args, *flags]) == 0
    assert (seen["spike_games"], seen["shutdown_grace_seconds"]) == (310, 900.0)

    seen.clear()
    assert rs.main(["cfg.json", *mode_args]) == 0
    assert seen["spike_games"] is None  # unset by default: the M3 paths are unchanged
    assert seen["shutdown_grace_seconds"] == DEFAULT_SHUTDOWN_GRACE_SECONDS


@pytest.mark.parametrize(
    "stop_args",
    [
        ["--spike-games", "0"],
        ["--spike-games", "-3"],
        ["--shutdown-grace-seconds", "-1"],
        ["--shutdown-grace-seconds", "nan"],
        ["--shutdown-grace-seconds", "inf"],
        ["--spike-games", "3", "--no-block"],  # nothing would be left to enforce the target
    ],
    ids=" ".join,
)
def test_the_cli_refuses_an_unusable_stop_argument_before_creating_a_run(
    tmp_path, capsys, monkeypatch, stop_args
):
    cfg = _write_config(tmp_path / "cfg.json", _base_raw(tmp_path))
    # Were one accepted, the test fails without starting a process it would leave behind.
    monkeypatch.setattr(rs, "cmd_launch", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit) as exc_info:
        rs.main([str(cfg), *stop_args])
    assert exc_info.value.code == 2
    assert stop_args[0] in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()  # refused before any provenance was written


def test_two_fresh_single_actor_runs_from_the_same_seed_play_an_identical_first_game(tmp_path):
    """Task 2's promise, exercised end to end through the real CLI launch path.

    ``publish_interval`` is set high enough that neither run can possibly
    publish a version beyond the mandatory seeded v0 within its bounded
    ``max_learner_steps`` -- so both runs' single actor necessarily plays its
    one game against the exact same (seed-derived) v0 weights, and the only
    remaining source of divergence would be the seeding itself.
    """
    raw_a = _base_raw(tmp_path / "a", publish_interval=10_000, checkpoint_count=1)
    raw_a["run_dir"] = str(tmp_path / "a" / "runs")
    cfg_a = _write_config(tmp_path / "cfg_a.json", raw_a)
    rs.cmd_launch(
        cfg_a,
        max_games_per_actor=1,
        max_learner_steps=1,
        block=True,
        now=1_700_000_000.0,
        network_config=MICRO_NET_CONFIG,
    )
    run_dir_a = _only_run_dir(tmp_path / "a")

    raw_b = _base_raw(tmp_path / "b", publish_interval=10_000, checkpoint_count=1)
    raw_b["run_dir"] = str(tmp_path / "b" / "runs")
    assert raw_b["run_seed"] == raw_a["run_seed"]
    cfg_b = _write_config(tmp_path / "cfg_b.json", raw_b)
    rs.cmd_launch(
        cfg_b,
        max_games_per_actor=1,
        max_learner_steps=1,
        block=True,
        now=1_700_000_999.0,  # a different launch time -> a different run_id
        network_config=MICRO_NET_CONFIG,
    )
    run_dir_b = _only_run_dir(tmp_path / "b")

    assert run_dir_a.name != run_dir_b.name  # distinct run identities, as expected

    (shard_a,) = sorted((run_dir_a / "shards").glob("*.npz"))
    (shard_b,) = sorted((run_dir_b / "shards").glob("*.npz"))
    records_a = read_shard(shard_a, MICRO).records
    records_b = read_shard(shard_b, MICRO).records

    assert len(records_a) == len(records_b)
    for ra, rb in zip(records_a, records_b, strict=True):
        assert ra.model_version == rb.model_version == 0
        assert ra.mover == rb.mover
        assert ra.ply == rb.ply
        assert ra.sparse_pi == rb.sparse_pi
        assert ra.z == rb.z
        assert ra.aux == rb.aux
        assert (ra.planes == rb.planes).all()


# ==============================================================================
# 2. The decisive test: kill -9 vs. an uninterrupted twin
# ==============================================================================

_TWIN_NUM_ROUNDS = 6
_TWIN_KILL_AFTER_ROUND = 3  # rounds 0, 1, 2 complete durably, then the worker parks


def _twin_raw(tmp_path: Path) -> dict:
    """A tiny, real micro-Blokus config for the twin harness.

    ``publish_interval`` is set above ``_TWIN_NUM_ROUNDS`` so no version
    beyond the mandatory seeded v0 is ever published within either twin --
    the whole comparison then never has to reason about a publish boundary
    at all, deliberately distinct from (and simpler than) the reserve-then-
    publish crash window #54/#59 already cover.
    """
    raw = _base_raw(
        tmp_path,
        publish_interval=1000,
        checkpoint_count=1,
        replay_warmup_positions=10_000,
        batch_size=2,
    )
    return raw


def _lockstep_worker(
    *,
    shard_dir: Path,
    ckpt_dir: Path,
    run_dir: Path,
    run_config,
    run_id: str,
    run_seed: int,
    network_config: NetworkConfig,
    num_rounds: int,
    park_after_round: int | None,
    control_dir: Path,
) -> None:
    """Play ``num_rounds`` (one game, one learner step) rounds in strict lockstep.

    A real ``multiprocessing.Process`` target (module-level, picklable --
    ``core.ipc``'s own process-model rule). One actor and the learner driven
    by hand in a single process: no pacing hook, no concurrent scheduling --
    exactly ``tests/test_ipc.py``'s single-process integration layer, reused
    here as the deterministic harness the kill-9 comparison needs.

    After each round's actor game and learner step both durably complete
    (shard published, checkpoint-if-due published, resume snapshot
    written), a ``round-<i>.done`` marker file is written. If ``i + 1 ==
    park_after_round``, the worker then blocks in an inert sleep loop
    instead of starting the next round, forever (until killed) -- an
    unambiguous, race-free point to signal it: every durable write for the
    completed rounds already happened, and nothing else happens until a
    signal arrives.

    Args:
        shard_dir: Shared shard directory.
        ckpt_dir: Shared checkpoint directory.
        run_dir: Shared run root (pacing file, metrics).
        run_config: The learner's full protocol (``core.runconfig.RunConfig``).
        run_id: This run's identity.
        run_seed: This run's recorded seed.
        network_config: Tiny network architecture, for speed.
        num_rounds: Total rounds to attempt (unless parked first).
        park_after_round: Park indefinitely after this many rounds complete,
            or ``None`` to run all ``num_rounds`` and exit normally.
        control_dir: Directory the caller polls for ``round-<i>.done``
            marker files.
    """
    game = BlokusDuo(config=MICRO_CONFIG)
    refresh = build_actor_refresh(game=game, ckpt_dir=ckpt_dir, network_config=network_config)
    actor = ActorDriver(
        game=game,
        self_play=run_config.self_play,
        run_id=run_id,
        actor_id=0,
        out_dir=shard_dir,
        run_seed=run_seed,
        refresh=refresh,
        max_games=1,
    )
    learner = LearnerDriver(
        game=game,
        run_config=run_config,
        shard_dir=shard_dir,
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
        network_config=network_config,
    )
    for i in range(num_rounds):
        actor.run()
        learner._run_step()
        (control_dir / f"round-{i}.done").write_text("")
        if park_after_round is not None and i + 1 == park_after_round:
            while True:
                time.sleep(0.05)
    (control_dir / "worker-finished").write_text("")


def _wait_for_file(path: Path, timeout: float, interval: float = 0.05) -> bool:
    """Poll for a file's existence.

    Args:
        path: The file to wait for.
        timeout: Seconds to keep polling.
        interval: Seconds between polls.

    Returns:
        ``True`` once ``path`` exists, ``False`` if ``timeout`` elapsed first.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(interval)
    return path.exists()


def _run_lockstep_process(ctx, **kwargs):
    proc = ctx.Process(target=_lockstep_worker, kwargs=kwargs, name=f"lockstep-{kwargs['run_id']}")
    proc.start()
    return proc


def _shard_records_by_seq(shard_dir: Path) -> dict:
    by_seq = {}
    for path in sorted(shard_dir.glob("shard-*.npz")):
        seq = int(path.stem.rsplit("-", 1)[1])
        by_seq[seq] = read_shard(path, MICRO)
    return by_seq


@pytest.mark.slow
def test_kill_nine_resume_twin_matches_an_uninterrupted_run(tmp_path):
    raw = _twin_raw(tmp_path)
    launch_config = LaunchConfig.from_dict(raw)
    ctx = multiprocessing.get_context("spawn")

    # --- Twin A: launched, killed mid-run, resumed through the real
    # core.run_identity.resolve_resume path, then run to completion. ---
    killed_run_id = "twin-killed"
    killed_root = run_root(launch_config, killed_run_id)
    write_provenance(
        killed_root,
        launch_config,
        RunRecord(run_id=killed_run_id, created_at=iso_now(1_700_000_000.0), entry_condition={}),
    )
    control_dir = killed_root / "control"
    control_dir.mkdir(parents=True)

    proc = _run_lockstep_process(
        ctx,
        shard_dir=killed_root / "shards",
        ckpt_dir=killed_root / "checkpoints",
        run_dir=killed_root,
        run_config=launch_config.run,
        run_id=killed_run_id,
        run_seed=launch_config.run.run_seed,
        network_config=MICRO_NET_CONFIG,
        num_rounds=_TWIN_NUM_ROUNDS,
        park_after_round=_TWIN_KILL_AFTER_ROUND,
        control_dir=control_dir,
    )
    try:
        marker = control_dir / f"round-{_TWIN_KILL_AFTER_ROUND - 1}.done"
        assert _wait_for_file(marker, timeout=90.0), (
            "worker never durably completed its pre-kill rounds"
        )
        time.sleep(0.3)  # safety margin: let the worker settle into its parked, inert sleep loop
        assert proc.is_alive(), "worker exited on its own before it could be killed"
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(30.0)
        assert not proc.is_alive()
        assert proc.exitcode != 0  # killed, not a clean voluntary exit
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5.0)

    # Simulate leftover temp artifacts from a hypothetical mid-write crash --
    # the resume path must ignore them (never enumerated by the shard glob /
    # published-checkpoint pattern -- core.replay_shard / core.checkpoint's
    # own contract), not merely happen not to encounter one here.
    (killed_root / "shards" / "shard-stray.npz.tmp-deadbeef").write_bytes(b"not a real shard")
    (killed_root / "checkpoints" / "ckpt-stray.pt.tmp-deadbeef").write_bytes(
        b"not a real checkpoint"
    )

    rounds_done = _TWIN_KILL_AFTER_ROUND
    assert _shard_count(killed_root) == rounds_done

    # Resume through the real CLI config-continuity contract.
    resolution = resolve_resume(killed_root, launch_config)
    assert resolution.non_material_diff == {}
    assert resolution.run_id == killed_run_id

    remaining = _TWIN_NUM_ROUNDS - rounds_done
    resume_control = killed_root / "control-resume"
    resume_control.mkdir()
    resume_proc = _run_lockstep_process(
        ctx,
        shard_dir=killed_root / "shards",
        ckpt_dir=killed_root / "checkpoints",
        run_dir=killed_root,
        run_config=resolution.effective_config.run,
        run_id=resolution.run_id,
        run_seed=resolution.effective_config.run.run_seed,
        network_config=MICRO_NET_CONFIG,
        num_rounds=remaining,
        park_after_round=None,
        control_dir=resume_control,
    )
    resume_proc.join(90.0)
    assert resume_proc.exitcode == 0
    assert (resume_control / "worker-finished").exists()

    # The stray temp files were ignored, never adopted as real artifacts --
    # left on disk untouched (nothing here deletes an unrelated file), never
    # counted as a shard or a published version (core.replay_shard's shard
    # glob / core.checkpoint's published-version pattern both exclude a
    # ".tmp-<uuid>" suffix by construction).
    assert _shard_count(killed_root) == _TWIN_NUM_ROUNDS
    assert (killed_root / "shards" / "shard-stray.npz.tmp-deadbeef").exists()
    assert (killed_root / "checkpoints" / "ckpt-stray.pt.tmp-deadbeef").exists()

    # --- Twin B: uninterrupted, same seed, its own run identity. ---
    twin_run_id = "twin-uninterrupted"
    twin_root = run_root(launch_config, twin_run_id)
    write_provenance(
        twin_root,
        launch_config,
        RunRecord(run_id=twin_run_id, created_at=iso_now(1_700_000_000.0), entry_condition={}),
    )
    twin_control = twin_root / "control"
    twin_control.mkdir(parents=True)
    twin_proc = _run_lockstep_process(
        ctx,
        shard_dir=twin_root / "shards",
        ckpt_dir=twin_root / "checkpoints",
        run_dir=twin_root,
        run_config=launch_config.run,
        run_id=twin_run_id,
        run_seed=launch_config.run.run_seed,
        network_config=MICRO_NET_CONFIG,
        num_rounds=_TWIN_NUM_ROUNDS,
        park_after_round=None,
        control_dir=twin_control,
    )
    twin_proc.join(90.0)
    assert twin_proc.exitcode == 0
    assert (twin_control / "worker-finished").exists()

    # --- Compare durable artifacts -------------------------------------------

    assert _shard_count(killed_root) == _shard_count(twin_root) == _TWIN_NUM_ROUNDS

    killed_by_seq = _shard_records_by_seq(killed_root / "shards")
    twin_by_seq = _shard_records_by_seq(twin_root / "shards")
    assert (
        set(killed_by_seq) == set(twin_by_seq) == set(range(_TWIN_NUM_ROUNDS))
    )  # (identical manifest order)

    for seq in sorted(killed_by_seq):
        killed_data = killed_by_seq[seq]
        twin_data = twin_by_seq[seq]
        assert len(killed_data.records) == len(twin_data.records)
        for kr, tr in zip(killed_data.records, twin_data.records, strict=True):
            assert kr.game_id[2] == tr.game_id[2] == seq  # identical game ids
            assert kr.mover == tr.mover
            assert kr.ply == tr.ply
            assert kr.model_version == tr.model_version
            assert kr.sparse_pi == tr.sparse_pi  # identical per-game moves (the policy target)
            assert kr.z == tr.z
            assert kr.aux == tr.aux
            assert (kr.planes == tr.planes).all()

    # Bit-equal final parameters (torch.equal, no tolerances).
    killed_final = load_checkpoint(resume_path(killed_root / "checkpoints"), MICRO)
    twin_final = load_checkpoint(resume_path(twin_root / "checkpoints"), MICRO)
    assert killed_final.learner_step == twin_final.learner_step == _TWIN_NUM_ROUNDS
    for key, killed_tensor in killed_final.model_state_dict.items():
        assert torch.equal(killed_tensor, twin_final.model_state_dict[key]), key
    for key, killed_tensor in killed_final.optimizer_state_dict.get("state", {}).items():
        twin_tensor = twin_final.optimizer_state_dict["state"][key]
        if isinstance(killed_tensor, dict):
            for sub_key, sub_val in killed_tensor.items():
                if torch.is_tensor(sub_val):
                    assert torch.equal(sub_val, twin_tensor[sub_key]), (key, sub_key)
