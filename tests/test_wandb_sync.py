"""``scripts/wandb_sync.py``: the optional W&B mirror over the run-dir metrics contract (issue #90).

Mirror-only: this script never writes to ``<run_dir>/metrics/`` and never
changes ``core.actor``/``core.learner``'s write path -- it only reads through
``core.metrics``/``core.observability``/``core.run_identity``'s existing,
frozen contracts and pushes them into Weights & Biases. Layers, cheapest
first:

1. **Sync-state persistence** -- the local sidecar this tool's own
   idempotency is built on (per-process record cursors, synced checkpoint
   versions, running actor cumulative sums). Pure JSON round-trip, no wandb.
2. **The guarded import** -- ``wandb`` is strictly opt-in; a missing install
   raises a clear, install-hinted ``ImportError`` rather than an ugly
   traceback. Tested by patching ``importlib.import_module`` so this runs
   with no real "wandb missing" environment needed.
3. **Record splitting** -- ``_split_learner_records``/``_split_actor_records``
   turn a process's raw ``core.metrics`` records into per-flush groups,
   holding back an unfinalized trailing group so a still-being-written flush
   is never logged half-complete. Pure, no wandb, no filesystem.
4. **Real wandb, offline mode** -- ``WANDB_MODE=offline`` writes locally with
   no network call (``pytest.importorskip("wandb")``-gated, so the whole
   battery still passes without the optional extra installed): run
   construction (id/config/tags/summary), and the idempotent end-to-end
   sync -- syncing the same run dir twice must not re-log anything.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from core.learner import SERIES_LOSS_POLICY, SERIES_LOSS_TOTAL, SERIES_LOSS_VALUE
from core.metrics import EpochMetricsWriter
from core.observability import (
    CHECKPOINT_PUBLISHED_KIND,
    SERIES_GAMES_COMPLETED,
    SERIES_LEARNER_STEP,
    SERIES_POSITIONS_EVALUATED,
    SERIES_SIMS_RUN,
    delta_record,
    gauge_record,
    total_record,
)
from core.run_identity import (
    ENTRY_CONDITION,
    LAUNCH_SCHEMA_VERSION,
    LaunchConfig,
    RunRecord,
    iso_now,
    run_root,
    write_provenance,
)
from core.runconfig import MICRO_RUN_CONFIG_PATH

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "wandb_sync.py"


def load_wandb_sync():
    """Import ``scripts/wandb_sync.py`` as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("wandb_sync", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ws = load_wandb_sync()


def _launch_raw(**overrides):
    """The pinned micro config plus a valid launcher block, doc keys stripped."""
    raw = json.loads(MICRO_RUN_CONFIG_PATH.read_text())
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    raw["num_actors"] = 1
    raw["device"] = "cpu"
    raw["schema_version"] = LAUNCH_SCHEMA_VERSION
    raw["runtime"] = {
        "refresh_poll_interval": 1.0,
        "pacing_poll_interval": 1.0,
        "ceiling_poll_interval": 1.0,
    }
    raw.update(overrides)
    return raw


def _launch_config(**overrides) -> LaunchConfig:
    return LaunchConfig.from_dict(_launch_raw(**overrides))


def _write_run(tmp_path, *, run_id="run-1", now=1_700_000_000.0) -> Path:
    """Write a minimal but real run dir (provenance only; caller adds metrics)."""
    lc = _launch_config(run_dir=str(tmp_path / "runs"))
    root = run_root(lc, run_id)
    write_provenance(
        root,
        lc,
        RunRecord(run_id=run_id, created_at=iso_now(now), entry_condition=ENTRY_CONDITION),
    )
    return root


# ==============================================================================
# 1. Sync-state persistence
# ==============================================================================


def test_sync_state_round_trips_through_to_dict_from_dict():
    state = ws.SyncState(
        proc_cursors={"learner": 4, "actor-0": 2},
        checkpoint_versions_synced=[0, 1],
        actor_totals={SERIES_GAMES_COMPLETED: 3.0},
    )
    assert ws.SyncState.from_dict(state.to_dict()) == state


def test_load_sync_state_is_empty_for_a_run_dir_with_no_sidecar(tmp_path):
    state = ws.load_sync_state(tmp_path)
    assert state == ws.SyncState()


def test_save_then_load_sync_state_round_trips(tmp_path):
    state = ws.SyncState(
        proc_cursors={"learner": 7},
        checkpoint_versions_synced=[0],
        actor_totals={SERIES_SIMS_RUN: 512.0},
    )
    ws.save_sync_state(tmp_path, state)
    assert (tmp_path / ws.SYNC_STATE_FILENAME).exists()
    assert ws.load_sync_state(tmp_path) == state


# ==============================================================================
# 2. The guarded wandb import
# ==============================================================================


def test_require_wandb_returns_the_module_when_installed():
    pytest.importorskip("wandb")
    wandb = ws._require_wandb()
    assert wandb.__name__ == "wandb"


def test_require_wandb_raises_install_hint_when_missing(monkeypatch):
    real_import_module = importlib.import_module

    def _fake_import_module(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(ws.importlib, "import_module", _fake_import_module)
    with pytest.raises(ImportError, match=r"pip install -e '\.\[wandb\]'"):
        ws._require_wandb()


# ==============================================================================
# 3. Record splitting
# ==============================================================================


def _learner_group(step, ts, *, loss_total=1.0, loss_value=0.5, loss_policy=0.5):
    return [
        total_record(SERIES_LEARNER_STEP, step, timestamp=ts),
        gauge_record(SERIES_LOSS_TOTAL, loss_total, timestamp=ts),
        gauge_record(SERIES_LOSS_VALUE, loss_value, timestamp=ts),
        gauge_record(SERIES_LOSS_POLICY, loss_policy, timestamp=ts),
    ]


def test_split_learner_records_empty_input():
    groups, checkpoints, consumed = ws._split_learner_records([], finalize=True)
    assert groups == []
    assert checkpoints == []
    assert consumed == 0


def test_split_learner_records_holds_back_an_unfinalized_trailing_group():
    records = _learner_group(1, 10.0)
    groups, checkpoints, consumed = ws._split_learner_records(records, finalize=False)
    assert groups == []
    assert checkpoints == []
    assert consumed == 0


def test_split_learner_records_finalize_true_emits_the_trailing_group():
    records = _learner_group(1, 10.0)
    groups, checkpoints, consumed = ws._split_learner_records(records, finalize=True)
    assert len(groups) == 1
    assert groups[0]["learner_step"] == 1
    assert groups[0]["timestamp"] == 10.0
    assert groups[0]["gauges"] == {
        SERIES_LOSS_TOTAL: 1.0,
        SERIES_LOSS_VALUE: 0.5,
        SERIES_LOSS_POLICY: 0.5,
    }
    assert checkpoints == []
    assert consumed == len(records)


def test_split_learner_records_a_second_total_record_finalizes_the_first_group():
    records = _learner_group(1, 10.0) + _learner_group(2, 11.0)
    groups, checkpoints, consumed = ws._split_learner_records(records, finalize=False)
    assert [g["learner_step"] for g in groups] == [1]
    assert consumed == 4  # only step 1's four records; step 2's group is still trailing


def test_split_learner_records_checkpoint_finalizes_the_preceding_group():
    checkpoint = {
        "kind": CHECKPOINT_PUBLISHED_KIND,
        "model_version": 0,
        "learner_step": 1,
        "timestamp": 10.5,
    }
    records = _learner_group(1, 10.0) + [checkpoint]
    groups, checkpoints, consumed = ws._split_learner_records(records, finalize=False)
    assert [g["learner_step"] for g in groups] == [1]
    assert checkpoints == [checkpoint]
    assert consumed == len(records)


def test_split_learner_records_checkpoint_before_any_group():
    checkpoint = {
        "kind": CHECKPOINT_PUBLISHED_KIND,
        "model_version": 0,
        "learner_step": 0,
        "timestamp": 9.0,
    }
    records = [checkpoint] + _learner_group(1, 10.0)
    groups, checkpoints, consumed = ws._split_learner_records(records, finalize=True)
    assert checkpoints == [checkpoint]
    assert [g["learner_step"] for g in groups] == [1]
    assert consumed == len(records)


def test_split_learner_records_cursor_replay_never_reprocesses_a_finalized_group():
    records = _learner_group(1, 10.0) + _learner_group(2, 11.0)
    groups1, _, consumed1 = ws._split_learner_records(records, finalize=False)
    assert [g["learner_step"] for g in groups1] == [1]
    remaining = records[consumed1:]
    groups2, _, consumed2 = ws._split_learner_records(remaining, finalize=True)
    assert [g["learner_step"] for g in groups2] == [2]
    assert consumed1 + consumed2 == len(records)


def _actor_group(ts, *, games=1, sims=32, positions=None):
    recs = [
        delta_record(SERIES_GAMES_COMPLETED, games, timestamp=ts),
        delta_record(SERIES_SIMS_RUN, sims, timestamp=ts),
    ]
    if positions is not None:
        recs.append(delta_record(SERIES_POSITIONS_EVALUATED, positions, timestamp=ts))
    return recs


def test_split_actor_records_holds_back_trailing_group_unless_finalized():
    records = _actor_group(10.0, positions=17)
    groups, consumed = ws._split_actor_records(records, finalize=False)
    assert groups == []
    assert consumed == 0

    groups, consumed = ws._split_actor_records(records, finalize=True)
    assert len(groups) == 1
    assert groups[0]["timestamp"] == 10.0
    assert groups[0]["deltas"] == {
        SERIES_GAMES_COMPLETED: 1,
        SERIES_SIMS_RUN: 32,
        SERIES_POSITIONS_EVALUATED: 17,
    }
    assert consumed == len(records)


def test_split_actor_records_omits_positions_evaluated_when_absent():
    records = _actor_group(10.0)  # no position_counter wired -- omitted, never a fabricated 0
    groups, consumed = ws._split_actor_records(records, finalize=True)
    assert groups[0]["deltas"] == {SERIES_GAMES_COMPLETED: 1, SERIES_SIMS_RUN: 32}
    assert consumed == 2


def test_split_actor_records_a_second_group_finalizes_the_first():
    records = _actor_group(10.0, positions=17) + _actor_group(11.0, positions=9)
    groups, consumed = ws._split_actor_records(records, finalize=False)
    assert len(groups) == 1
    assert groups[0]["deltas"][SERIES_POSITIONS_EVALUATED] == 17
    assert consumed == 3


# ==============================================================================
# 4. Real wandb, offline mode
# ==============================================================================


@pytest.fixture
def wandb_offline(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("WANDB_SILENT", "true")
    yield


def test_init_wandb_run_carries_identity_config_tags_and_summary(tmp_path, wandb_offline):
    wandb = ws._require_wandb()
    root = _write_run(tmp_path, run_id="micro-run-1")
    run = ws._init_wandb_run(wandb, root, project="alpha-games-test")
    try:
        assert run.id == "micro-run-1"
        assert run.config["game"] == "blokus_duo"
        assert run.config["game_config"] == "MICRO_CONFIG"
        assert "game:blokus_duo" in run.tags
        assert "game_config:MICRO_CONFIG" in run.tags
        assert run.summary["run_id"] == "micro-run-1"
        assert isinstance(run.summary["orientation_hash"], str)
    finally:
        run.finish()


def test_sync_once_is_idempotent_on_a_repeated_call(tmp_path, wandb_offline):
    wandb = ws._require_wandb()
    root = _write_run(tmp_path, run_id="micro-run-2")
    writer = EpochMetricsWriter(root, "learner")
    for rec in _learner_group(1, 10.0) + _learner_group(2, 11.0):
        writer.append(rec)

    run = ws._init_wandb_run(wandb, root, project="alpha-games-test")
    try:
        state = ws.load_sync_state(root)
        changed_first = ws.sync_once(run, root, state, finalize=True)
        assert changed_first is True
        assert state.proc_cursors["learner"] == 8

        changed_second = ws.sync_once(run, root, state, finalize=True)
        assert changed_second is False
        assert state.proc_cursors["learner"] == 8
    finally:
        run.finish()


def test_sync_once_picks_up_only_newly_appended_records_on_a_later_call(tmp_path, wandb_offline):
    wandb = ws._require_wandb()
    root = _write_run(tmp_path, run_id="micro-run-3")
    writer = EpochMetricsWriter(root, "learner")
    for rec in _learner_group(1, 10.0):
        writer.append(rec)

    run = ws._init_wandb_run(wandb, root, project="alpha-games-test")
    try:
        state = ws.load_sync_state(root)
        ws.sync_once(run, root, state, finalize=True)
        assert state.proc_cursors["learner"] == 4

        for rec in _learner_group(2, 11.0):
            writer.append(rec)
        changed = ws.sync_once(run, root, state, finalize=True)
        assert changed is True
        assert state.proc_cursors["learner"] == 8
    finally:
        run.finish()


def test_sync_once_resync_from_a_freshly_loaded_state_stays_idempotent(tmp_path, wandb_offline):
    """Re-running the sync tool from scratch (state reloaded from disk) must not duplicate."""
    wandb = ws._require_wandb()
    root = _write_run(tmp_path, run_id="micro-run-4")
    writer = EpochMetricsWriter(root, "learner")
    for rec in _learner_group(1, 10.0):
        writer.append(rec)

    run = ws._init_wandb_run(wandb, root, project="alpha-games-test")
    try:
        state = ws.load_sync_state(root)
        ws.sync_once(run, root, state, finalize=True)
        ws.save_sync_state(root, state)
    finally:
        run.finish()

    run2 = ws._init_wandb_run(wandb, root, project="alpha-games-test")
    try:
        reloaded = ws.load_sync_state(root)
        changed = ws.sync_once(run2, root, reloaded, finalize=True)
        assert changed is False
    finally:
        run2.finish()


def test_sync_once_syncs_actor_deltas_as_a_running_cumulative(tmp_path, wandb_offline):
    wandb = ws._require_wandb()
    root = _write_run(tmp_path, run_id="micro-run-5", now=1_700_000_000.0)
    writer = EpochMetricsWriter(root, "actor-0")
    for rec in _actor_group(1_700_000_005.0, positions=17):
        writer.append(rec)
    for rec in _actor_group(1_700_000_010.0, positions=9):
        writer.append(rec)

    run = ws._init_wandb_run(wandb, root, project="alpha-games-test")
    try:
        state = ws.load_sync_state(root)
        changed = ws.sync_once(run, root, state, finalize=True)
        assert changed is True
        assert state.actor_totals[SERIES_GAMES_COMPLETED] == 2
        assert state.actor_totals[SERIES_POSITIONS_EVALUATED] == 26
        assert state.proc_cursors["actor-0"] == 6
    finally:
        run.finish()


def test_sync_once_logs_checkpoint_markers_as_summary_points(tmp_path, wandb_offline):
    wandb = ws._require_wandb()
    root = _write_run(tmp_path, run_id="micro-run-6", now=1_700_000_000.0)
    learner_writer = EpochMetricsWriter(root, "learner")
    actor_writer = EpochMetricsWriter(root, "actor-0")
    for rec in _actor_group(1_700_000_001.0, positions=5):
        actor_writer.append(rec)
    for rec in _learner_group(1, 1_700_000_002.0):
        learner_writer.append(rec)
    learner_writer.append(
        {
            "kind": CHECKPOINT_PUBLISHED_KIND,
            "model_version": 0,
            "learner_step": 1,
            "timestamp": 1_700_000_003.0,
        }
    )

    run = ws._init_wandb_run(wandb, root, project="alpha-games-test")
    try:
        state = ws.load_sync_state(root)
        ws.sync_once(run, root, state, finalize=True)
        assert state.checkpoint_versions_synced == [0]
        summary = run.summary["checkpoint_0"]
        assert summary["learner_step"] == 1
        assert summary["positions_evaluated"] == 5.0

        changed_again = ws.sync_once(run, root, state, finalize=True)
        assert changed_again is False
        assert state.checkpoint_versions_synced == [0]
    finally:
        run.finish()


# ==============================================================================
# 5. CLI
# ==============================================================================


def test_main_backfill_is_idempotent_across_two_invocations(tmp_path, wandb_offline):
    pytest.importorskip("wandb")
    root = _write_run(tmp_path, run_id="micro-run-cli")
    writer = EpochMetricsWriter(root, "learner")
    for rec in _learner_group(1, time.time()):
        writer.append(rec)

    assert ws.main([str(root), "--project", "alpha-games-test"]) == 0
    assert ws.main([str(root), "--project", "alpha-games-test"]) == 0
    state = ws.load_sync_state(root)
    assert state.proc_cursors["learner"] == 4
