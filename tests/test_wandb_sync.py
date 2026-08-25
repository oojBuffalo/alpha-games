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


def test_main_one_shot_without_finalize_holds_back_the_trailing_group(tmp_path, wandb_offline):
    """Default one-shot mode (no ``--finalize``) never consumes a trailing group (comment 3)."""
    pytest.importorskip("wandb")
    root = _write_run(tmp_path, run_id="micro-run-cli-no-finalize")
    writer = EpochMetricsWriter(root, "learner")
    for rec in _learner_group(1, time.time()):
        writer.append(rec)

    assert ws.main([str(root), "--project", "alpha-games-test"]) == 0
    assert ws.main([str(root), "--project", "alpha-games-test"]) == 0
    state = ws.load_sync_state(root)
    assert state.proc_cursors.get("learner", 0) == 0


def test_main_backfill_is_idempotent_across_two_invocations_with_finalize(tmp_path, wandb_offline):
    pytest.importorskip("wandb")
    root = _write_run(tmp_path, run_id="micro-run-cli")
    writer = EpochMetricsWriter(root, "learner")
    for rec in _learner_group(1, time.time()):
        writer.append(rec)

    assert ws.main([str(root), "--project", "alpha-games-test", "--finalize"]) == 0
    assert ws.main([str(root), "--project", "alpha-games-test", "--finalize"]) == 0
    state = ws.load_sync_state(root)
    assert state.proc_cursors["learner"] == 4


# ==============================================================================
# 6. Idempotency (explicit step), hold-back defaults, and the multi-actor
#    watermark -- all wandb-free: a minimal fake run double records exactly
#    what this script would send wandb, with no network and no real wandb
#    import, so these run in CI without the [wandb] extra installed.
# ==============================================================================


class _FakeRun:
    """A minimal ``wandb.Run`` double: records calls, makes no network calls.

    Exposes exactly the surface ``scripts/wandb_sync.py`` calls on a live
    run (``log``, ``summary``, ``define_metric``) so :func:`ws.sync_once`
    and :func:`ws._define_metrics` can be exercised without ``wandb``
    installed at all.
    """

    def __init__(self):
        self.logged: list[dict] = []
        self.summary: dict = {}
        self.metric_defines: dict[str, str | None] = {}

    def log(self, payload, *, step=None, commit=None):
        self.logged.append({"payload": dict(payload), "step": step, "commit": commit})

    def define_metric(self, name, *, step_metric=None):
        self.metric_defines[name] = step_metric


def test_write_ahead_plan_reviewer_scenario_new_record_becomes_eligible_before_retry(tmp_path):
    """Reviewer's exact scenario (comment: 'next_step is not stable across a crash if new
    records arrive before retry'). Pass 1 computes a plan logging learner L1 at step 0 and
    actor A1 at step 1; simulate a crash after those rows are logged but before the plan is
    applied/cleared (the sidecar still holds the pending plan, cursors unadvanced). Before
    retry, learner L2 becomes finalized. Retry must replay L1@0 and A1@1 verbatim (identical
    payload/step to the pre-crash attempt -- what a real server drops), then assign L2 a
    *fresh* step (2) rather than colliding with A1's step or dropping L2."""
    root = _write_run(tmp_path, run_id="run-plan-new-record", now=0.0)
    learner_writer = EpochMetricsWriter(root, "learner")
    actor_writer = EpochMetricsWriter(root, "actor-0")
    for rec in _learner_group(1, 10.0) + _learner_group(2, 11.0):  # L2 not yet finalized
        learner_writer.append(rec)
    for rec in _game(10.0) + _game(20.0):  # A1 finalized by the second game's boundary
        actor_writer.append(rec)

    state = ws.load_sync_state(root)
    plan = ws._compute_batch(root, state, finalize=False)
    assert len(plan["rows"]) == 2
    assert plan["rows"][0]["payload"]["learner/learner_step"] == 1
    assert plan["rows"][1]["payload"]["actor/proc"] == "actor-0"
    assert [row["step"] for row in plan["rows"]] == [0, 1]

    # Persist the plan (write-ahead), log every row -- then "crash": never apply/clear it.
    state.pending_plan = plan
    ws.save_sync_state(root, state)
    pre_crash_run = _FakeRun()
    for row in plan["rows"]:
        pre_crash_run.log(row["payload"], step=row["step"], commit=True)

    # Before retry, L2 becomes finalized (a third learner group proves L2 done).
    learner_writer.append(
        {
            "kind": "total",
            "series": "learner_step",
            "value": 3,
            "timestamp": 12.0,
        }
    )

    # "Process restart": reload state from disk -- the pending plan survived the crash.
    state2 = ws.load_sync_state(root)
    assert state2.pending_plan is not None
    fresh_run = _FakeRun()
    changed = ws.sync_once(fresh_run, root, state2, finalize=False)
    assert changed is True

    # L1 and A1 replay verbatim: same step, same payload as the pre-crash attempt.
    assert fresh_run.logged[0]["step"] == pre_crash_run.logged[0]["step"] == 0
    assert fresh_run.logged[0]["payload"] == pre_crash_run.logged[0]["payload"]
    assert fresh_run.logged[1]["step"] == pre_crash_run.logged[1]["step"] == 1
    assert fresh_run.logged[1]["payload"] == pre_crash_run.logged[1]["payload"]
    # L2 is picked up fresh -- a new step, never colliding with A1's already-used step 1.
    assert len(fresh_run.logged) == 3
    assert fresh_run.logged[2]["step"] == 2
    assert fresh_run.logged[2]["payload"]["learner/learner_step"] == 2
    assert state2.pending_plan is None
    assert state2.next_step == 3


def test_write_ahead_plan_crash_before_any_row_logged(tmp_path):
    """Comment A: a crash right after the plan is persisted, before a single row is logged --
    retry must log every planned row, verbatim, with nothing lost."""
    root = _write_run(tmp_path, run_id="run-plan-crash-before-log", now=0.0)
    writer = EpochMetricsWriter(root, "learner")
    for rec in _learner_group(1, 10.0):
        writer.append(rec)

    state = ws.load_sync_state(root)
    plan = ws._compute_batch(root, state, finalize=True)
    assert len(plan["rows"]) == 1

    state.pending_plan = plan
    ws.save_sync_state(root, state)
    # "Crash" -- no run.log call ever happens for this attempt.

    state2 = ws.load_sync_state(root)
    assert state2.pending_plan is not None
    fresh_run = _FakeRun()
    changed = ws.sync_once(fresh_run, root, state2, finalize=True)
    assert changed is True
    assert len(fresh_run.logged) == 1
    assert fresh_run.logged[0]["step"] == plan["rows"][0]["step"] == 0
    assert fresh_run.logged[0]["payload"] == plan["rows"][0]["payload"]
    assert fresh_run.logged[0]["commit"] is True
    assert state2.pending_plan is None


def test_write_ahead_plan_crash_mid_batch_prefix_accepted(tmp_path):
    """Comment A: a crash after only a prefix of the plan's rows were logged -- retry
    replays the *entire* plan (including the already-accepted prefix) verbatim; a real
    server drops that prefix via the step mechanism, so nothing is duplicated, and the
    rest of the batch (never accepted) is logged for the first time."""
    root = _write_run(tmp_path, run_id="run-plan-crash-mid-batch", now=0.0)
    learner_writer = EpochMetricsWriter(root, "learner")
    actor_writer = EpochMetricsWriter(root, "actor-0")
    for rec in _learner_group(1, 10.0) + _learner_group(2, 11.0):
        learner_writer.append(rec)
    for rec in _game(10.0) + _game(20.0):
        actor_writer.append(rec)

    state = ws.load_sync_state(root)
    plan = ws._compute_batch(root, state, finalize=False)
    assert len(plan["rows"]) == 2

    state.pending_plan = plan
    ws.save_sync_state(root, state)
    pre_crash_run = _FakeRun()
    pre_crash_run.log(plan["rows"][0]["payload"], step=plan["rows"][0]["step"], commit=True)
    # "Crash" -- the second row (A1) was never sent.

    state2 = ws.load_sync_state(root)
    fresh_run = _FakeRun()
    changed = ws.sync_once(fresh_run, root, state2, finalize=False)
    assert changed is True
    assert len(fresh_run.logged) == 2
    for original, replayed in zip(plan["rows"], fresh_run.logged, strict=True):
        assert replayed["step"] == original["step"]
        assert replayed["payload"] == original["payload"]
    # The already-accepted prefix matches exactly what the crashed attempt had logged.
    assert fresh_run.logged[0]["step"] == pre_crash_run.logged[0]["step"]
    assert fresh_run.logged[0]["payload"] == pre_crash_run.logged[0]["payload"]
    assert state2.pending_plan is None


def test_one_shot_default_holds_back_an_incomplete_flush_then_completes_on_the_next_sync(
    tmp_path,
):
    """Comment 3: reproduce the reviewer's scenario -- a snapshot lands after an actor has
    appended games_completed/sims_run but before positions_evaluated. The partial group must
    be held back (not logged, cursor not consumed); once the rest lands (proven done by the
    next game's boundary record), a second sync -- still without --finalize -- emits exactly
    one complete group and leaves the new trailing game held back in turn."""
    root = _write_run(tmp_path, run_id="run-partial-flush", now=0.0)
    writer = EpochMetricsWriter(root, "actor-0")
    writer.append(delta_record(SERIES_GAMES_COMPLETED, 1, timestamp=10.0))
    writer.append(delta_record(SERIES_SIMS_RUN, 32, timestamp=10.0))
    # positions_evaluated has not landed yet: this actor is still mid-flush.

    fake_run = _FakeRun()
    state = ws.load_sync_state(root)
    changed = ws.sync_once(fake_run, root, state, finalize=False)
    assert changed is False
    assert fake_run.logged == []
    assert state.proc_cursors.get("actor-0", 0) == 0

    # The rest of game 1's flush lands, then game 2 starts -- proving game 1 is done.
    writer.append(delta_record(SERIES_POSITIONS_EVALUATED, 17, timestamp=10.0))
    writer.append(delta_record(SERIES_GAMES_COMPLETED, 1, timestamp=20.0))
    writer.append(delta_record(SERIES_SIMS_RUN, 40, timestamp=20.0))

    changed = ws.sync_once(fake_run, root, state, finalize=False)
    assert changed is True
    assert len(fake_run.logged) == 1
    payload = fake_run.logged[0]["payload"]
    assert payload["actor/games_completed"] == 1
    assert payload["actor/sims_run"] == 32
    assert payload["actor/positions_evaluated"] == 17
    # Game 2 (still trailing, unfinalized) stays held back, not consumed.
    assert state.proc_cursors["actor-0"] == 3
    assert state.actor_buffer.get("actor-0", []) == []


def _game(ts, *, games=1, sims=10):
    return [
        delta_record(SERIES_GAMES_COMPLETED, games, timestamp=ts),
        delta_record(SERIES_SIMS_RUN, sims, timestamp=ts),
    ]


def test_multiactor_watermark_buffers_and_releases_in_global_sorted_order(tmp_path):
    """Comment 4: reproduce the reviewer's exact scenario -- actor 0 groups at t=10,30,40
    and actor 1 at t=20 -- across two polls (plus a --finalize pass for the group that
    would otherwise stay buffered indefinitely). Asserts release order is globally sorted,
    cumulative totals are correct at release time, and a slower actor holds back a faster
    one's later groups until the watermark (or --finalize) releases them."""
    root = _write_run(tmp_path, run_id="run-watermark", now=0.0)
    actor0 = EpochMetricsWriter(root, "actor-0")
    actor1 = EpochMetricsWriter(root, "actor-1")

    # actor 0: games at t=10,30,40 (t=50 only to finalize t=40 via boundary evidence).
    for rec in _game(10.0) + _game(30.0) + _game(40.0) + _game(50.0):
        actor0.append(rec)
    # actor 1: game at t=20 (t=25 only to finalize t=20).
    for rec in _game(20.0) + _game(25.0):
        actor1.append(rec)

    fake_run = _FakeRun()
    state = ws.load_sync_state(root)
    changed = ws.sync_once(fake_run, root, state, finalize=False)
    assert changed is True

    poll1 = [
        (row["payload"]["actor/wall_clock_s"], row["payload"]["actor/proc"])
        for row in fake_run.logged
    ]
    assert poll1 == [(10.0, "actor-0"), (20.0, "actor-1")]
    assert fake_run.logged[0]["payload"]["actor/games_completed"] == 1
    assert fake_run.logged[0]["payload"]["actor/sims_run"] == 10
    assert fake_run.logged[1]["payload"]["actor/games_completed"] == 2
    assert fake_run.logged[1]["payload"]["actor/sims_run"] == 20
    # t=30 and t=40 stay buffered -- actor 1's watermark (20) hasn't reached them.
    assert [g["timestamp"] for g in state.actor_buffer["actor-0"]] == [30.0, 40.0]

    # actor 1 advances: t=35 (t=45 only to finalize t=35).
    for rec in _game(35.0) + _game(45.0):
        actor1.append(rec)

    changed = ws.sync_once(fake_run, root, state, finalize=False)
    assert changed is True
    # actor 1's own file also still holds t=25 (written only to finalize t=20's group in
    # poll 1) -- it becomes releasable now too, alongside t=30 and the newly-finalized t=35.
    poll2_rows = fake_run.logged[2:]
    poll2 = [
        (row["payload"]["actor/wall_clock_s"], row["payload"]["actor/proc"]) for row in poll2_rows
    ]
    assert poll2 == [(25.0, "actor-1"), (30.0, "actor-0"), (35.0, "actor-1")]
    assert [row["payload"]["actor/games_completed"] for row in poll2_rows] == [3, 4, 5]
    assert [row["payload"]["actor/sims_run"] for row in poll2_rows] == [30, 40, 50]
    # Every release across both polls, taken together, is globally sorted by wall clock.
    all_released = poll1 + poll2
    assert [ts for ts, _ in all_released] == sorted(ts for ts, _ in all_released)
    # t=40 (actor 0) is still buffered -- actor 1's watermark (35) hasn't reached it yet.
    assert [g["timestamp"] for g in state.actor_buffer["actor-0"]] == [40.0]
    assert state.actor_buffer.get("actor-1", []) == []

    # --finalize flushes every remaining buffered/trailing group regardless of watermark
    # (actor 0's t=50 and actor 1's t=45 -- both still entirely unsplit until now -- and
    # the previously-held-back t=40 all release together).
    changed = ws.sync_once(fake_run, root, state, finalize=True)
    assert changed is True
    poll3_rows = fake_run.logged[5:]
    poll3 = [
        (row["payload"]["actor/wall_clock_s"], row["payload"]["actor/proc"]) for row in poll3_rows
    ]
    assert poll3 == [(40.0, "actor-0"), (45.0, "actor-1"), (50.0, "actor-0")]
    assert state.actor_buffer == {}

    # The full release sequence across every poll is globally sorted by wall clock.
    every_release = [
        (row["payload"]["actor/wall_clock_s"], row["payload"]["actor/proc"])
        for row in fake_run.logged
    ]
    assert [ts for ts, _ in every_release] == sorted(ts for ts, _ in every_release)

    # throughput/* parallel-axis fields carry the same values as actor/* on every release.
    for row in fake_run.logged:
        payload = row["payload"]
        assert payload["throughput/games_vs_positions"] == payload["actor/games_completed"]
        assert payload["throughput/sims_vs_positions"] == payload["actor/sims_run"]

    # Steps are strictly increasing across the whole run, in release order.
    steps = [row["step"] for row in fake_run.logged]
    assert steps == list(range(len(steps)))


def test_multiactor_watermark_stalls_on_a_process_with_no_groups_yet(tmp_path):
    """A discovered actor process (its metrics file exists) that has not yet completed a
    single flush contributes no watermark and blocks release entirely -- documented stall
    behavior, not a bug: this script cannot tell "about to report" from "already dead"
    without --finalize."""
    root = _write_run(tmp_path, run_id="run-watermark-stall", now=0.0)
    actor0 = EpochMetricsWriter(root, "actor-0")
    for rec in _game(10.0) + _game(20.0):  # t=20 finalizes t=10
        actor0.append(rec)
    EpochMetricsWriter(root, "actor-1")  # exists, has written nothing yet

    fake_run = _FakeRun()
    state = ws.load_sync_state(root)
    changed = ws.sync_once(fake_run, root, state, finalize=False)
    assert changed is False
    assert fake_run.logged == []
    assert [g["timestamp"] for g in state.actor_buffer["actor-0"]] == [10.0]

    # --finalize still flushes it -- along with t=20, which by now has also been split
    # into its own finalized (trailing-under-finalize) group.
    changed = ws.sync_once(fake_run, root, state, finalize=True)
    assert changed is True
    assert [row["payload"]["actor/wall_clock_s"] for row in fake_run.logged] == [10.0, 20.0]


def test_checkpoint_payload_carries_the_positions_and_gpu_hours_axis_fields(tmp_path):
    root = _write_run(tmp_path, run_id="run-checkpoint-axes", now=0.0)
    learner_writer = EpochMetricsWriter(root, "learner")
    actor_writer = EpochMetricsWriter(root, "actor-0")
    for rec in _game(1.0, sims=5):
        actor_writer.append(rec)
    actor_writer.append(delta_record(SERIES_POSITIONS_EVALUATED, 5, timestamp=1.0))
    for rec in _learner_group(1, 2.0):
        learner_writer.append(rec)
    learner_writer.append(
        {
            "kind": CHECKPOINT_PUBLISHED_KIND,
            "model_version": 0,
            "learner_step": 1,
            "timestamp": 3.0,
        }
    )

    fake_run = _FakeRun()
    state = ws.load_sync_state(root)
    ws.sync_once(fake_run, root, state, finalize=True)

    checkpoint_rows = [
        row for row in fake_run.logged if "checkpoint/model_version" in row["payload"]
    ]
    assert len(checkpoint_rows) == 1
    payload = checkpoint_rows[0]["payload"]
    assert (
        payload["checkpoint/positions_evaluated_axis"] == payload["checkpoint/positions_evaluated"]
    )
    assert payload["checkpoint/gpu_hours_axis"] == payload["checkpoint/gpu_hours"]
    assert payload["checkpoint/marker_vs_positions"] == payload["checkpoint/model_version"]
    assert payload["checkpoint/marker_vs_gpu_hours"] == payload["checkpoint/model_version"]


def test_clock_rollback_clamps_wall_clock_axis_and_never_regresses(tmp_path):
    """Comment B: reviewer's exact scenario -- one actor appends timestamps 10,20,30,15,16
    (a clock rollback after t=30), plus a successor game to finalize the t=16 group under
    the default finalize=False. The released actor/wall_clock_s values must never regress:
    15 and 16 release pinned at the pre-rollback high-water mark (30), not at their own raw
    (smaller) timestamps, and cumulative totals still advance by exactly one game per row."""
    root = _write_run(tmp_path, run_id="run-rollback-single", now=0.0)
    writer = EpochMetricsWriter(root, "actor-0")
    for rec in _game(10.0) + _game(20.0) + _game(30.0) + _game(15.0) + _game(16.0) + _game(40.0):
        writer.append(rec)

    fake_run = _FakeRun()
    state = ws.load_sync_state(root)
    changed = ws.sync_once(fake_run, root, state, finalize=False)
    assert changed is True

    wall_clock = [row["payload"]["actor/wall_clock_s"] for row in fake_run.logged]
    assert wall_clock == [10.0, 20.0, 30.0, 30.0, 30.0]
    assert wall_clock == sorted(wall_clock)  # never regresses, even across the rollback
    games = [row["payload"]["actor/games_completed"] for row in fake_run.logged]
    assert games == [1, 2, 3, 4, 5]
    sims = [row["payload"]["actor/sims_run"] for row in fake_run.logged]
    assert sims == [10, 20, 30, 40, 50]
    # The t=40 successor (needed only to finalize the t=16 group) stays held back as the
    # new trailing group -- nothing beyond the 5 real games is released yet.
    assert len(fake_run.logged) == 5
    assert state.actor_watermarks["actor-0"] == 30.0


def test_clock_rollback_with_second_actor_global_sort_and_cumulative_totals(tmp_path):
    """Comment B: the same clock-rollback actor alongside a second, normally-advancing
    actor. The global watermark and release sort must both use effective_ts (never the raw
    timestamp): actor 1's own group at t=35 is held back because actor 0's watermark (capped
    at 30 by the rollback) hasn't reached it, even though actor 1's *own* stream is fine --
    demonstrating the watermark and the clamp composing correctly. --finalize then flushes
    everything still buffered, in global sorted order, with correct running cumulatives."""
    root = _write_run(tmp_path, run_id="run-rollback-two-actors", now=0.0)
    actor0 = EpochMetricsWriter(root, "actor-0")
    actor1 = EpochMetricsWriter(root, "actor-1")
    for rec in _game(10.0) + _game(20.0) + _game(30.0) + _game(15.0) + _game(16.0) + _game(40.0):
        actor0.append(rec)
    for rec in _game(25.0) + _game(35.0) + _game(45.0):
        actor1.append(rec)

    fake_run = _FakeRun()
    state = ws.load_sync_state(root)
    changed = ws.sync_once(fake_run, root, state, finalize=False)
    assert changed is True

    released = [
        (row["payload"]["actor/proc"], row["payload"]["actor/wall_clock_s"])
        for row in fake_run.logged
    ]
    assert released == [
        ("actor-0", 10.0),
        ("actor-0", 20.0),
        ("actor-1", 25.0),
        ("actor-0", 30.0),
        ("actor-0", 30.0),
        ("actor-0", 30.0),
    ]
    wall_clock = [ts for _, ts in released]
    assert wall_clock == sorted(wall_clock)
    games = [row["payload"]["actor/games_completed"] for row in fake_run.logged]
    assert games == [1, 2, 3, 4, 5, 6]
    sims = [row["payload"]["actor/sims_run"] for row in fake_run.logged]
    assert sims == [10, 20, 30, 40, 50, 60]
    # actor 1's own t=35 group is held back: the GLOBAL watermark is capped at actor 0's
    # rollback-pinned high-water mark (30), not actor 1's own (35) -- nothing is dropped.
    assert [g["effective_ts"] for g in state.actor_buffer["actor-1"]] == [35.0]

    changed2 = ws.sync_once(fake_run, root, state, finalize=True)
    assert changed2 is True
    tail = [
        (row["payload"]["actor/proc"], row["payload"]["actor/wall_clock_s"])
        for row in fake_run.logged[6:]
    ]
    assert tail == [("actor-1", 35.0), ("actor-0", 40.0), ("actor-1", 45.0)]
    assert [row["payload"]["actor/games_completed"] for row in fake_run.logged[6:]] == [7, 8, 9]
    assert state.actor_buffer == {}
    # The complete release sequence, across both passes, never regresses.
    every_wall_clock = [row["payload"]["actor/wall_clock_s"] for row in fake_run.logged]
    assert every_wall_clock == sorted(every_wall_clock)


def test_clock_rollback_effective_ts_deterministic_across_crash_replay(tmp_path):
    """Comment B: effective_ts assignment is a pure function of file-append order plus the
    persisted high-water mark, so it is identical across a crash and retry -- confirmed two
    ways: (1) computing a plan twice from the same unchanged confirmed state yields byte-for-
    byte identical rows; (2) running the rollback scenario through a simulated crash (comment
    A's write-ahead plan) reproduces the exact same clamped actor/wall_clock_s values."""
    root = _write_run(tmp_path, run_id="run-rollback-determinism", now=0.0)
    writer = EpochMetricsWriter(root, "actor-0")
    for rec in _game(10.0) + _game(20.0) + _game(30.0) + _game(15.0) + _game(16.0) + _game(40.0):
        writer.append(rec)

    state = ws.load_sync_state(root)
    plan_a = ws._compute_batch(root, state, finalize=False)
    plan_b = ws._compute_batch(root, state, finalize=False)
    assert plan_a == plan_b

    # Simulate a crash mid-replay (comment A's mechanism) and confirm the clamped values
    # the crashed attempt already sent match exactly what the retry re-sends.
    state.pending_plan = plan_a
    ws.save_sync_state(root, state)
    pre_crash_run = _FakeRun()
    for row in plan_a["rows"][:3]:
        pre_crash_run.log(row["payload"], step=row["step"], commit=True)

    state2 = ws.load_sync_state(root)
    fresh_run = _FakeRun()
    changed = ws.sync_once(fresh_run, root, state2, finalize=False)
    assert changed is True
    assert len(fresh_run.logged) == len(plan_a["rows"])
    for original, replayed in zip(plan_a["rows"], fresh_run.logged, strict=True):
        assert replayed["step"] == original["step"]
        assert replayed["payload"] == original["payload"]
    # Only the pre-crash prefix (3 of 5 rows) was actually sent before the simulated crash;
    # compare just that prefix against the retry's replay of the same rows.
    for original, replayed in zip(pre_crash_run.logged, fresh_run.logged[:3], strict=True):
        assert (
            replayed["payload"]["actor/wall_clock_s"] == original["payload"]["actor/wall_clock_s"]
        )


# ==============================================================================
# 7. Custom axes (wandb-free): the full define_metric set issue #90 requires.
# ==============================================================================


def test_define_metrics_matches_exactly_the_axes_issue_90_requires():
    """Issue #90: learner_step for the learner; wall-clock and positions for actor
    throughput; learner-step, positions, and GPU-hours for checkpoint markers. No
    extra axes beyond those."""
    fake_run = _FakeRun()
    ws._define_metrics(fake_run)

    assert fake_run.metric_defines == {
        "learner/learner_step": None,
        "learner/*": "learner/learner_step",
        "actor/wall_clock_s": None,
        "actor/*": "actor/wall_clock_s",
        "throughput/positions_evaluated": None,
        "throughput/games_vs_positions": "throughput/positions_evaluated",
        "throughput/sims_vs_positions": "throughput/positions_evaluated",
        "checkpoint/learner_step": None,
        "checkpoint/*": "checkpoint/learner_step",
        "checkpoint/positions_evaluated_axis": None,
        "checkpoint/marker_vs_positions": "checkpoint/positions_evaluated_axis",
        "checkpoint/gpu_hours_axis": None,
        "checkpoint/marker_vs_gpu_hours": "checkpoint/gpu_hours_axis",
    }

    # Grouped by subject: exactly the axis set issue #90 names, no more, no fewer.
    learner_axes = {v for k, v in fake_run.metric_defines.items() if k.startswith("learner/") and v}
    actor_axes = {
        v
        for k, v in fake_run.metric_defines.items()
        if (k.startswith("actor/") or k.startswith("throughput/")) and v
    }
    checkpoint_axes = {
        v for k, v in fake_run.metric_defines.items() if k.startswith("checkpoint/") and v
    }
    assert learner_axes == {"learner/learner_step"}
    assert actor_axes == {"actor/wall_clock_s", "throughput/positions_evaluated"}
    assert checkpoint_axes == {
        "checkpoint/learner_step",
        "checkpoint/positions_evaluated_axis",
        "checkpoint/gpu_hours_axis",
    }
