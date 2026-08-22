"""The M3 observability contract: ``core/observability.py`` (§12 M3, issue #62).

Four layers, cheapest first:

1. **Primitives in isolation** -- record builders, :class:`PositionCounter`,
   :func:`~core.observability.count_positions`, and
   ``core.metrics.list_procs`` -- pure functions/small objects, no files.
2. **The reducer, scripted** -- hand-built per-process epoch files (never
   through a real driver) covering the kind taxonomy, the GPU-hour segment
   algebra, rate edge cases, the deterministic tie-break, and the full
   two-actor + learner + orchestrator golden with publishes interleaved
   between flushes, an epoch restart, and idempotent re-reduction.
3. **Driver wiring** -- ``core.actor.ActorDriver``'s opt-in
   ``metrics_writer``/``position_counter`` flush at real game boundaries.
4. **Live-path smoke** -- a small, fast, single-process real run (mirrors
   ``tests/test_ipc.py``'s layer-4 pattern: real drivers, no
   multiprocessing) whose emitted epoch files reduce to sane numbers.
"""

from __future__ import annotations

import dataclasses
import math
import time
from types import SimpleNamespace

import pytest

from core.actor import VALIDATE_TIER_SIMS, ActorDriver
from core.checkpoint import list_published_versions
from core.ipc import build_actor_pacing, build_actor_refresh
from core.learner import LearnerDriver
from core.metrics import EpochMetricsWriter, iter_epoch_records, list_procs
from core.network import NetworkConfig
from core.observability import (
    CHECKPOINT_PUBLISHED_KIND,
    KIND_DELTA,
    KIND_GAUGE,
    KIND_SEGMENT_END,
    KIND_SEGMENT_START,
    KIND_TOTAL,
    PositionCounter,
    ReducedRun,
    _ordered_records,  # white-box: the pinned tie-break rule
    count_positions,
    delta_record,
    gauge_record,
    is_real_cuda,
    reduce_run,
    segment_end_record,
    segment_start_record,
    total_record,
)
from core.replay_shard import read_shard
from core.runconfig import TrainingConfig, load_run_config
from games.tictactoe import TicTacToe

TTT = TicTacToe()

# Mirrors tests/test_actor.py's tiny net -- CPU-cheap, real inference.
TTT_NET_CONFIG = NetworkConfig(2, (3, 3), (9,), trunk_blocks=1, trunk_channels=4)


# ==============================================================================
# 1. Primitives in isolation
# ==============================================================================


def test_list_procs_is_empty_for_a_fresh_dir_and_discovers_every_writer(tmp_path):
    assert list_procs(tmp_path) == frozenset()
    EpochMetricsWriter(tmp_path, "actor-0")
    EpochMetricsWriter(tmp_path, "actor-1")
    EpochMetricsWriter(tmp_path, "learner")
    EpochMetricsWriter(tmp_path, "orchestrator")
    assert list_procs(tmp_path) == frozenset({"actor-0", "actor-1", "learner", "orchestrator"})


def test_record_builders_carry_the_pinned_kind_and_fields():
    d = delta_record("games_completed", 3, timestamp=1.0)
    assert d == {"kind": KIND_DELTA, "series": "games_completed", "value": 3, "timestamp": 1.0}
    g = gauge_record("loss_total", 0.5, timestamp=2.0)
    assert g == {"kind": KIND_GAUGE, "series": "loss_total", "value": 0.5, "timestamp": 2.0}
    total = total_record("learner_step", 4, timestamp=3.0)
    assert total == {"kind": KIND_TOTAL, "series": "learner_step", "value": 4, "timestamp": 3.0}


def test_record_builders_default_timestamp_to_now():
    before = time.time()
    rec = delta_record("games_completed", 1)
    after = time.time()
    assert before <= rec["timestamp"] <= after


def test_segment_records_carry_device_and_is_cuda():
    start = segment_start_record(device="cpu", timestamp=1.0)
    assert start == {
        "kind": KIND_SEGMENT_START,
        "device": "cpu",
        "is_cuda": False,
        "timestamp": 1.0,
    }
    end = segment_end_record(timestamp=2.0)
    assert end == {"kind": KIND_SEGMENT_END, "timestamp": 2.0}
    assert is_real_cuda("cpu") is False  # never real CUDA regardless of hardware


def test_records_round_trip_through_an_epoch_writer(tmp_path):
    writer = EpochMetricsWriter(tmp_path, "actor-0")
    writer.append(delta_record("games_completed", 1, timestamp=1.0))
    writer.append(gauge_record("loss_total", 0.25, timestamp=2.0))
    records = list(iter_epoch_records(tmp_path, "actor-0"))
    assert records == [
        {"kind": "delta", "series": "games_completed", "value": 1, "timestamp": 1.0},
        {"kind": "gauge", "series": "loss_total", "value": 0.25, "timestamp": 2.0},
    ]


def test_position_counter_add_and_drain(tmp_path):
    counter = PositionCounter()
    assert counter.total == 0
    counter.add(3)
    counter.add(2)
    assert counter.total == 5
    assert counter.drain() == 5
    assert counter.total == 0  # drained, not merely read
    assert counter.drain() == 0  # draining an already-empty counter is a no-op


def test_position_counter_rejects_negative_additions():
    counter = PositionCounter()
    with pytest.raises(ValueError, match="position count"):
        counter.add(-1)


def test_count_positions_batch_one_counts_one_per_call_and_preserves_the_result():
    counter = PositionCounter()
    calls = []

    def evaluate(game, state):
        calls.append((game, state))
        return 0.25, {1: 0.5}

    wrapped = count_positions(evaluate, counter)
    result = wrapped("g", "s")
    assert result == (0.25, {1: 0.5})  # wrapping never changes the return value
    assert calls == [("g", "s")]  # nor the arguments forwarded to the wrapped call
    for _ in range(4):
        wrapped("g", "s")
    assert counter.total == 5  # 1 per call, batch-1 bridge


def test_count_positions_batched_stub_counts_forwards_times_batch_size():
    """The M5-proof property: a batched call counts its own batch cardinality."""
    counter = PositionCounter()

    def stub_batched_evaluate(states):
        return [0.0] * len(states)

    wrapped = count_positions(stub_batched_evaluate, counter, batch_size=lambda states: len(states))
    forwards = 7
    batch = [object(), object(), object()]
    for _ in range(forwards):
        wrapped(batch)
    assert counter.total == forwards * len(batch)


# ==============================================================================
# 2. The reducer, scripted
# ==============================================================================


def test_kind_semantics_delta_sums_gauge_latest_wins_total_never_summed(tmp_path):
    w0 = EpochMetricsWriter(tmp_path, "actor-0")
    w0.append(delta_record("games_completed", 2, timestamp=1.0))
    w0.append(delta_record("games_completed", 3, timestamp=2.0))
    w1 = EpochMetricsWriter(tmp_path, "actor-0")  # a restart -> a new epoch
    assert w1.epoch == 1
    w1.append(delta_record("games_completed", 4, timestamp=3.0))

    learner = EpochMetricsWriter(tmp_path, "learner")
    learner.append(gauge_record("loss_total", 1.0, timestamp=1.0))
    learner.append(gauge_record("loss_total", 0.5, timestamp=2.0))
    learner.append(total_record("learner_step", 10, timestamp=1.0))
    learner.append(total_record("learner_step", 20, timestamp=2.0))  # NOT 10 + 20

    result = reduce_run(tmp_path)
    assert result.totals["games_completed"] == 9  # 2 + 3 + 4, summed across epochs
    assert result.gauges["loss_total"] == 0.5  # latest in time order, never averaged/summed
    assert result.totals["learner_step"] == 20  # the coordinator's exact last value


def test_totals_default_to_zero_and_everything_else_is_empty_for_an_untouched_run(tmp_path):
    result = reduce_run(tmp_path)
    assert result == ReducedRun(
        totals={
            "games_completed": 0.0,
            "positions_evaluated": 0.0,
            "sims_run": 0.0,
            "learner_step": 0.0,
        },
        rates={
            "games_per_hour": pytest.approx(float("nan"), nan_ok=True),
            "sims_per_sec": pytest.approx(float("nan"), nan_ok=True),
            "learner_steps_per_sec": pytest.approx(float("nan"), nan_ok=True),
        },
        gauges={},
        gpu_hours=0.0,
        checkpoints={},
    )


def test_rates_are_nan_for_a_single_flush_never_a_fabricated_zero(tmp_path):
    w = EpochMetricsWriter(tmp_path, "actor-0")
    w.append(delta_record("games_completed", 1, timestamp=1.0))
    result = reduce_run(tmp_path)
    assert math.isnan(result.rates["games_per_hour"])
    assert result.totals["games_completed"] == 1.0  # the total itself is still well-defined


def test_gpu_hours_are_single_counted_regardless_of_concurrent_actor_count(tmp_path):
    """The Test Strategy's GPU-time golden: overlapping actor activity
    never multiplies gpu_hours.
    """
    orch = EpochMetricsWriter(tmp_path, "orchestrator")
    orch.append(segment_start_record(device="cpu", timestamp=0.0))
    orch.append(segment_end_record(timestamp=3600.0))  # exactly one hour, one device

    for i in range(3):  # three actors, all "active" throughout the segment
        w = EpochMetricsWriter(tmp_path, f"actor-{i}")
        w.append(delta_record("games_completed", 1, timestamp=100.0 + i))
        w.append(delta_record("games_completed", 1, timestamp=3500.0 + i))

    result = reduce_run(tmp_path)
    assert result.gpu_hours == 1.0  # never 3.0
    assert result.totals["games_completed"] == 6


def test_an_unterminated_gpu_segment_contributes_zero(tmp_path):
    orch = EpochMetricsWriter(tmp_path, "orchestrator")
    orch.append(segment_start_record(device="cpu", timestamp=0.0))
    result = reduce_run(tmp_path)
    assert result.gpu_hours == 0.0  # conservative: never estimated/interpolated


def test_an_unmatched_segment_end_is_ignored_not_raised(tmp_path):
    orch = EpochMetricsWriter(tmp_path, "orchestrator")
    orch.append(segment_end_record(timestamp=5.0))  # no prior start
    result = reduce_run(tmp_path)  # must not raise
    assert result.gpu_hours == 0.0


def test_tie_break_orders_equal_timestamps_by_process_name_then_append_order(tmp_path):
    a1 = EpochMetricsWriter(tmp_path, "actor-1")
    a0 = EpochMetricsWriter(tmp_path, "actor-0")
    a1.append(delta_record("games_completed", 1, timestamp=5.0))
    a0.append(delta_record("games_completed", 1, timestamp=5.0))
    a0.append(delta_record("games_completed", 1, timestamp=5.0))

    ordered = _ordered_records(tmp_path)
    assert [(proc, seq) for _, proc, seq, _ in ordered] == [
        ("actor-0", 0),
        ("actor-0", 1),
        ("actor-1", 0),
    ]


def test_reduce_run_is_idempotent(tmp_path):
    w = EpochMetricsWriter(tmp_path, "actor-0")
    w.append(delta_record("games_completed", 1, timestamp=1.0))
    w.append(delta_record("positions_evaluated", 7, timestamp=1.0))
    first = reduce_run(tmp_path)
    second = reduce_run(tmp_path)
    # NaN != NaN under plain equality (both rates are legitimately undefined
    # here -- a single flush, module docstring's edge case), so the rates
    # dict is compared NaN-aware field by field; everything else is a plain
    # value and compares directly.
    assert first.totals == second.totals
    assert first.gauges == second.gauges
    assert first.gpu_hours == second.gpu_hours
    assert first.checkpoints == second.checkpoints
    assert first.rates.keys() == second.rates.keys()
    for key in first.rates:
        a, b = first.rates[key], second.rates[key]
        assert (math.isnan(a) and math.isnan(b)) or a == b


def _append_checkpoint_published(writer, *, version, learner_step, timestamp):
    writer.append(
        {
            "kind": CHECKPOINT_PUBLISHED_KIND,
            "model_version": version,
            "learner_step": learner_step,
            "timestamp": timestamp,
        }
    )


def test_scripted_two_actor_learner_orchestrator_golden_with_interleaved_publishes(tmp_path):
    """The amendment's hand-built golden: publishes interleaved between flushes,
    an epoch restart between two of them, and hand-computed totals/rates/join
    coordinates -- see the accompanying inline arithmetic.
    """
    orch = EpochMetricsWriter(tmp_path, "orchestrator")
    a0 = EpochMetricsWriter(tmp_path, "actor-0")
    a1 = EpochMetricsWriter(tmp_path, "actor-1")
    learner = EpochMetricsWriter(tmp_path, "learner")

    orch.append(segment_start_record(device="cpu", timestamp=0.0))
    a0.append(delta_record("games_completed", 1, timestamp=1.0))
    a0.append(delta_record("positions_evaluated", 10, timestamp=1.0))
    a0.append(delta_record("sims_run", 128, timestamp=1.0))
    a1.append(delta_record("games_completed", 1, timestamp=2.0))
    a1.append(delta_record("positions_evaluated", 20, timestamp=2.0))
    a1.append(delta_record("sims_run", 256, timestamp=2.0))
    _append_checkpoint_published(learner, version=0, learner_step=0, timestamp=3.0)
    orch.append(segment_end_record(timestamp=3.5))  # segment #1 closes: 3.5s
    learner.append(total_record("learner_step", 1, timestamp=4.0))
    learner.append(gauge_record("loss_total", 0.9, timestamp=4.0))
    learner.append(gauge_record("replay_ratio", 0.1, timestamp=4.0))
    a0.append(delta_record("games_completed", 1, timestamp=5.0))
    a0.append(delta_record("positions_evaluated", 15, timestamp=5.0))
    a0.append(delta_record("sims_run", 128, timestamp=5.0))
    learner.append(total_record("learner_step", 2, timestamp=6.0))
    learner.append(gauge_record("loss_total", 0.8, timestamp=6.0))
    orch2 = EpochMetricsWriter(tmp_path, "orchestrator")  # an orchestrator restart
    assert orch2.epoch == 1
    orch2.append(segment_start_record(device="cpu", timestamp=6.5))  # segment #2: never closes
    _append_checkpoint_published(learner, version=1, learner_step=2, timestamp=7.0)
    a1.append(delta_record("games_completed", 1, timestamp=8.0))
    a1.append(delta_record("positions_evaluated", 25, timestamp=8.0))
    a1.append(delta_record("sims_run", 256, timestamp=8.0))
    a0_restarted = EpochMetricsWriter(tmp_path, "actor-0")  # actor-0 crashes and restarts
    assert a0_restarted.epoch == 1
    a0_restarted.append(delta_record("games_completed", 1, timestamp=9.0))
    a0_restarted.append(delta_record("positions_evaluated", 5, timestamp=9.0))
    a0_restarted.append(delta_record("sims_run", 128, timestamp=9.0))
    learner.append(total_record("learner_step", 3, timestamp=10.0))
    learner.append(gauge_record("loss_total", 0.7, timestamp=10.0))
    _append_checkpoint_published(learner, version=2, learner_step=3, timestamp=11.0)

    result = reduce_run(tmp_path)

    # totals: games=2(a0 ep0)+1(a0 ep1)+2(a1)=5; positions=25+5+45=75; sims=256+128+512=896
    assert result.totals["games_completed"] == 5
    assert result.totals["positions_evaluated"] == 75
    assert result.totals["sims_run"] == 896
    assert result.totals["learner_step"] == 3  # exact last total, never summed

    assert result.gauges["loss_total"] == 0.7  # latest in time order
    assert result.gauges["replay_ratio"] == 0.1  # written once, still the latest

    # gpu_hours: only segment #1 (3.5s) completed; segment #2 never closes
    assert result.gpu_hours == pytest.approx(3.5 / 3600.0)

    # rates: games_completed/sims_run records span t=[1, 9] (5 records, window 8s)
    assert result.rates["games_per_hour"] == pytest.approx(5 / (8 / 3600))
    assert result.rates["sims_per_sec"] == pytest.approx(896 / 8)
    # learner_step total records span t=[4, 10]: (3 - 1) / (10 - 4)
    assert result.rates["learner_steps_per_sec"] == pytest.approx(2 / 6)

    # the x-axis join, exact up to one flush period (module docstring)
    assert result.checkpoints == {
        0: (0, 30.0, pytest.approx(0.0)),  # positions at/before t=3: 10 + 20
        1: (2, 45.0, pytest.approx(3.5 / 3600.0)),  # at/before t=7: +15 (t=5); segment #1 closed
        2: (3, 75.0, pytest.approx(3.5 / 3600.0)),  # at/before t=11: +25(t=8) +5(t=9)
    }

    # idempotent under re-reduction
    assert reduce_run(tmp_path) == result


# ==============================================================================
# 3. Driver wiring: ActorDriver's opt-in metrics flush
# ==============================================================================


def _stub_evaluator(game, state):
    """A minimal real evaluator (never ``None``) -- exercises the counting path."""
    return 0.0, {a: 0.0 for a in game.legal_moves(state)}


def _self_play_cfg(**overrides):
    return dataclasses.replace(load_run_config().self_play, sims=VALIDATE_TIER_SIMS, **overrides)


def test_actor_driver_flushes_games_sims_and_positions_at_game_boundaries(tmp_path):
    shard_dir, run_dir = tmp_path / "shards", tmp_path / "run"
    counter = PositionCounter()
    wrapped = count_positions(_stub_evaluator, counter)
    metrics_writer = EpochMetricsWriter(run_dir, "actor-0")
    driver = ActorDriver(
        game=TTT,
        self_play=_self_play_cfg(),
        run_id="run",
        actor_id=0,
        out_dir=shard_dir,
        run_seed=3,
        refresh=lambda: (wrapped, 0),
        max_games=3,
        metrics_writer=metrics_writer,
        position_counter=counter,
    )
    paths = driver.run()
    assert len(paths) == 3

    plies_per_game = [len(read_shard(p, TTT).records) for p in paths]

    records = list(iter_epoch_records(run_dir, "actor-0"))
    games = [r for r in records if r["series"] == "games_completed"]
    sims = [r for r in records if r["series"] == "sims_run"]
    positions = [r for r in records if r["series"] == "positions_evaluated"]

    assert len(games) == len(sims) == len(positions) == 3
    assert all(r["kind"] == "delta" for r in records)
    assert [r["value"] for r in games] == [1, 1, 1]
    assert [r["value"] for r in sims] == [p * VALIDATE_TIER_SIMS for p in plies_per_game]
    assert all(r["value"] > 0 for r in positions)  # a real evaluator was actually called

    # reduce_run sums exactly what was flushed -- no double counting, no loss
    result = reduce_run(run_dir)
    assert result.totals["games_completed"] == 3
    assert result.totals["sims_run"] == sum(p * VALIDATE_TIER_SIMS for p in plies_per_game)
    assert result.totals["positions_evaluated"] == sum(r["value"] for r in positions)


def test_actor_driver_without_a_position_counter_omits_that_series(tmp_path):
    shard_dir, run_dir = tmp_path / "shards", tmp_path / "run"
    metrics_writer = EpochMetricsWriter(run_dir, "actor-0")
    driver = ActorDriver(
        game=TTT,
        self_play=_self_play_cfg(),
        run_id="run",
        actor_id=0,
        out_dir=shard_dir,
        run_seed=3,
        refresh=lambda: (None, 0),  # the M0 uniform-prior path, no evaluator at all
        max_games=1,
        metrics_writer=metrics_writer,
    )
    driver.run()
    records = list(iter_epoch_records(run_dir, "actor-0"))
    series = {r["series"] for r in records}
    assert series == {"games_completed", "sims_run"}  # never a fabricated positions_evaluated


def test_actor_driver_without_metrics_writer_writes_nothing(tmp_path):
    shard_dir, run_dir = tmp_path / "shards", tmp_path / "run"
    driver = ActorDriver(
        game=TTT,
        self_play=_self_play_cfg(),
        run_id="run",
        actor_id=0,
        out_dir=shard_dir,
        run_seed=3,
        refresh=lambda: (None, 0),
        max_games=1,
    )
    driver.run()
    assert not (run_dir / "metrics").exists()  # backward-compatible: no wiring, no output


# ==============================================================================
# 4. Live-path smoke: a small, fast, single-process real run
# ==============================================================================


def _stub_run_config(training, run_seed=1):
    """A minimal duck-typed run-config stand-in (mirrors tests/test_ipc.py)."""
    return SimpleNamespace(
        training=training,
        run_seed=run_seed,
        to_dict=lambda: {"training": dataclasses.asdict(training), "run_seed": run_seed},
    )


@pytest.mark.slow
def test_live_path_smoke_a_real_single_process_run_reduces_to_sane_numbers(tmp_path):
    """Real ActorDriver + LearnerDriver + an orchestrator GPU-hour segment,
    wired through core.ipc's real seams, single-process (mirrors
    tests/test_ipc.py's layer-4 pattern -- no multiprocessing, kept fast).
    """
    shard_dir, ckpt_dir, run_dir = tmp_path / "shards", tmp_path / "ckpt", tmp_path / "run"
    training = TrainingConfig(
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
        publish_interval=1,
        checkpoint_count=100,
        # Kept high on purpose (mirrors tests/test_ipc.py): this test isolates
        # observability wiring, not D5 pacing dynamics, and a real floor/
        # ceiling enforcement would starve on so little data.
        replay_warmup_positions=1000,
    )
    learner = LearnerDriver(
        game=TTT,
        run_config=_stub_run_config(training),
        shard_dir=shard_dir,
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
        network_config=TTT_NET_CONFIG,
    )  # publishes v0

    orchestrator_writer = EpochMetricsWriter(run_dir, "orchestrator")
    orchestrator_writer.append(segment_start_record(device="cpu"))

    counter = PositionCounter()
    refresh = build_actor_refresh(
        game=TTT, ckpt_dir=ckpt_dir, network_config=TTT_NET_CONFIG, position_counter=counter
    )
    pacing = build_actor_pacing(run_dir)
    actor_metrics = EpochMetricsWriter(run_dir, "actor-0")
    actor = ActorDriver(
        game=TTT,
        self_play=_self_play_cfg(),
        run_id="live-smoke",
        actor_id=0,
        out_dir=shard_dir,
        run_seed=11,
        refresh=refresh,
        pacing=pacing,
        max_games=1,
        metrics_writer=actor_metrics,
        position_counter=counter,
    )

    num_rounds = 5
    for _ in range(num_rounds):
        actor.run()  # one game, at whatever version is currently latest
        learner._run_step()  # publish_interval=1 -> a new version every step

    orchestrator_writer.append(segment_end_record())

    for proc in ("actor-0", "learner", "orchestrator"):
        assert list(iter_epoch_records(run_dir, proc))  # every process wrote something parseable

    result = reduce_run(run_dir)

    assert result.totals["games_completed"] == num_rounds
    assert result.totals["positions_evaluated"] > 0
    assert result.totals["sims_run"] > 0
    assert result.totals["learner_step"] == num_rounds

    assert math.isfinite(result.gauges["loss_total"])
    assert math.isfinite(result.gauges["loss_value"])
    assert math.isfinite(result.gauges["loss_policy"])
    assert "loss_aux" not in result.gauges  # TTT declares no aux heads

    for value in result.rates.values():
        assert math.isnan(value) or math.isfinite(value)  # never inf, never a silent crash

    assert result.gpu_hours >= 0.0

    published = list_published_versions(ckpt_dir)
    assert sorted(result.checkpoints) == list(published)  # one coordinate per published version
    for learner_step, positions_evaluated, gpu_hours in result.checkpoints.values():
        assert learner_step >= 0
        assert positions_evaluated >= 0
        assert gpu_hours >= 0
