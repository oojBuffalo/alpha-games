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

from core import learner as learner_module
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
    KIND_SHUTDOWN_FORCED,
    KIND_SPIKE_STOP,
    KIND_TOTAL,
    SERIES_INFERENCE_BATCHES,
    SERIES_LEARNER_STEP,
    SERIES_REPLAY_RATIO,
    WINDOW_BOUND_BY_REPLAY_WARMUP,
    WINDOW_BOUND_BY_WARMUP_GAMES,
    WINDOW_FEWER_THAN_TWO_RECORDS,
    WINDOW_NO_RECORD,
    WINDOW_REPLAY_WARMUP_NOT_REACHED,
    WINDOW_TOO_FEW_GAMES,
    WINDOW_TOTAL_DECREASED,
    WINDOW_ZERO_DENOMINATOR,
    WINDOW_ZERO_SPAN,
    InferenceBatchCounter,
    PositionCounter,
    ReducedRun,
    WindowedRun,
    _ordered_records,  # white-box: the pinned tie-break rule
    count_positions,
    delta_record,
    gauge_record,
    inference_batch_size_series,
    is_real_cuda,
    reduce_run,
    reduce_window,
    segment_end_record,
    segment_start_record,
    shutdown_forced_record,
    spike_stop_record,
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


def test_spike_stop_record_carries_the_target_and_the_observed_total():
    rec = spike_stop_record(target=6, games_completed=7, timestamp=3.0)
    assert rec == {"kind": KIND_SPIKE_STOP, "target": 6, "games_completed": 7, "timestamp": 3.0}
    before = time.time()
    assert spike_stop_record(target=1, games_completed=1)["timestamp"] >= before


def test_shutdown_forced_record_names_the_escalated_process():
    rec = shutdown_forced_record(process="actor-2", pid=4242, timestamp=5.0)
    assert rec == {
        "kind": KIND_SHUTDOWN_FORCED,
        "process": "actor-2",
        "pid": 4242,
        "timestamp": 5.0,
    }
    before = time.time()
    assert shutdown_forced_record(process="learner", pid=1)["timestamp"] >= before


def test_reduce_run_ignores_the_orchestrator_marker_records(tmp_path):
    orch = EpochMetricsWriter(tmp_path, "orchestrator")
    orch.append(segment_start_record(device="cpu", timestamp=0.0))
    orch.append(spike_stop_record(target=6, games_completed=6, timestamp=1800.0))
    orch.append(shutdown_forced_record(process="actor-0", pid=1, timestamp=3000.0))
    orch.append(segment_end_record(timestamp=3600.0))

    result = reduce_run(tmp_path)

    assert result.gpu_hours == 1.0
    assert result.totals == {
        "games_completed": 0.0,
        "positions_evaluated": 0.0,
        "sims_run": 0.0,
        "learner_step": 0.0,
    }
    assert result.gauges == {} and result.checkpoints == {}


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


def test_inference_batch_size_buckets_are_powers_of_two_with_an_open_top():
    expected = {
        1: "inference_batch_size_1",
        2: "inference_batch_size_2_3",
        3: "inference_batch_size_2_3",
        4: "inference_batch_size_4_7",
        7: "inference_batch_size_4_7",
        8: "inference_batch_size_8_15",
        15: "inference_batch_size_8_15",
        16: "inference_batch_size_16_31",
        31: "inference_batch_size_16_31",
        32: "inference_batch_size_32_63",
        63: "inference_batch_size_32_63",
        64: "inference_batch_size_64_plus",
        4096: "inference_batch_size_64_plus",
    }
    assert {size: inference_batch_size_series(size) for size in expected} == expected
    with pytest.raises(ValueError, match="batch size"):
        inference_batch_size_series(0)


def test_inference_batch_counter_drains_batches_and_nonzero_buckets_then_resets():
    counter = InferenceBatchCounter()
    assert counter.drain() == {SERIES_INFERENCE_BATCHES: 0}  # an honest zero, no buckets
    for size in (1, 1, 3, 8, 8, 8, 64):
        counter.add(size)
    counter.add(0)  # an empty call ran no forward: not a batch
    assert counter.drain() == {
        SERIES_INFERENCE_BATCHES: 7,
        "inference_batch_size_1": 2,
        "inference_batch_size_2_3": 1,
        "inference_batch_size_8_15": 3,
        "inference_batch_size_64_plus": 1,
    }
    assert counter.drain() == {SERIES_INFERENCE_BATCHES: 0}
    with pytest.raises(ValueError, match="batch size"):
        counter.add(-1)


def test_count_positions_feeds_a_batch_counter_one_batch_per_call():
    """One wrapper, two counters: positions sum the cardinalities, batches count the calls."""
    positions, batches = PositionCounter(), InferenceBatchCounter()

    def stub_batched_evaluate(game, states):
        return [(0.0, None)] * len(states)

    def size(game, states):
        return len(states)

    wrapped = count_positions(stub_batched_evaluate, positions, batch_size=size)
    wrapped = count_positions(wrapped, batches, batch_size=size)
    for states in (["a"], ["a", "b", "a"], ["a", "b", "c", "d", "e"]):
        assert wrapped("g", states) == [(0.0, None)] * len(states)
    drained = batches.drain()
    assert positions.total == 9 and drained[SERIES_INFERENCE_BATCHES] == 3
    # Mean fill is recoverable from the two totals: 9 leaves / 3 batches.
    assert sum(v for k, v in drained.items() if k != SERIES_INFERENCE_BATCHES) == 3


def test_reduce_run_sums_the_inference_batch_series_like_any_delta(tmp_path):
    for proc, games in (("actor-0", [(5, {1: 2, 4: 3}), (2, {1: 2})]), ("actor-1", [(1, {64: 1})])):
        writer = EpochMetricsWriter(tmp_path, proc)
        for batches, sizes in games:
            writer.append(delta_record(SERIES_INFERENCE_BATCHES, batches))
            for size, count in sizes.items():
                writer.append(delta_record(inference_batch_size_series(size), count))
    totals = reduce_run(tmp_path).totals
    assert totals[SERIES_INFERENCE_BATCHES] == 8
    assert totals["inference_batch_size_1"] == 4
    assert totals["inference_batch_size_4_7"] == 3
    assert totals["inference_batch_size_64_plus"] == 1
    assert "inference_batch_size_2_3" not in totals  # never observed => absent, i.e. zero


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
        0: (0, 30.0, pytest.approx(3.0 / 3600.0)),  # positions at/before t=3: 10 + 20
        1: (2, 45.0, pytest.approx(4.0 / 3600.0)),  # at/before t=7: +15 (t=5); segment #1 closed
        2: (3, 75.0, pytest.approx(8.0 / 3600.0)),  # at/before t=11: +25(t=8) +5(t=9)
    }

    # idempotent under re-reduction
    assert reduce_run(tmp_path) == result


def test_a_crashed_segment_superseded_by_a_resume_is_credited_only_through_its_last_record(
    tmp_path,
):
    """Crash then resume: launch A (t=0h) publishes v0 at 1h and dies with no
    ``segment_end``; launch B starts at 3h, publishes v1 at 4h, and ends at 5h.
    A is credited only through its last record (1h), never the 2h downtime, so
    v1 sits at 1h + 1h = 2h (not 5h) and the run total is 1h + 2h = 3h.
    """
    hour = 3600.0
    orch = EpochMetricsWriter(tmp_path, "orchestrator")
    learner = EpochMetricsWriter(tmp_path, "learner")
    orch.append(segment_start_record(device="cpu", timestamp=0.0))
    _append_checkpoint_published(learner, version=0, learner_step=0, timestamp=1 * hour)
    orch_resumed = EpochMetricsWriter(tmp_path, "orchestrator")  # A crashed: no segment_end
    orch_resumed.append(segment_start_record(device="cpu", timestamp=3 * hour))
    _append_checkpoint_published(learner, version=1, learner_step=10, timestamp=4 * hour)
    orch_resumed.append(segment_end_record(timestamp=5 * hour))

    result = reduce_run(tmp_path)

    gpu_axis = [result.checkpoints[v][2] for v in sorted(result.checkpoints)]
    assert gpu_axis == [pytest.approx(1.0), pytest.approx(2.0)]
    assert result.gpu_hours == pytest.approx(3.0)
    assert gpu_axis == sorted(gpu_axis)  # non-decreasing across the crash
    assert all(x <= result.gpu_hours for x in gpu_axis)


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
    # never a fabricated positions_evaluated
    assert series == {
        "games_completed",
        "sims_run",
        "positions_played",
        "positions_stored",
        "consecutive_mover_transitions",
    }


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


# ==============================================================================
# 5. The windowed reducer, scripted (§12 M5)
# ==============================================================================
#
# Every rate below is fixture arithmetic over hand-written timestamps. Nothing
# here measures throughput.


def _flush_game(
    writer, timestamp, *, stored=5, played=None, sims=100, evals=None, extra=(), omit=()
):
    """Append one finished game's flush: the actor's own series, on one timestamp.

    Args:
        writer: The actor's epoch writer.
        timestamp: The flush's single timestamp (the game's completion time).
        stored: The game's ``positions_stored`` delta.
        played: The game's ``positions_played`` delta. Defaults to ``stored``.
        sims: The game's ``sims_run`` delta.
        evals: The game's ``positions_evaluated`` delta, or ``None`` to omit
            the series (an actor with no position counter).
        extra: Further ``(series, value)`` deltas of the same flush.
        omit: Series names this flush does not write at all.
    """
    played = stored if played is None else played
    deltas = [
        ("games_completed", 1),
        ("sims_run", sims),
        ("positions_played", played),
        ("positions_stored", stored),
        *extra,
    ]
    if evals is not None:
        deltas.append(("positions_evaluated", evals))
    for series, value in deltas:
        if series not in omit:
            writer.append(delta_record(series, value, timestamp=timestamp))


def _seven_games(tmp_path, **flush_kwargs):
    """Seven games, 5 stored positions each, completing at t = 10, 20, ..., 70.

    Two actors alternate, so run-wide game ``k`` (1-based) completes at
    ``10 * k`` and cumulative stored positions after it are ``5 * k``.
    """
    actors = [EpochMetricsWriter(tmp_path, "actor-0"), EpochMetricsWriter(tmp_path, "actor-1")]
    for k in range(1, 8):
        _flush_game(actors[k % 2], 10.0 * k, **flush_kwargs)
    return actors


def _measurements(result):
    """Every field of a windowed result except its two verdict fields."""
    values = dataclasses.asdict(result)
    del values["reason"], values["unavailable"]
    return values


def _assert_insufficient(result, reason):
    """Assert a window-level verdict: a named reason, and no number anywhere."""
    assert result.reason == reason
    measurements = _measurements(result)
    assert all(value is None for value in measurements.values()), measurements
    assert result.unavailable == dict.fromkeys(measurements, reason)


def test_window_excludes_a_fast_warm_up_the_whole_run_average_includes(tmp_path):
    """The whole-run rate clears a floor that the warm-up-excluded window does not."""
    actors = [EpochMetricsWriter(tmp_path, "actor-0"), EpochMetricsWriter(tmp_path, "actor-1")]
    for k in range(1, 11):  # ten warm-up games, one a second
        _flush_game(actors[k % 2], float(k), sims=1000, evals=900)
    for k in range(1, 5):  # four measured games, one every hundred seconds
        _flush_game(actors[k % 2], 10.0 + 100.0 * k, sims=400, evals=300)
    floor = 100.0

    whole_run = reduce_run(tmp_path).rates["games_per_hour"]
    window = reduce_window(tmp_path, warmup_games=10, measure_games=4, replay_warmup_positions=20)

    assert whole_run == pytest.approx(14 * 3600 / 409)  # first record to last: ~123
    assert whole_run >= floor
    assert window.reason is None
    assert window.games_per_hour == pytest.approx(3600 * 4 / 400)  # 36
    assert window.games_per_hour < floor
    assert (window.window_start, window.window_end) == (10.0, 410.0)
    assert (window.start_game_ordinal, window.end_game_ordinal) == (10, 14)
    assert window.games_in_window == 4
    assert window.binding_condition == WINDOW_BOUND_BY_WARMUP_GAMES
    # Summed over the four measured games only: none of the warm-up's deltas.
    assert window.stored_positions_per_hour == pytest.approx(4 * 5 * 3600 / 400)
    assert window.sims_per_sec == pytest.approx(4 * 400 / 400)
    assert window.net_evals_per_sec == pytest.approx(4 * 300 / 400)


def test_window_starts_at_the_stored_positions_condition_when_it_is_the_later(tmp_path):
    _seven_games(tmp_path)
    # Game 2 meets the game count at t=20; stored positions reach 18 at game 4, t=40.
    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)

    assert window.reason is None
    assert window.binding_condition == WINDOW_BOUND_BY_REPLAY_WARMUP
    assert (window.window_start, window.window_end) == (40.0, 60.0)
    assert (window.start_game_ordinal, window.end_game_ordinal) == (4, 6)
    assert window.games_per_hour == pytest.approx(3600 * 2 / 20)


def test_window_starts_at_the_game_count_condition_when_it_is_the_later(tmp_path):
    _seven_games(tmp_path)
    # Stored positions reach 8 at game 2, t=20; game 3 meets the game count at t=30.
    window = reduce_window(tmp_path, warmup_games=3, measure_games=2, replay_warmup_positions=8)

    assert window.reason is None
    assert window.binding_condition == WINDOW_BOUND_BY_WARMUP_GAMES
    assert (window.window_start, window.window_end) == (30.0, 50.0)
    assert (window.start_game_ordinal, window.end_game_ordinal) == (3, 5)


def test_window_start_met_by_both_conditions_at_one_game_names_the_stored_positions(tmp_path):
    _seven_games(tmp_path)
    window = reduce_window(tmp_path, warmup_games=2, measure_games=1, replay_warmup_positions=10)
    assert window.window_start == 20.0
    assert window.binding_condition == WINDOW_BOUND_BY_REPLAY_WARMUP


def test_window_start_counts_stored_positions_not_played_ones(tmp_path):
    """Under the drop-policy most played positions are never stored; pacing follows stored."""
    _seven_games(tmp_path, stored=5, played=17)
    # Played positions pass 18 at game 2; stored positions only at game 4.
    window = reduce_window(tmp_path, warmup_games=1, measure_games=2, replay_warmup_positions=18)

    assert window.binding_condition == WINDOW_BOUND_BY_REPLAY_WARMUP
    assert (window.window_start, window.start_game_ordinal) == (40.0, 4)
    assert window.stored_positions_per_game == pytest.approx(5.0)
    assert window.played_positions_per_game == pytest.approx(17.0)


def test_window_with_no_warm_up_games_still_waits_for_the_stored_positions(tmp_path):
    _seven_games(tmp_path)
    window = reduce_window(tmp_path, warmup_games=0, measure_games=3, replay_warmup_positions=11)
    assert window.binding_condition == WINDOW_BOUND_BY_REPLAY_WARMUP
    assert (window.start_game_ordinal, window.end_game_ordinal) == (3, 6)


def test_window_with_no_warm_up_games_opens_no_earlier_than_the_first_stored_position(tmp_path):
    """Zero warm-up games is met before any game; the stored positions then bind.

    At least one stored position is always required, and the first exists
    only once the first game is flushed, so zero and one warm-up game open
    the same window -- and it is the stored-positions condition that bound it.
    """
    _seven_games(tmp_path)
    none = reduce_window(tmp_path, warmup_games=0, measure_games=6, replay_warmup_positions=1)
    one = reduce_window(tmp_path, warmup_games=1, measure_games=6, replay_warmup_positions=1)

    assert none == one
    assert none.binding_condition == WINDOW_BOUND_BY_REPLAY_WARMUP
    assert (none.start_game_ordinal, none.end_game_ordinal) == (1, 7)
    # Seven games cannot hold a seven-game window: the first only opens it.
    short = reduce_window(tmp_path, warmup_games=0, measure_games=7, replay_warmup_positions=1)
    _assert_insufficient(short, WINDOW_TOO_FEW_GAMES)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        # Seven games: the warm-up alone needs eight.
        (dict(warmup_games=8, measure_games=1, replay_warmup_positions=1), "too_few_games"),
        # The window opens at game 3 and needs five more; four remain.
        (dict(warmup_games=3, measure_games=5, replay_warmup_positions=1), "too_few_games"),
        # Thirty-five positions are stored in all.
        (dict(warmup_games=1, measure_games=1, replay_warmup_positions=36), "replay_warmup"),
    ],
    ids=["warm_up_never_ends", "measurement_never_ends", "stored_positions_never_reached"],
)
def test_window_that_never_completes_is_insufficient_data_with_a_named_reason(
    tmp_path, kwargs, reason
):
    _seven_games(tmp_path)
    expected = {
        "too_few_games": WINDOW_TOO_FEW_GAMES,
        "replay_warmup": WINDOW_REPLAY_WARMUP_NOT_REACHED,
    }[reason]
    _assert_insufficient(reduce_window(tmp_path, **kwargs), expected)


def test_window_over_an_empty_run_dir_is_insufficient_data(tmp_path):
    result = reduce_window(tmp_path, warmup_games=1, measure_games=1, replay_warmup_positions=1)
    _assert_insufficient(result, WINDOW_TOO_FEW_GAMES)


def test_window_spanning_zero_time_is_insufficient_data_never_a_rate(tmp_path):
    _flush_game(EpochMetricsWriter(tmp_path, "actor-0"), 10.0)
    _flush_game(EpochMetricsWriter(tmp_path, "actor-1"), 10.0)  # the same instant
    result = reduce_window(tmp_path, warmup_games=1, measure_games=1, replay_warmup_positions=1)
    _assert_insufficient(result, WINDOW_ZERO_SPAN)


def test_window_counts_a_game_by_its_run_wide_ordinal_not_its_timestamp_alone(tmp_path):
    """A game sharing the start's timestamp but ordered after it is inside the window."""
    a0, a1 = EpochMetricsWriter(tmp_path, "actor-0"), EpochMetricsWriter(tmp_path, "actor-1")
    _flush_game(a0, 10.0, sims=1)
    _flush_game(a0, 20.0, sims=10)  # run-wide game 2: the start
    _flush_game(a1, 20.0, sims=100)  # game 3, same timestamp, ordered after (proc name)
    _flush_game(a0, 30.0, sims=1000)  # game 4: the end

    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=1)

    assert (window.window_start, window.window_end) == (20.0, 30.0)
    assert (window.start_game_ordinal, window.end_game_ordinal) == (2, 4)
    # Games 3 and 4 only: the start game's own sims belong to the warm-up.
    assert window.sims_per_sec == pytest.approx((100 + 1000) / 10)


def _write_learner_steps(tmp_path, steps):
    """Append the learner's per-step ``learner_step`` total and ``replay_ratio`` gauge.

    Args:
        tmp_path: The run directory.
        steps: ``(timestamp, learner_step, replay_ratio)`` triples.
    """
    learner = EpochMetricsWriter(tmp_path, "learner")
    for timestamp, step, ratio in steps:
        learner.append(total_record("learner_step", step, timestamp=timestamp))
        learner.append(gauge_record(SERIES_REPLAY_RATIO, ratio, timestamp=timestamp))


def test_window_learner_fields_are_endpoint_readings_of_the_learner_series(tmp_path):
    _seven_games(tmp_path)
    _write_learner_steps(tmp_path, [(5.0, 3, 1.0), (35.0, 7, 5.5), (55.0, 9, 4.2), (65.0, 12, 3.9)])
    # The window is [40, 60]: step 7 is the reading at its start, step 9 at its end.
    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)

    assert window.learner_steps_in_window == 2
    assert window.learner_steps_per_sec == pytest.approx(2 / 20)
    assert window.replay_ratio_at_window_end == 4.2  # the last gauge at or before t=60


def test_window_learner_reading_on_a_boundary_timestamp_belongs_to_that_boundary(tmp_path):
    _seven_games(tmp_path)
    _write_learner_steps(tmp_path, [(40.0, 7, 5.5), (60.0, 9, 4.2)])  # exactly on [40, 60]
    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)

    assert window.learner_steps_in_window == 2
    assert window.replay_ratio_at_window_end == 4.2


@pytest.mark.parametrize(
    "steps",
    [
        [(35.0, 20, 1.0), (55.0, 11, 1.0)],  # ends below where it started
        [(35.0, 20, 1.0), (45.0, 11, 1.0), (55.0, 25, 1.0)],  # dips, then ends above
    ],
    ids=["ends_lower", "dips_and_recovers"],
)
def test_window_over_a_learner_step_total_that_went_backwards_reports_no_step_count(
    tmp_path, steps
):
    """A learner restarted from an older step inside the window: no honest difference."""
    _seven_games(tmp_path)
    _write_learner_steps(tmp_path, steps)
    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)

    assert window.reason is None
    for name in ("learner_steps_in_window", "learner_steps_per_sec"):
        assert getattr(window, name) is None, name
        assert window.unavailable[name] == WINDOW_TOTAL_DECREASED, name


def test_window_with_a_parked_learner_reports_zero_steps_as_a_measurement(tmp_path):
    """No step record inside the window is an exact zero, not missing data."""
    _seven_games(tmp_path)
    _write_learner_steps(tmp_path, [(35.0, 7, 5.5), (65.0, 8, 3.9)])
    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)

    assert window.learner_steps_in_window == 0
    assert window.learner_steps_per_sec == 0.0
    assert window.replay_ratio_at_window_end == 5.5  # above the ceiling: no paced step
    assert "learner_steps_per_sec" not in window.unavailable


def test_window_without_a_learner_reading_at_its_start_states_so_never_a_zero(tmp_path):
    _seven_games(tmp_path)
    learner = EpochMetricsWriter(tmp_path, "learner")
    learner.append(total_record("learner_step", 1, timestamp=45.0))  # first step: mid-window
    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)

    assert window.reason is None  # the window itself is sound
    assert window.games_per_hour == pytest.approx(3600 * 2 / 20)
    assert window.learner_steps_in_window is None
    assert window.learner_steps_per_sec is None
    assert window.replay_ratio_at_window_end is None
    assert window.unavailable["learner_steps_in_window"] == WINDOW_FEWER_THAN_TWO_RECORDS
    assert window.unavailable["learner_steps_per_sec"] == WINDOW_FEWER_THAN_TWO_RECORDS
    assert window.unavailable["replay_ratio_at_window_end"] == WINDOW_NO_RECORD


def test_window_field_is_none_exactly_when_it_is_named_unavailable(tmp_path):
    _seven_games(tmp_path)  # no positions_evaluated series, no learner
    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)

    assert window.net_evals_per_sec is None
    assert window.unavailable["net_evals_per_sec"] == WINDOW_NO_RECORD
    missing = {name for name, value in _measurements(window).items() if value is None}
    assert missing == set(window.unavailable)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(warmup_games=-1, measure_games=1, replay_warmup_positions=1),
        dict(warmup_games=1, measure_games=0, replay_warmup_positions=1),
        dict(warmup_games=1, measure_games=1, replay_warmup_positions=-1),
        dict(warmup_games=1, measure_games=1, replay_warmup_positions=0),
    ],
    ids=["negative_warmup", "zero_measure", "negative_replay_warmup", "zero_replay_warmup"],
)
def test_window_rejects_an_impossible_protocol(tmp_path, kwargs):
    with pytest.raises(ValueError):
        reduce_window(tmp_path, **kwargs)


def test_window_reads_the_same_series_names_the_learner_writes():
    assert SERIES_REPLAY_RATIO == learner_module.SERIES_REPLAY_RATIO
    assert SERIES_LEARNER_STEP == learner_module.SERIES_LEARNER_STEP


def test_reduce_run_is_unchanged_by_the_windowed_reducer(tmp_path):
    _seven_games(tmp_path, evals=7)
    _write_learner_steps(tmp_path, [(5.0, 3, 1.0), (65.0, 12, 3.9)])  # every rate finite
    before = reduce_run(tmp_path)
    reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)
    assert reduce_run(tmp_path) == before
    assert before.rates["games_per_hour"] == pytest.approx(7 * 3600 / 60)  # first to last
    assert [f.name for f in dataclasses.fields(ReducedRun)] == [
        "totals",
        "rates",
        "gauges",
        "gpu_hours",
        "checkpoints",
    ]


# The windowed reducer's contract, frozen: a field added, dropped, renamed or
# reordered is a deliberate change to what the downstream reports format.
WINDOWED_RUN_FIELDS = (
    "reason",
    "unavailable",
    "window_start",
    "window_end",
    "start_game_ordinal",
    "end_game_ordinal",
    "binding_condition",
    "games_in_window",
    "games_per_hour",
    "stored_positions_per_hour",
    "sims_per_sec",
    "net_evals_per_sec",
    "learner_steps_in_window",
    "learner_steps_per_sec",
    "replay_ratio_at_window_end",
    "moves_full_512",
    "moves_mid_256",
    "moves_fast_64",
    "mid_tier_fraction",
    "stored_positions_per_game",
    "played_positions_per_game",
    "inference_batches",
    "inference_batch_sizes",
    "mean_batch_fill",
    "batch_fill_percentiles",
    "consecutive_mover_transitions",
    "actor_amp_active",
    "is_cuda",
)


def test_windowed_run_field_list_is_frozen():
    assert tuple(f.name for f in dataclasses.fields(WindowedRun)) == WINDOWED_RUN_FIELDS


def _amp_gauge(writer, value, timestamp):
    writer.append(gauge_record("actor_amp_active", value, timestamp=timestamp))


def _every_series_run(tmp_path):
    """A two-actor schedule-shaped run carrying every series the reducer reads.

    Two warm-up games (t = 10, 20), then three measured ones (t = 30, 40, 50)
    whose deltas are all distinct from the warm-up's, so a warm-up delta
    leaking into any window sum changes it.
    """
    orch = EpochMetricsWriter(tmp_path, "orchestrator")
    orch.append({"kind": "segment_start", "device": "cuda", "is_cuda": True, "timestamp": 0.0})
    a0, a1 = EpochMetricsWriter(tmp_path, "actor-0"), EpochMetricsWriter(tmp_path, "actor-1")
    _amp_gauge(a0, 1.0, 1.0)
    _amp_gauge(a1, 1.0, 1.5)
    warm_up = [
        ("moves_full_512", 2),
        ("moves_mid_256", 7),
        ("moves_fast_64", 1),
        ("consecutive_mover_transitions", 9),
        ("inference_batches", 1000),
        ("inference_batch_size_2_3", 1000),
    ]
    for writer, timestamp in ((a0, 10.0), (a1, 20.0)):
        _flush_game(writer, timestamp, stored=9, played=10, sims=3456, evals=2500, extra=warm_up)
    _flush_game(
        a0,
        30.0,
        stored=4,
        played=10,
        sims=512 * 2 + 256 * 2 + 64 * 6,
        evals=1500,
        extra=[
            ("consecutive_mover_transitions", 1),
            ("moves_full_512", 2),
            ("moves_mid_256", 2),
            ("moves_fast_64", 6),
            ("inference_batches", 100),
            ("inference_batch_size_1", 10),
            ("inference_batch_size_8_15", 60),
            ("inference_batch_size_16_31", 30),
        ],
    )
    _flush_game(
        a1,
        40.0,
        stored=3,
        played=8,
        sims=512 * 2 + 256 * 1 + 64 * 5,
        evals=1000,
        extra=[
            ("consecutive_mover_transitions", 0),
            ("moves_full_512", 2),
            ("moves_mid_256", 1),
            ("moves_fast_64", 5),
            ("inference_batches", 50),
            ("inference_batch_size_8_15", 20),
            ("inference_batch_size_32_63", 30),
        ],
    )
    _flush_game(
        a0,
        50.0,
        stored=2,
        played=6,
        sims=512 * 2 + 64 * 4,
        evals=600,
        extra=[
            ("consecutive_mover_transitions", 2),
            ("moves_full_512", 2),  # no mid-tier ply: that series is not written
            ("moves_fast_64", 4),
            ("inference_batches", 50),
            ("inference_batch_size_1", 40),
            ("inference_batch_size_64_plus", 10),
        ],
    )
    _write_learner_steps(tmp_path, [(15.0, 4, 6.0), (35.0, 5, 4.5), (45.0, 7, 3.75)])
    return a0, a1


def test_window_over_a_run_carrying_every_series_reduces_field_by_field(tmp_path):
    _every_series_run(tmp_path)

    window = reduce_window(tmp_path, warmup_games=2, measure_games=3, replay_warmup_positions=18)

    sims = (512 * 2 + 256 * 2 + 64 * 6) + (512 * 2 + 256 + 64 * 5) + (512 * 2 + 64 * 4)
    assert dataclasses.asdict(window) == {
        "reason": None,
        "unavailable": {},
        "window_start": 20.0,
        "window_end": 50.0,
        "start_game_ordinal": 2,
        "end_game_ordinal": 5,
        "binding_condition": WINDOW_BOUND_BY_REPLAY_WARMUP,  # 18 stored at game 2, with (a)
        "games_in_window": 3,
        "games_per_hour": pytest.approx(3 * 3600 / 30),
        "stored_positions_per_hour": pytest.approx((4 + 3 + 2) * 3600 / 30),
        "sims_per_sec": pytest.approx(sims / 30),
        "net_evals_per_sec": pytest.approx((1500 + 1000 + 600) / 30),
        "learner_steps_in_window": 7 - 4,
        "learner_steps_per_sec": pytest.approx(3 / 30),
        "replay_ratio_at_window_end": 3.75,
        "moves_full_512": 6,
        "moves_mid_256": 3,
        "moves_fast_64": 15,
        "mid_tier_fraction": pytest.approx(3 / 18),
        "stored_positions_per_game": pytest.approx(9 / 3),
        "played_positions_per_game": pytest.approx(24 / 3),
        "inference_batches": 200,
        "inference_batch_sizes": {
            "inference_batch_size_1": 50,
            "inference_batch_size_2_3": 0,
            "inference_batch_size_4_7": 0,
            "inference_batch_size_8_15": 80,
            "inference_batch_size_16_31": 30,
            "inference_batch_size_32_63": 30,
            "inference_batch_size_64_plus": 10,
        },
        "mean_batch_fill": pytest.approx(3100 / 200),
        # Nearest rank over 200 forward passes: the 20th, 100th and 180th smallest.
        "batch_fill_percentiles": {10: (1, 1), 50: (8, 15), 90: (32, 63)},
        "consecutive_mover_transitions": 3,
        "actor_amp_active": True,
        "is_cuda": True,
    }
    # The schedule's own identity holds over the window's sums.
    assert sims == 512 * 6 + 256 * 3 + 64 * 15


def test_window_over_a_validate_tier_run_names_the_absent_series_never_zeroes(tmp_path):
    """No tier, batch, transition, AMP or segment record: each field says so."""
    _seven_games(tmp_path)  # its flushes carry only the always-written series
    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)

    assert window.reason is None
    assert window.stored_positions_per_game == pytest.approx(5.0)
    assert window.played_positions_per_game == pytest.approx(5.0)
    absent = (
        "moves_full_512",
        "moves_mid_256",
        "moves_fast_64",
        "mid_tier_fraction",
        "inference_batches",
        "inference_batch_sizes",
        "mean_batch_fill",
        "batch_fill_percentiles",
        "consecutive_mover_transitions",
        "actor_amp_active",
        "is_cuda",
    )
    for name in absent:
        assert getattr(window, name) is None, name
        assert window.unavailable[name] == WINDOW_NO_RECORD, name
    missing = {name for name, value in _measurements(window).items() if value is None}
    assert missing == set(window.unavailable)


def test_window_ratio_over_a_zero_denominator_is_named_never_computed(tmp_path):
    actor = EpochMetricsWriter(tmp_path, "actor-0")
    _flush_game(actor, 10.0)
    for timestamp in (20.0, 30.0):  # two-ply games: full tier only, no forward pass at all
        _flush_game(
            actor,
            timestamp,
            stored=2,
            evals=0,
            extra=[("moves_full_512", 2), ("inference_batches", 0)],
        )
    window = reduce_window(tmp_path, warmup_games=1, measure_games=2, replay_warmup_positions=1)

    assert (window.moves_full_512, window.moves_mid_256, window.moves_fast_64) == (4, 0, 0)
    assert window.inference_batches == 0
    assert set(window.inference_batch_sizes.values()) == {0}
    for name in ("mid_tier_fraction", "mean_batch_fill", "batch_fill_percentiles"):
        assert getattr(window, name) is None, name
        assert window.unavailable[name] == WINDOW_ZERO_DENOMINATOR, name


def test_window_batch_fill_percentile_in_the_open_top_bucket_has_no_upper_bound(tmp_path):
    actor = EpochMetricsWriter(tmp_path, "actor-0")
    _flush_game(actor, 10.0)
    extra = [("inference_batches", 4), ("inference_batch_size_64_plus", 4)]
    _flush_game(actor, 20.0, evals=400, extra=extra)
    window = reduce_window(tmp_path, warmup_games=1, measure_games=1, replay_warmup_positions=1)

    assert window.batch_fill_percentiles == {10: (64, None), 50: (64, None), 90: (64, None)}
    assert window.mean_batch_fill == pytest.approx(100.0)


def test_window_actor_amp_is_live_only_if_every_actor_record_through_the_window_end_is(
    tmp_path,
):
    a0, a1 = _every_series_run(tmp_path)
    kwargs = dict(warmup_games=2, measure_games=3, replay_warmup_positions=18)

    _amp_gauge(a1, 0.0, 60.0)  # a restart after the window's end (t=50): not its evidence
    assert reduce_window(tmp_path, **kwargs).actor_amp_active is True

    _amp_gauge(a1, 0.0, 45.0)  # one actor epoch inside it ran without autocast
    assert reduce_window(tmp_path, **kwargs).actor_amp_active is False

    _amp_gauge(a1, 1.0, 46.0)  # a later live epoch does not undo that
    assert reduce_window(tmp_path, **kwargs).actor_amp_active is False


def test_window_is_cuda_is_read_from_the_segment_the_window_ran_in(tmp_path):
    _every_series_run(tmp_path)  # opens a CUDA segment at t=0
    orch = EpochMetricsWriter(tmp_path, "orchestrator")
    orch.append(segment_start_record(device="cpu", timestamp=25.0))  # a CPU relaunch mid-window
    window = reduce_window(tmp_path, warmup_games=2, measure_games=3, replay_warmup_positions=18)
    assert window.is_cuda is False


def test_window_per_game_field_whose_series_is_absent_is_named_never_zero(tmp_path):
    actor = EpochMetricsWriter(tmp_path, "actor-0")
    _flush_game(actor, 10.0, omit=("positions_played",))
    for timestamp in (20.0, 30.0):
        _flush_game(actor, timestamp, omit=("positions_played", "positions_stored"))
    window = reduce_window(tmp_path, warmup_games=1, measure_games=2, replay_warmup_positions=5)

    assert window.reason is None
    fields = ("stored_positions_per_game", "played_positions_per_game", "stored_positions_per_hour")
    for name in fields:
        assert getattr(window, name) is None, name
        assert window.unavailable[name] == WINDOW_NO_RECORD, name


def test_window_ignores_a_delta_that_belongs_to_no_finished_game(tmp_path):
    actors = _seven_games(tmp_path)
    # Stray records inside the window [40, 60]: no games_completed opens their flush.
    actors[1].append(delta_record("sims_run", 999, timestamp=55.0))
    EpochMetricsWriter(tmp_path, "learner").append(delta_record("sims_run", 999, timestamp=50.0))
    window = reduce_window(tmp_path, warmup_games=2, measure_games=2, replay_warmup_positions=18)

    assert window.sims_per_sec == pytest.approx((100 + 100) / 20)


def test_window_batch_fill_percentile_is_the_nearest_rank_rounded_up(tmp_path):
    actor = EpochMetricsWriter(tmp_path, "actor-0")
    _flush_game(actor, 10.0)
    extra = [
        ("inference_batches", 3),
        ("inference_batch_size_1", 1),
        ("inference_batch_size_2_3", 1),
        ("inference_batch_size_4_7", 1),
    ]
    _flush_game(actor, 20.0, extra=extra)  # no positions_evaluated series
    window = reduce_window(tmp_path, warmup_games=1, measure_games=1, replay_warmup_positions=1)

    # Ranks ceil(0.3), ceil(1.5), ceil(2.7) of three passes: the 1st, 2nd and 3rd.
    assert window.batch_fill_percentiles == {10: (1, 1), 50: (2, 3), 90: (4, 7)}
    # Batches were counted but the leaves were not: missing data, not a zero denominator.
    assert window.mean_batch_fill is None
    assert window.unavailable["mean_batch_fill"] == WINDOW_NO_RECORD
