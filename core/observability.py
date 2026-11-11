"""The M3 observability contract: kinds, the reducer, the counting wrapper (issue #62).

``core.metrics`` owns *storage*: one append-only epoch-stamped JSONL file per
process, never shared, never rewound across a restart. This module owns
*meaning* -- what a record's ``kind`` field says about how it may be
aggregated, the one function (:func:`reduce_run`) allowed to aggregate them
over a whole run and its sibling (:func:`reduce_window`, §12 M5) that
aggregates one warm-up-excluded window of the same records, and the counting
primitive that turns network inference into the positions-evaluated series.
Design doc §1 (the primary criterion's plot
x-axes: cumulative network evaluations and GPU-hours) and §12 M3's
observability bullet are the source of this contract; M4's eval harness
(issue #72) consumes :func:`reduce_run`'s return value as-is, so every field
name and aggregation rule below is frozen, not an implementation detail.

**The three-kind taxonomy (every numeric series is exactly one).** A record's
``kind`` field is either a series kind or a structural record type:

* ``"delta"`` -- the increment observed since the writer's own previous
  flush. Deltas sum correctly across flushes, epochs, restarts, and
  processes with no double counting *by construction*: a restarted process
  opens a new epoch file (``core.metrics``) and its first delta after
  restart is relative to zero, not to whatever the dead process last held in
  memory. :data:`SERIES_GAMES_COMPLETED`, :data:`SERIES_POSITIONS_EVALUATED`,
  and :data:`SERIES_SIMS_RUN` are the pinned delta series (actor-owned).
  The *inference-batch accounting* series (§12 M5) are further actor-owned
  deltas, flushed with them at every game boundary but not among the series
  an M3 run is required to carry: :data:`SERIES_INFERENCE_BATCHES` counts
  network forward passes, and one ``inference_batch_size_*`` series per
  size bucket (:func:`inference_batch_size_series`: ``1``, ``2_3``, ``4_7``,
  ``8_15``, ``16_31``, ``32_63``, ``64_plus``) counts the forward passes whose
  leaf count fell in that bucket. A bucket series is written only when its
  count is non-zero, so an absent one reads as zero. The buckets sum to
  ``inference_batches``; mean batch fill is ``positions_evaluated /
  inference_batches`` exactly, and the buckets bound any percentile of it.
  The *per-tier move* series (§12 M5) are actor-owned deltas of the same
  standing -- flushed at the game boundary, not required of an M3 run:
  :data:`SERIES_MOVES_FULL_512`, :data:`SERIES_MOVES_MID_256` and
  :data:`SERIES_MOVES_FAST_64` count the plies a game searched at each D6
  playout-cap tier. They are written only by an actor running playout-cap
  randomization, and each only when its count is non-zero, so an absent one
  reads as zero; over such a run they sum to the plies played, and
  ``sims_run`` is ``512 * full + 256 * mid + 64 * fast`` exactly.
  The *position accounting* series (§12 M5) are actor-owned deltas of the
  same standing: :data:`SERIES_POSITIONS_PLAYED` counts the positions a game
  searched and played, :data:`SERIES_POSITIONS_STORED` the ones it kept as
  training samples (D12 drops the fast tier's). Both are written once per
  game, at every game boundary, in either mode, after the game's shard is
  published: ``positions_stored`` is exactly the number of samples that
  game put in its shard. With playout-cap randomization off the two are
  equal; with it on, ``positions_stored = moves_full_512 + moves_mid_256``
  and ``positions_played = positions_stored + moves_fast_64``.
  :data:`SERIES_CONSECUTIVE_MOVER_TRANSITIONS` is written with them, once
  per game and zero included: the plies of that game whose mover also made
  the ply before (a blocked skip: the adapter skipped an opponent with no
  legal action; an explicit pass action alternates and adds none). It is
  counted over every ply played, before any position is dropped, because
  the stored samples of a game with dropped plies no longer show it -- two
  adjacent stored samples sharing a mover may only have lost the ply
  between them.
* ``"gauge"`` -- a latest-value observation, never summed: loss components,
  the D5 replay ratio (learner-owned). Only the most recent record in run
  time order matters. :data:`SERIES_ACTOR_AMP_ACTIVE` is the one actor-owned
  gauge: ``1.0`` iff inference autocast was *observed* live inside the context
  an actor's leaf evaluations run under, ``0.0`` otherwise -- written once per
  actor epoch, at that actor's first evaluation (``core.ipc``), so every
  actor's own metrics file states what its forwards ran under.
* ``"total"`` -- a coordinator-owned exact cumulative that must never be
  summed across records (summing an already-cumulative value overcounts).
  :data:`SERIES_LEARNER_STEP` is the pinned total series: exactly one writer
  (the learner) ever advances it, so "the coordinator's max/last value" and
  "the true count" always agree.
* ``"segment_start"`` / ``"segment_end"`` -- the orchestrator's GPU-hour
  bracketing records (module docstring of :func:`reduce_run` below).
* ``"checkpoint_published"`` -- the learner's marker (``core.learner``,
  issue #60), unowned by this module but read by :func:`reduce_run`'s x-axis
  join.
* ``"spike_stop"`` -- the orchestrator's marker that a bounded run reached
  its run-wide completed-game target (:func:`spike_stop_record`; §12 M5).
  Written at most once per launch, before the shutdown it triggers, so its
  ``games_completed`` is the total observed at the stop: a signaled actor
  still finishes its in-flight game, and the run's final total may be higher.
  :func:`reduce_run` ignores it, like every kind it does not name.
* ``"shutdown_forced"`` -- the orchestrator's record that a child outlived
  the shutdown grace period and was killed (:func:`shutdown_forced_record`;
  ``core.ipc.LaunchedRun.shutdown``), one per killed process. A killed child
  ran no clean path -- an in-flight game or step was lost, its last flush
  may be missing -- so a run directory carrying one says its own evidence
  may be incomplete. :func:`reduce_run` ignores it too.

Every record additionally carries a ``timestamp`` (``time.time()``,
informational for equality purposes elsewhere but load-bearing for the
reducer's time-ordering -- see :func:`reduce_run`).

**Field names are part of the frozen contract.** A delta/gauge/total record
is ``{"kind", "series", "value", "timestamp"}`` -- :func:`delta_record`,
:func:`gauge_record`, and :func:`total_record` are the only sanctioned
builders, so every writer in this codebase (``core.actor``, ``core.learner``,
``core.ipc``) constructs records through them rather than hand-rolling dicts.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from core.metrics import iter_epoch_records, list_procs

# --- the kind taxonomy -------------------------------------------------------

KIND_DELTA = "delta"
KIND_GAUGE = "gauge"
KIND_TOTAL = "total"
KIND_SEGMENT_START = "segment_start"
KIND_SEGMENT_END = "segment_end"
KIND_SPIKE_STOP = "spike_stop"
KIND_SHUTDOWN_FORCED = "shutdown_forced"

# Owned by core.learner (issue #60); defined here so both core.learner (the
# writer) and this module (the reducer) share one literal with no drift.
# core.learner re-exports this name for backward-compatible imports.
CHECKPOINT_PUBLISHED_KIND = "checkpoint_published"

# --- pinned series names (design doc §1 x-axes + §13 throughput wall) -------

SERIES_GAMES_COMPLETED = "games_completed"
SERIES_POSITIONS_EVALUATED = "positions_evaluated"
SERIES_SIMS_RUN = "sims_run"
SERIES_LEARNER_STEP = "learner_step"

# The learner's per-step D5 replay-ratio gauge. ``core.learner`` owns and
# writes it; the name is repeated here, as ``learner_step``'s is, because
# that module imports this one.
SERIES_REPLAY_RATIO = "replay_ratio"

# Actor-owned gauge (module docstring): the observed inference-autocast state.
SERIES_ACTOR_AMP_ACTIVE = "actor_amp_active"

# Inference-batch accounting (module docstring): forward passes run, plus one
# delta series per leaf-count bucket. Bucket ``i`` holds sizes in
# ``[2**i, 2**(i+1))``; the last is open-ended.
SERIES_INFERENCE_BATCHES = "inference_batches"
_INFERENCE_BATCH_SIZE_SERIES = (
    "inference_batch_size_1",
    "inference_batch_size_2_3",
    "inference_batch_size_4_7",
    "inference_batch_size_8_15",
    "inference_batch_size_16_31",
    "inference_batch_size_32_63",
    "inference_batch_size_64_plus",
)

# Per-tier move accounting (module docstring): plies searched at each D6
# playout-cap tier. The budgets are in the names because an actor runs
# playout-cap randomization only at the D6 pins (``core.actor``).
SERIES_MOVES_FULL_512 = "moves_full_512"
SERIES_MOVES_MID_256 = "moves_mid_256"
SERIES_MOVES_FAST_64 = "moves_fast_64"

# Position accounting (module docstring): positions a game played, and the
# ones it stored as training samples.
SERIES_POSITIONS_PLAYED = "positions_played"
SERIES_POSITIONS_STORED = "positions_stored"

# Plies whose mover also made the previous ply (module docstring), counted
# over the complete played sequence.
SERIES_CONSECUTIVE_MOVER_TRANSITIONS = "consecutive_mover_transitions"

RATE_GAMES_PER_HOUR = "games_per_hour"
RATE_SIMS_PER_SEC = "sims_per_sec"
RATE_LEARNER_STEPS_PER_SEC = "learner_steps_per_sec"

_SECONDS_PER_HOUR = 3600.0


# --- record builders ---------------------------------------------------------


def _series_record(
    kind: str, series: str, value: float, *, timestamp: float | None = None
) -> dict[str, Any]:
    """Build one flat ``{kind, series, value, timestamp}`` record.

    Args:
        kind: One of :data:`KIND_DELTA` / :data:`KIND_GAUGE` / :data:`KIND_TOTAL`.
        series: The series name (module docstring's pinned names, or any
            forward-compatible additional series a future caller declares).
        value: The record's numeric payload.
        timestamp: Wall-clock seconds. Defaults to ``time.time()``.

    Returns:
        The record, ready for ``core.metrics.EpochMetricsWriter.append``.
    """
    return {
        "kind": kind,
        "series": series,
        "value": value,
        "timestamp": timestamp if timestamp is not None else time.time(),
    }


def delta_record(series: str, value: float, *, timestamp: float | None = None) -> dict[str, Any]:
    """Build one ``delta`` record -- the increment since the writer's last flush.

    Args:
        series: The series name.
        value: The increment (never negative -- a delta only ever grows a
            cumulative total).
        timestamp: Wall-clock seconds. Defaults to ``time.time()``.

    Returns:
        The record.
    """
    return _series_record(KIND_DELTA, series, value, timestamp=timestamp)


def gauge_record(series: str, value: float, *, timestamp: float | None = None) -> dict[str, Any]:
    """Build one ``gauge`` record -- a latest-value observation, never summed.

    Args:
        series: The series name.
        value: The observed value.
        timestamp: Wall-clock seconds. Defaults to ``time.time()``.

    Returns:
        The record.
    """
    return _series_record(KIND_GAUGE, series, value, timestamp=timestamp)


def total_record(series: str, value: float, *, timestamp: float | None = None) -> dict[str, Any]:
    """Build one ``total`` record -- a coordinator-owned exact cumulative.

    Args:
        series: The series name.
        value: The exact cumulative value as of this record (never a delta).
        timestamp: Wall-clock seconds. Defaults to ``time.time()``.

    Returns:
        The record.
    """
    return _series_record(KIND_TOTAL, series, value, timestamp=timestamp)


def is_real_cuda(device: str) -> bool:
    """Return whether ``device`` names a genuinely available CUDA device.

    "Real-CUDA-vs-CPU-fallback recorded so M5's numbers are comparable"
    (tasks/m3/011): a caller can ask for ``"cuda"`` on a machine where CUDA
    is unavailable (``core.ipc`` never inspects ``torch.cuda.is_available()``
    to *decide* the device -- module docstring there -- but the segment
    record still needs to know which one actually happened).

    Args:
        device: A torch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        ``True`` iff ``device`` names a CUDA device and CUDA is actually
        available in this process.
    """
    return device.lower().startswith("cuda") and torch.cuda.is_available()


def segment_start_record(*, device: str, timestamp: float | None = None) -> dict[str, Any]:
    """Build the orchestrator's GPU-hour segment-start record.

    Args:
        device: The torch device string the run was launched with.
        timestamp: Wall-clock seconds. Defaults to ``time.time()``.

    Returns:
        ``{"kind": "segment_start", "device", "is_cuda", "timestamp"}``.
    """
    return {
        "kind": KIND_SEGMENT_START,
        "device": device,
        "is_cuda": is_real_cuda(device),
        "timestamp": timestamp if timestamp is not None else time.time(),
    }


def segment_end_record(*, timestamp: float | None = None) -> dict[str, Any]:
    """Build the orchestrator's GPU-hour segment-end record.

    Args:
        timestamp: Wall-clock seconds. Defaults to ``time.time()``.

    Returns:
        ``{"kind": "segment_end", "timestamp"}``.
    """
    return {
        "kind": KIND_SEGMENT_END,
        "timestamp": timestamp if timestamp is not None else time.time(),
    }


def spike_stop_record(
    *, target: int, games_completed: int, timestamp: float | None = None
) -> dict[str, Any]:
    """Build the orchestrator's run-wide spike-stop record.

    Args:
        target: The run-wide completed-game target the launch was given.
        games_completed: The run-wide completed-game total observed when the
            target was found reached (at least ``target``).
        timestamp: Wall-clock seconds. Defaults to ``time.time()``.

    Returns:
        ``{"kind": "spike_stop", "target", "games_completed", "timestamp"}``.
    """
    return {
        "kind": KIND_SPIKE_STOP,
        "target": target,
        "games_completed": games_completed,
        "timestamp": timestamp if timestamp is not None else time.time(),
    }


def shutdown_forced_record(
    *, process: str, pid: int | None, timestamp: float | None = None
) -> dict[str, Any]:
    """Build the orchestrator's record of one child killed at shutdown.

    Args:
        process: The killed process's name (``"learner"``, ``"actor-<id>"``).
        pid: Its OS process id.
        timestamp: Wall-clock seconds. Defaults to ``time.time()``.

    Returns:
        ``{"kind": "shutdown_forced", "process", "pid", "timestamp"}``.
    """
    return {
        "kind": KIND_SHUTDOWN_FORCED,
        "process": process,
        "pid": pid,
        "timestamp": timestamp if timestamp is not None else time.time(),
    }


# --- the positions-evaluated counting wrapper --------------------------------


class PositionCounter:
    """A cumulative count of positions evaluated (design doc §1 x-axis).

    Not thread-safe: one :class:`~core.actor.ActorDriver` drives one
    single-threaded MCTS search through its evaluator synchronously, so a
    plain mutable counter is exactly the right amount of machinery.
    """

    def __init__(self) -> None:
        self._total = 0

    @property
    def total(self) -> int:
        """The count accumulated since construction or the last :meth:`drain`."""
        return self._total

    def add(self, n: int) -> None:
        """Add ``n`` positions to the running total.

        Args:
            n: Positions to add.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"position count must be >= 0, got {n}")
        self._total += n

    def drain(self) -> int:
        """Return the count accumulated since the last drain, resetting to 0.

        This is the actor-flush primitive: draining once per between-game
        flush yields exactly that game's positions-evaluated ``delta``
        (module docstring's taxonomy) -- the reducer never has to diff a
        cumulative itself.

        Returns:
            The count since the previous ``drain()`` (or construction).
        """
        n = self._total
        self._total = 0
        return n


def inference_batch_size_series(size: int) -> str:
    """Return the size-bucket series a forward pass over ``size`` leaves counts toward.

    Args:
        size: The number of leaves evaluated in one forward pass.

    Returns:
        The bucket's series name: ``inference_batch_size_1``, ``_2_3``,
        ``_4_7``, ``_8_15``, ``_16_31``, ``_32_63`` or ``_64_plus``.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError(f"inference batch size must be >= 1, got {size}")
    last = len(_INFERENCE_BATCH_SIZE_SERIES) - 1
    return _INFERENCE_BATCH_SIZE_SERIES[min(size.bit_length() - 1, last)]


class InferenceBatchCounter:
    """Counts network forward passes and how full each one was (§12 M5).

    The companion of :class:`PositionCounter` for the same evaluator calls:
    that one sums the leaves, this one counts the calls and buckets their
    sizes. Not thread-safe, for the same reason.
    """

    def __init__(self) -> None:
        """Start with no forward passes recorded."""
        self._batches = 0
        self._by_series: dict[str, int] = {}

    def add(self, n: int) -> None:
        """Record one evaluator call that evaluated ``n`` leaves.

        Args:
            n: The call's leaf count. ``0`` records nothing: an empty call ran
                no forward pass.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"inference batch size must be >= 0, got {n}")
        if n == 0:
            return
        series = inference_batch_size_series(n)
        self._batches += 1
        self._by_series[series] = self._by_series.get(series, 0) + 1

    def drain(self) -> dict[str, int]:
        """Return the deltas accumulated since the last drain, resetting to zero.

        Returns:
            ``{series: delta}``: always :data:`SERIES_INFERENCE_BATCHES` (``0``
            if no forward pass ran), plus each size-bucket series with a
            non-zero count.
        """
        drained = {SERIES_INFERENCE_BATCHES: self._batches, **self._by_series}
        self._batches = 0
        self._by_series = {}
        return drained


def count_positions(
    fn: Callable[..., Any],
    counter: PositionCounter | InferenceBatchCounter,
    *,
    batch_size: Callable[..., int] | None = None,
) -> Callable[..., Any]:
    """Wrap ``fn`` so every call adds the positions it evaluated to ``counter``.

    Positions evaluated, not forward calls, is the pinned unit (issue #62):
    at M3's real bridge (``core.network.make_network_evaluator``), each call
    evaluates exactly one leaf position, so the default ``batch_size``
    (unconditionally ``1``) already *is* "positions == forward calls". A
    future batched bridge (M5) supplies a ``batch_size`` callable that reads
    the call's actual cardinality from its arguments instead (e.g.
    ``lambda game, states: len(states)``) -- the wrapping mechanics here are
    unchanged either way, which is exactly the property that lets the §1
    x-axis survive M5 batching unchanged, without touching
    ``core.mcts``. This wrapper never alters ``fn``'s return value or
    behavior -- it only observes call cardinality and forwards through.

    Args:
        fn: The callable to wrap -- ``core.mcts.Evaluator``'s
            ``(game, state) -> (value, priors)`` shape by default, or any
            other call shape when ``batch_size`` is supplied to match it.
        counter: Receives every call's position count through ``add``: a
            :class:`PositionCounter` sums them, an
            :class:`InferenceBatchCounter` records each as one batch.
        batch_size: Given the exact ``(*args, **kwargs)`` a call passed to
            the wrapped function, returns how many positions that call
            represents. Defaults to an unconditional ``1``.

    Returns:
        A wrapped callable with ``fn``'s call signature and return value.
    """
    size_fn = batch_size if batch_size is not None else (lambda *args, **kwargs: 1)

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        counter.add(size_fn(*args, **kwargs))
        return result

    return wrapped


# --- the reducer --------------------------------------------------------------


@dataclass(frozen=True)
class ReducedRun:
    """The frozen return shape of :func:`reduce_run` (issue #62; consumed by M4 #72).

    Attributes:
        totals: Global cumulative totals, keyed by series name -- every
            ``delta`` series summed across every process/epoch/file, plus
            every ``total`` series at its coordinator-owned exact value
            (never summed). :data:`SERIES_GAMES_COMPLETED`,
            :data:`SERIES_POSITIONS_EVALUATED`, :data:`SERIES_SIMS_RUN`, and
            :data:`SERIES_LEARNER_STEP` are always present (``0.0`` if the
            series was never written); any other series name a caller
            declares appears dynamically when observed.
        rates: Exactly :data:`RATE_GAMES_PER_HOUR`, :data:`RATE_SIMS_PER_SEC`,
            and :data:`RATE_LEARNER_STEPS_PER_SEC` -- each an aggregate delta
            over an aggregate wall-clock window derived from that series' own
            flush timestamps (never by summing per-process instantaneous
            rates). ``float("nan")`` when the series has fewer than two
            records, or its window spans zero wall-clock time -- an
            undefined rate, stated rather than silently reported as ``0.0``.
        gauges: The latest-in-run-time-order value of every ``gauge`` series
            (loss components, replay ratio) -- never summed. Absent for a
            series never written (e.g. ``loss_aux`` on a no-aux game).
        gpu_hours: Total single-counted orchestrator-owned active GPU time,
            in hours, summed only over *completed* segments (a
            ``segment_start`` with a later matching ``segment_end``) --
            never multiplied by how many actor/learner processes shared the
            device. An unterminated segment (the orchestrator has not yet
            written, or never wrote, a matching end) contributes ``0.0``
            until it closes: a conservative, documented undercount, not an
            estimate.
        checkpoints: Per-version §1 x-axis coordinates, keyed by
            ``model_version``: ``(learner_step, positions_evaluated,
            gpu_hours)`` at each ``checkpoint_published`` marker's position
            in run time order. ``learner_step`` is exact (the marker's own
            field); ``positions_evaluated`` is the cumulative actor-delta sum
            at or before the marker (exact **up to one actor flush period** --
            a stated bound, never interpolated); ``gpu_hours`` is the
            single-counted completed-segment time plus elapsed active-segment
            time at the marker timestamp, even if that segment never closes.
    """

    totals: dict[str, float]
    rates: dict[str, float]
    gauges: dict[str, float]
    gpu_hours: float
    checkpoints: dict[int, tuple[int, float, float]]


def _ordered_records(run_dir: Path | str) -> list[tuple[float, str, int, dict[str, Any]]]:
    """Return every record under ``run_dir``, globally time-ordered, deterministically.

    **Time-ordering / tie-break rule (part of the frozen contract):** records
    are ordered by ``(timestamp, proc, per-proc sequence index)`` --
    ``per-proc sequence index`` is that process's own append order across
    every one of its epoch files, in ascending-epoch order
    (``core.metrics.iter_epoch_records``'s own contract). Two records can
    never share a full key (the per-proc sequence index alone already
    disambiguates same-process records; ``proc`` disambiguates the rest), so
    the order is total and reproducible even when many records share the
    exact same wall-clock ``timestamp`` (common at test speed, and possible
    in production between processes with no shared clock).

    Args:
        run_dir: The run's root directory.

    Returns:
        ``(timestamp, proc, seq, record)`` tuples in the pinned total order.
    """
    items: list[tuple[float, str, int, dict[str, Any]]] = []
    for proc in sorted(list_procs(run_dir)):
        for seq, rec in enumerate(iter_epoch_records(run_dir, proc)):
            items.append((float(rec.get("timestamp", 0.0)), proc, seq, rec))
    items.sort(key=lambda item: (item[0], item[1], item[2]))
    return items


def _delta_rate(
    window: tuple[float, float, int] | None, aggregate: float, *, unit_seconds: float
) -> float:
    """Return a delta series' rate: aggregate delta over its own flush window.

    Args:
        window: ``(first_timestamp, last_timestamp, record_count)`` across
            every record of this series, or ``None`` if never observed.
        aggregate: The series' summed delta value (``totals[series]``).
        unit_seconds: Seconds per rate unit (``3600.0`` for a per-hour rate,
            ``1.0`` for a per-second rate).

    Returns:
        ``aggregate / (span / unit_seconds)``, or ``float("nan")`` if fewer
        than two records exist or the window spans zero wall-clock time
        (module docstring's documented edge case).
    """
    if window is None or window[2] < 2:
        return float("nan")
    first_ts, last_ts, _ = window
    span = last_ts - first_ts
    if span <= 0:
        return float("nan")
    return aggregate / (span / unit_seconds)


def _total_rate(
    window: tuple[float, float, float, float, int] | None, *, unit_seconds: float
) -> float:
    """Return a ``total`` series' rate: the endpoint value difference over its window.

    A ``total`` series is already cumulative, so its own aggregate delta
    over a window is the *difference* between the window's first and last
    recorded values -- never a sum of the raw values (which are each
    already-cumulative and would wildly overcount).

    Args:
        window: ``(first_timestamp, first_value, last_timestamp, last_value,
            record_count)``, or ``None`` if never observed.
        unit_seconds: Seconds per rate unit.

    Returns:
        ``(last_value - first_value) / (span / unit_seconds)``, or
        ``float("nan")`` under the same edge cases as :func:`_delta_rate`.
    """
    if window is None or window[4] < 2:
        return float("nan")
    first_ts, first_val, last_ts, last_val, _ = window
    span = last_ts - first_ts
    if span <= 0:
        return float("nan")
    return (last_val - first_val) / (span / unit_seconds)


def reduce_run(run_dir: Path | str) -> ReducedRun:
    """Aggregate every per-process metrics file under ``run_dir`` into one report.

    The **only** sanctioned aggregation point (module docstring): nothing
    else anywhere sums a delta series, diffs a total series, or pairs GPU
    segments. Pure and idempotent -- reads every file under
    ``run_dir/metrics/`` and returns a fresh, independently-computed
    :class:`ReducedRun` each call; calling it twice in a row (or once before
    and once after an unrelated read) returns equal objects, and a process
    restart's new epoch file is picked up transparently on the next call
    (``core.metrics.iter_epoch_records`` already reads across every epoch).

    Single forward pass, in the pinned time order (:func:`_ordered_records`):
    delta values accumulate into per-series running sums; ``total`` records
    overwrite (never add to) their series' last-known value; ``gauge``
    records do the same; ``segment_start``/``segment_end`` pairs
    accumulate completed-segment seconds; and each ``checkpoint_published``
    marker snapshots the running positions-evaluated sum and GPU seconds
    (completed segments plus active-segment time through the marker
    timestamp) *as of that point in the scan* -- which, by construction of
    the single forward pass, is exactly "summed over every record at or
    before the marker's position in run time order".

    At most one segment is open at a time: the orchestrator is a single
    writer that never legitimately opens two at once, and it appends
    ``segment_end`` only on a clean shutdown (``core.ipc``). A
    ``segment_start`` that arrives while a segment is still open therefore
    means the earlier launch crashed: that segment is *abandoned*, credited
    only up to the last record observed before the new start (a lower bound
    on how long the crashed launch was alive -- never the downtime between
    the crash and the resume), and replaced by the new one. Abandoned
    credit counts toward both the top-level ``gpu_hours`` and every later
    checkpoint coordinate, so checkpoint coordinates stay non-decreasing
    across a crash/resume and a checkpoint published inside a closed
    segment never exceeds the final total. A segment still open at the end
    of the scan contributes to the checkpoint coordinates inside it but not
    to the top-level total (conservative: never estimated).

    Args:
        run_dir: The run's root directory (``core.metrics.metrics_dir``).

    Returns:
        The aggregated :class:`ReducedRun`.
    """
    ordered = _ordered_records(run_dir)

    delta_sums: dict[str, float] = {}
    delta_window: dict[str, tuple[float, float, int]] = {}
    total_last: dict[str, float] = {}
    total_window: dict[str, tuple[float, float, float, float, int]] = {}
    gauge_last: dict[str, float] = {}

    running_positions = 0.0
    gpu_seconds = 0.0
    open_segment_start: float | None = None
    last_ts: float | None = None
    checkpoints: dict[int, tuple[int, float, float]] = {}

    for ts, _proc, _seq, rec in ordered:
        kind = rec.get("kind")
        if kind == KIND_DELTA:
            series = rec["series"]
            value = float(rec["value"])
            delta_sums[series] = delta_sums.get(series, 0.0) + value
            if series == SERIES_POSITIONS_EVALUATED:
                running_positions += value
            window = delta_window.get(series)
            delta_window[series] = (ts, ts, 1) if window is None else (window[0], ts, window[2] + 1)
        elif kind == KIND_TOTAL:
            series = rec["series"]
            value = float(rec["value"])
            total_last[series] = value
            window = total_window.get(series)
            total_window[series] = (
                (ts, value, ts, value, 1)
                if window is None
                else (window[0], window[1], ts, value, window[4] + 1)
            )
        elif kind == KIND_GAUGE:
            gauge_last[rec["series"]] = float(rec["value"])
        elif kind == KIND_SEGMENT_START:
            if open_segment_start is not None and last_ts is not None:
                # The previous launch crashed without a segment_end: credit it
                # only through its last observed record, never the downtime.
                gpu_seconds += max(0.0, last_ts - open_segment_start)
            open_segment_start = ts
        elif kind == KIND_SEGMENT_END:
            if open_segment_start is not None:  # an unmatched end (no open start) is ignored
                gpu_seconds += max(0.0, ts - open_segment_start)
                open_segment_start = None
        elif kind == CHECKPOINT_PUBLISHED_KIND:
            active = 0.0 if open_segment_start is None else max(0.0, ts - open_segment_start)
            checkpoints[rec["model_version"]] = (
                rec["learner_step"],
                running_positions,
                (gpu_seconds + active) / _SECONDS_PER_HOUR,
            )
        last_ts = ts

    totals = {
        name: 0.0 for name in (SERIES_GAMES_COMPLETED, SERIES_POSITIONS_EVALUATED, SERIES_SIMS_RUN)
    }
    totals.update(delta_sums)
    totals[SERIES_LEARNER_STEP] = total_last.get(SERIES_LEARNER_STEP, 0.0)
    for name, value in total_last.items():
        totals[name] = value

    rates = {
        RATE_GAMES_PER_HOUR: _delta_rate(
            delta_window.get(SERIES_GAMES_COMPLETED),
            delta_sums.get(SERIES_GAMES_COMPLETED, 0.0),
            unit_seconds=_SECONDS_PER_HOUR,
        ),
        RATE_SIMS_PER_SEC: _delta_rate(
            delta_window.get(SERIES_SIMS_RUN),
            delta_sums.get(SERIES_SIMS_RUN, 0.0),
            unit_seconds=1.0,
        ),
        RATE_LEARNER_STEPS_PER_SEC: _total_rate(
            total_window.get(SERIES_LEARNER_STEP), unit_seconds=1.0
        ),
    }

    return ReducedRun(
        totals=totals,
        rates=rates,
        gauges=gauge_last,
        gpu_hours=gpu_seconds / _SECONDS_PER_HOUR,
        checkpoints=checkpoints,
    )


# --- the windowed reducer (§12 M5) ---------------------------------------------

# Which start condition bound a window (``WindowedRun.binding_condition``).
WINDOW_BOUND_BY_WARMUP_GAMES = "warmup_games"
WINDOW_BOUND_BY_REPLAY_WARMUP = "replay_warmup_positions"

# Insufficient-data reasons. The first three are window-level
# (``WindowedRun.reason``): no window exists, so no field holds a number.
WINDOW_TOO_FEW_GAMES = "too_few_games_for_window"
WINDOW_REPLAY_WARMUP_NOT_REACHED = "replay_warmup_positions_not_reached"
WINDOW_ZERO_SPAN = "zero_span_window"
# The rest are per-field (``WindowedRun.unavailable``), inside a sound window.
WINDOW_FEWER_THAN_TWO_RECORDS = "fewer_than_two_records"
WINDOW_NO_RECORD = "no_record"
WINDOW_ZERO_DENOMINATOR = "zero_denominator"
WINDOW_TOTAL_DECREASED = "total_decreased"

# The percentiles of forward-pass size ``WindowedRun.batch_fill_percentiles`` reports.
WINDOW_BATCH_FILL_PERCENTILES = (10, 50, 90)


@dataclass(frozen=True)
class WindowedRun:
    """The return shape of :func:`reduce_window`: one warm-up-excluded window (§12 M5).

    Every field but the two verdict fields is a measurement, and each is
    ``None`` exactly when :attr:`unavailable` names it -- never ``0.0``, never
    ``nan``, so arithmetic on a missing value fails loudly instead of
    comparing quietly.

    Attributes:
        reason: The window-level verdict. ``None`` when the window exists.
            Otherwise the named reason no window could be formed --
            :data:`WINDOW_TOO_FEW_GAMES` (the warm-up game count was never
            reached, or fewer than ``measure_games`` games completed after
            the start), :data:`WINDOW_REPLAY_WARMUP_NOT_REACHED` (cumulative
            stored positions never reached the threshold) or
            :data:`WINDOW_ZERO_SPAN` (the start and the end carry one
            timestamp) -- and every measurement is ``None``.
        unavailable: ``{field name: reason}`` for every measurement that is
            ``None``. Under a window-level verdict, every measurement mapped
            to :attr:`reason`. Inside a sound window, the per-field reasons:
            :data:`WINDOW_NO_RECORD` (the field's source series has no
            usable record: none in the window for a summed series, none at
            or before the window end for a latest-value one),
            :data:`WINDOW_FEWER_THAN_TWO_RECORDS` (the ``learner_step``
            series has no record at or before the window start, so the two
            endpoint readings its difference needs do not exist),
            :data:`WINDOW_TOTAL_DECREASED` (the ``learner_step`` total went
            backwards inside the window, so no endpoint difference counts
            its steps) and :data:`WINDOW_ZERO_DENOMINATOR` (a ratio whose
            series are present but whose denominator summed to zero).
        window_start: Timestamp of the start flush (:func:`reduce_window`).
        window_end: Timestamp of the end flush.
        start_game_ordinal: Run-wide cumulative ``games_completed`` through
            the start flush. Games up to and including it are warm-up.
        end_game_ordinal: Run-wide cumulative ``games_completed`` through the
            end flush, the last game of the window.
        binding_condition: Which start condition was met later --
            :data:`WINDOW_BOUND_BY_WARMUP_GAMES` or
            :data:`WINDOW_BOUND_BY_REPLAY_WARMUP`; the second when one flush
            meets both.
        games_in_window: ``end_game_ordinal - start_game_ordinal``:
            ``measure_games``, every actor writing one game per flush.
        games_per_hour: ``3600 * games_in_window / (window_end -
            window_start)`` -- the pinned rate form. Exact: both boundaries
            are game completions.
        stored_positions_per_hour: The window's summed ``positions_stored``
            deltas, per hour of the window span.
        sims_per_sec: The window's summed ``sims_run`` deltas, per second.
        net_evals_per_sec: The window's summed ``positions_evaluated``
            deltas, per second.
        learner_steps_in_window: The ``learner_step`` total series' endpoint
            difference: its last value at or before ``window_end`` minus its
            last value at or before ``window_start``. Never a sum of the
            series' values, which are each already cumulative. The two
            readings may be one record: no step completed in the window, an
            exact ``0`` (the learner writes one record per step). Unavailable
            if any reading in the window is below the one before it (a
            learner restarted from an older step on a resumed run).
        learner_steps_per_sec: ``learner_steps_in_window`` per second of the
            window span.
        replay_ratio_at_window_end: The learner's last ``replay_ratio``
            gauge at or before ``window_end``. Recorded beside the rate, not
            judged here: above the D5 ceiling it means the learner took no
            paced step by the window's end.
        moves_full_512: The window's summed ``moves_full_512`` deltas: plies
            searched at the always-full tier.
        moves_mid_256: The window's summed ``moves_mid_256`` deltas.
        moves_fast_64: The window's summed ``moves_fast_64`` deltas. An actor
            writes a tier series only when its count is non-zero, so once any
            of the three has a record in the window an absent one is ``0``;
            with none (a validate-tier run) all three are unavailable.
        mid_tier_fraction: ``moves_mid_256 / (moves_mid_256 +
            moves_fast_64)``: the empirical mid-tier fraction at plies >= 2,
            the always-full plies being exactly the ones before them.
        stored_positions_per_game: The window's summed ``positions_stored``
            deltas over ``games_in_window``.
        played_positions_per_game: The window's summed ``positions_played``
            deltas over ``games_in_window``.
        inference_batches: The window's summed ``inference_batches`` deltas:
            network forward passes.
        inference_batch_sizes: The size distribution of those forward
            passes: ``{bucket series: summed delta}`` over every
            ``inference_batch_size_*`` series, an absent bucket reading
            ``0`` (it is written only when non-zero).
        mean_batch_fill: The window's summed ``positions_evaluated`` over
            its summed ``inference_batches``: mean leaves per forward pass,
            exact.
        batch_fill_percentiles: ``{percentile: (lowest size, highest size)}``
            for each of :data:`WINDOW_BATCH_FILL_PERCENTILES`: the bounds of
            the size bucket holding that percentile's forward pass, by
            nearest rank over ``inference_batch_sizes``. Bounds, not an
            interpolated estimate; the highest is ``None`` in the open-ended
            top bucket.
        consecutive_mover_transitions: The window's summed
            ``consecutive_mover_transitions`` deltas.
        actor_amp_active: Whether inference autocast was observed live:
            ``True`` iff every ``actor_amp_active`` gauge record at or before
            ``window_end``, from every actor epoch, reads ``1.0``. One
            record reading otherwise makes it ``False``.
        is_cuda: The ``is_cuda`` field of the last ``segment_start`` record
            at or before ``window_end``: whether the launch the window ran
            in had a real CUDA device.
    """

    reason: str | None
    unavailable: dict[str, str]
    window_start: float | None = None
    window_end: float | None = None
    start_game_ordinal: int | None = None
    end_game_ordinal: int | None = None
    binding_condition: str | None = None
    games_in_window: int | None = None
    games_per_hour: float | None = None
    stored_positions_per_hour: float | None = None
    sims_per_sec: float | None = None
    net_evals_per_sec: float | None = None
    learner_steps_in_window: int | None = None
    learner_steps_per_sec: float | None = None
    replay_ratio_at_window_end: float | None = None
    moves_full_512: int | None = None
    moves_mid_256: int | None = None
    moves_fast_64: int | None = None
    mid_tier_fraction: float | None = None
    stored_positions_per_game: float | None = None
    played_positions_per_game: float | None = None
    inference_batches: int | None = None
    inference_batch_sizes: dict[str, int] | None = None
    mean_batch_fill: float | None = None
    batch_fill_percentiles: dict[int, tuple[int, int | None]] | None = None
    consecutive_mover_transitions: int | None = None
    actor_amp_active: bool | None = None
    is_cuda: bool | None = None


def _game_flushes(
    ordered: list[tuple[float, str, int, dict[str, Any]]],
) -> list[tuple[float, dict[str, Any]]]:
    """Group the actors' delta records into per-game flushes, in run time order.

    An actor flushes every series of a finished game on one timestamp,
    ``games_completed`` first (``core.actor``). A flush is therefore a
    ``games_completed`` delta plus the delta records that follow it from the
    same process on the same timestamp -- contiguous in the pinned order
    (:func:`_ordered_records`). A delta record outside any such group belongs
    to no game and is not counted.

    Args:
        ordered: Every record, in the pinned total order.

    Returns:
        ``(timestamp, {series: summed delta})`` per flush, in that order.
    """
    flushes: list[tuple[float, dict[str, Any]]] = []
    owner: tuple[str, float] | None = None
    for ts, proc, _seq, rec in ordered:
        if rec.get("kind") != KIND_DELTA:
            continue
        series = rec["series"]
        if series == SERIES_GAMES_COMPLETED:
            flushes.append((ts, {}))
            owner = (proc, ts)
        elif owner != (proc, ts):
            continue
        deltas = flushes[-1][1]
        deltas[series] = deltas.get(series, 0) + rec["value"]
    return flushes


def _window_start(
    cumulative_games: list[Any],
    cumulative_stored: list[Any],
    *,
    warmup_games: int,
    replay_warmup_positions: int,
) -> tuple[int | None, str]:
    """Apply the window-start rule -- the one place it is written (§12 M5).

    The window starts at the **later** of two flushes: (a) the one at which
    the run-wide cumulative ``games_completed`` reaches ``warmup_games``, and
    (b) the one at which the run-wide cumulative ``positions_stored`` reaches
    ``replay_warmup_positions`` -- the point from which the learner's D5
    pacing is live.

    ``warmup_games = 0`` is met before any game. (b) needs at least one
    stored position, which exists only once the first game is flushed, so
    (b) then binds and the window opens at the first flush or later: zero
    and one warm-up game give the same window.

    Args:
        cumulative_games: Run-wide cumulative ``games_completed`` through
            each flush, in run time order.
        cumulative_stored: Run-wide cumulative ``positions_stored`` through
            each flush, in the same order.
        warmup_games: Condition (a)'s game count.
        replay_warmup_positions: Condition (b)'s stored-position count,
            at least 1.

    Returns:
        ``(index of the start flush, which condition bound it)`` -- (b) when
        one flush meets both -- or ``(None, the insufficient-data reason)``
        when a condition is never met.
    """
    games_at = next((i for i, n in enumerate(cumulative_games) if n >= warmup_games), None)
    if games_at is None:
        return None, WINDOW_TOO_FEW_GAMES
    stored_at = next(
        (i for i, n in enumerate(cumulative_stored) if n >= replay_warmup_positions), None
    )
    if stored_at is None:
        return None, WINDOW_REPLAY_WARMUP_NOT_REACHED
    if stored_at >= games_at:
        return stored_at, WINDOW_BOUND_BY_REPLAY_WARMUP
    return games_at, WINDOW_BOUND_BY_WARMUP_GAMES


def _last_at_or_before(
    ordered: list[tuple[float, str, int, dict[str, Any]]], kind: str, series: str, timestamp: float
) -> Any:
    """Return a series' last recorded value at or before ``timestamp``.

    Args:
        ordered: Every record, in the pinned total order.
        kind: The record kind to read (:data:`KIND_TOTAL` or :data:`KIND_GAUGE`).
        series: The series name.
        timestamp: The inclusive upper bound.

    Returns:
        The value, or ``None`` if the series has no record that early.
    """
    value = None
    for ts, _proc, _seq, rec in ordered:
        if ts > timestamp:
            break
        if rec.get("kind") == kind and rec.get("series") == series:
            value = rec["value"]
    return value


def _ratio(numerator: Any, denominator: Any) -> float | None:
    """Return ``numerator / denominator``, or ``None`` where that is undefined.

    Args:
        numerator: The numerator, or ``None`` if its series has no record.
        denominator: The denominator, or ``None`` likewise.

    Returns:
        The quotient, or ``None`` if either is ``None`` or the denominator
        is zero.
    """
    return None if numerator is None or not denominator else numerator / denominator


def _batch_fill_percentiles(sizes: dict[str, Any]) -> dict[int, tuple[int, int | None]] | None:
    """Return the size-bucket bounds of each reported percentile of forward-pass size.

    Args:
        sizes: Forward-pass counts keyed by every bucket series, smallest
            bucket first (``_INFERENCE_BATCH_SIZE_SERIES`` order).

    Returns:
        ``{percentile: (lowest size, highest size)}`` over
        :data:`WINDOW_BATCH_FILL_PERCENTILES`, by nearest rank -- bucket
        ``i`` holds sizes ``2**i`` through ``2**(i+1) - 1`` and the top
        bucket has no highest size (``None``) -- or ``None`` if no forward
        pass was counted.
    """
    total = sum(sizes.values())
    if not total:
        return None
    top = len(sizes) - 1
    bounds: dict[int, tuple[int, int | None]] = {}
    for percentile in WINDOW_BATCH_FILL_PERCENTILES:
        rank = -(-percentile * total // 100)  # ceiling: the nearest-rank position
        seen = 0
        for i, count in enumerate(sizes.values()):
            seen += count
            if seen >= rank:
                bounds[percentile] = (2**i, None if i == top else 2 ** (i + 1) - 1)
                break
    return bounds


def _insufficient_window(reason: str) -> WindowedRun:
    """Return the verdict for a window that could not be formed.

    Args:
        reason: The window-level reason.

    Returns:
        A :class:`WindowedRun` with every measurement ``None`` and named
        unavailable for ``reason``.
    """
    measurements = [f.name for f in fields(WindowedRun) if f.name not in ("reason", "unavailable")]
    return WindowedRun(reason=reason, unavailable=dict.fromkeys(measurements, reason))


def reduce_window(
    run_dir: Path | str,
    *,
    warmup_games: int,
    measure_games: int,
    replay_warmup_positions: int,
) -> WindowedRun:
    """Reduce a run's metrics over one run-wide, warm-up-excluded window (§12 M5).

    The sibling of :func:`reduce_run`, over the same records in the same
    pinned order, and the only other place a delta series is summed or a
    total series differenced. :func:`reduce_run` is first-record-to-last
    with no warm-up slice and is unchanged; this slices one window out. It
    returns measurements and data-sufficiency verdicts only -- no floor, no
    pass or fail.

    **The window.** Run-wide: every actor's finished games, merged in the
    pinned record order, each game one flush (:func:`_game_flushes`). It
    starts at the later of two conditions (:func:`_window_start`) and ends at
    the flush at which the run-wide cumulative ``games_completed`` reaches
    the start's own cumulative count plus ``measure_games``. Both boundaries
    are read on one clock, the metrics records' timestamps. The games after
    the start flush through the end flush are the window's games; a game is
    placed by its run-wide ordinal, so one that shares the start's timestamp
    but is ordered after it is inside.

    **Aggregation rules.** A summed (``delta``) field adds the window's
    games' flushes, whole. A flush is stamped when its game finishes, so a
    window's sums include the part of its first games played before
    ``window_start`` and exclude the games still in flight at ``window_end``:
    each such field is exact **up to one actor flush period** (one game per
    actor) at each boundary -- a stated bound, never interpolated.
    ``games_per_hour`` carries no such bound. A ``total`` series
    (``learner_step``) is differenced between its endpoint readings, never
    summed. A ``gauge`` is read at its last record at or before the boundary.
    Learner and gauge records are placed by timestamp alone.

    **Insufficient data is an explicit verdict.** A window that cannot be
    formed returns a named :attr:`WindowedRun.reason` and no number; a field
    whose own series cannot support it is ``None`` with its reason in
    :attr:`WindowedRun.unavailable`.

    Args:
        run_dir: The run's root directory (``core.metrics.metrics_dir``).
        warmup_games: Completed games, run-wide, before the window may start
            (``throughput.warmup_games``).
        measure_games: Completed games the window spans
            (``throughput.measure_games``).
        replay_warmup_positions: Cumulative stored positions before the
            window may start (``training.replay_warmup_positions``).

    Returns:
        The :class:`WindowedRun`.

    Raises:
        ValueError: If ``warmup_games`` is negative, or
            ``replay_warmup_positions`` or ``measure_games`` is not positive
            (the bounds the run config puts on the three).
    """
    if warmup_games < 0 or replay_warmup_positions < 1 or measure_games < 1:
        raise ValueError(
            "need warmup_games >= 0, replay_warmup_positions >= 1 and measure_games >= 1; got "
            f"{warmup_games}, {replay_warmup_positions}, {measure_games}"
        )
    ordered = _ordered_records(run_dir)
    flushes = _game_flushes(ordered)
    cumulative_games: list[Any] = []
    cumulative_stored: list[Any] = []
    games = stored = 0
    for _ts, deltas in flushes:
        games += deltas[SERIES_GAMES_COMPLETED]
        stored += deltas.get(SERIES_POSITIONS_STORED, 0)
        cumulative_games.append(games)
        cumulative_stored.append(stored)

    start, binding = _window_start(
        cumulative_games,
        cumulative_stored,
        warmup_games=warmup_games,
        replay_warmup_positions=replay_warmup_positions,
    )
    if start is None:
        return _insufficient_window(binding)
    target = cumulative_games[start] + measure_games
    end = next((i for i in range(start + 1, len(flushes)) if cumulative_games[i] >= target), None)
    if end is None:
        return _insufficient_window(WINDOW_TOO_FEW_GAMES)
    window_start, window_end = flushes[start][0], flushes[end][0]
    span = window_end - window_start
    if span <= 0:
        return _insufficient_window(WINDOW_ZERO_SPAN)

    sums: dict[str, Any] = {}
    for _ts, deltas in flushes[start + 1 : end + 1]:
        for series, value in deltas.items():
            sums[series] = sums.get(series, 0) + value

    def per(series: str, unit_seconds: float) -> float | None:
        """Return a summed series' rate over the window.

        Args:
            series: The delta series.
            unit_seconds: Seconds in the rate's time unit.

        Returns:
            The window's sum per unit of time, or ``None`` if the series has
            no record in the window.
        """
        return sums[series] / (span / unit_seconds) if series in sums else None

    games_in_window = cumulative_games[end] - cumulative_games[start]
    step_at_start = _last_at_or_before(ordered, KIND_TOTAL, SERIES_LEARNER_STEP, window_start)
    # The reading at the start, then every reading inside the window, in order.
    steps = [step_at_start] + [
        rec["value"]
        for ts, _proc, _seq, rec in ordered
        if window_start < ts <= window_end
        and rec.get("kind") == KIND_TOTAL
        and rec.get("series") == SERIES_LEARNER_STEP
    ]
    steps_decreased = step_at_start is not None and any(
        later < earlier for earlier, later in pairwise(steps)
    )
    learner_steps = None if step_at_start is None or steps_decreased else steps[-1] - steps[0]

    tiers = (SERIES_MOVES_FULL_512, SERIES_MOVES_MID_256, SERIES_MOVES_FAST_64)
    has_tiers = any(series in sums for series in tiers)
    full, mid, fast = (sums.get(series, 0) if has_tiers else None for series in tiers)
    batches = sums.get(SERIES_INFERENCE_BATCHES)
    sizes = (
        None
        if batches is None
        else {series: sums.get(series, 0) for series in _INFERENCE_BATCH_SIZE_SERIES}
    )
    through_end = [rec for ts, _proc, _seq, rec in ordered if ts <= window_end]
    amp = [
        rec["value"]
        for rec in through_end
        if rec.get("kind") == KIND_GAUGE and rec.get("series") == SERIES_ACTOR_AMP_ACTIVE
    ]
    segments = [rec for rec in through_end if rec.get("kind") == KIND_SEGMENT_START]

    values: dict[str, Any] = {
        "window_start": window_start,
        "window_end": window_end,
        "start_game_ordinal": cumulative_games[start],
        "end_game_ordinal": cumulative_games[end],
        "binding_condition": binding,
        "games_in_window": games_in_window,
        "games_per_hour": games_in_window / (span / _SECONDS_PER_HOUR),
        "stored_positions_per_hour": per(SERIES_POSITIONS_STORED, _SECONDS_PER_HOUR),
        "sims_per_sec": per(SERIES_SIMS_RUN, 1.0),
        "net_evals_per_sec": per(SERIES_POSITIONS_EVALUATED, 1.0),
        "learner_steps_in_window": learner_steps,
        "learner_steps_per_sec": None if learner_steps is None else learner_steps / span,
        "replay_ratio_at_window_end": _last_at_or_before(
            ordered, KIND_GAUGE, SERIES_REPLAY_RATIO, window_end
        ),
        "moves_full_512": full,
        "moves_mid_256": mid,
        "moves_fast_64": fast,
        "mid_tier_fraction": _ratio(mid, mid + fast) if has_tiers else None,
        "stored_positions_per_game": _ratio(sums.get(SERIES_POSITIONS_STORED), games_in_window),
        "played_positions_per_game": _ratio(sums.get(SERIES_POSITIONS_PLAYED), games_in_window),
        "inference_batches": batches,
        "inference_batch_sizes": sizes,
        "mean_batch_fill": _ratio(sums.get(SERIES_POSITIONS_EVALUATED), batches),
        "batch_fill_percentiles": None if sizes is None else _batch_fill_percentiles(sizes),
        "consecutive_mover_transitions": sums.get(SERIES_CONSECUTIVE_MOVER_TRANSITIONS),
        "actor_amp_active": all(value == 1.0 for value in amp) if amp else None,
        "is_cuda": segments[-1].get("is_cuda") if segments else None,
    }
    # The reason a field is None when it is not simply a series with no record.
    reasons = dict.fromkeys(
        ("learner_steps_in_window", "learner_steps_per_sec"),
        WINDOW_TOTAL_DECREASED if steps_decreased else WINDOW_FEWER_THAN_TWO_RECORDS,
    )
    if has_tiers:
        reasons["mid_tier_fraction"] = WINDOW_ZERO_DENOMINATOR
    if batches is not None:
        reasons["batch_fill_percentiles"] = WINDOW_ZERO_DENOMINATOR
        if SERIES_POSITIONS_EVALUATED in sums:
            reasons["mean_batch_fill"] = WINDOW_ZERO_DENOMINATOR
    unavailable = {
        name: reasons.get(name, WINDOW_NO_RECORD) for name, value in values.items() if value is None
    }
    return WindowedRun(reason=None, unavailable=unavailable, **values)
