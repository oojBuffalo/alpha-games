"""The M3 observability contract: kinds, the reducer, the counting wrapper (issue #62).

``core.metrics`` owns *storage*: one append-only epoch-stamped JSONL file per
process, never shared, never rewound across a restart. This module owns
*meaning* -- what a record's ``kind`` field says about how it may be
aggregated, the one function (:func:`reduce_run`) allowed to aggregate them,
and the counting primitive that turns network inference into the
positions-evaluated series. Design doc §1 (the primary criterion's plot
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
* ``"gauge"`` -- a latest-value observation, never summed: loss components,
  the D5 replay ratio. Only the most recent record in run time order matters
  (learner-owned).
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
from dataclasses import dataclass
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

# Owned by core.learner (issue #60); defined here so both core.learner (the
# writer) and this module (the reducer) share one literal with no drift.
# core.learner re-exports this name for backward-compatible imports.
CHECKPOINT_PUBLISHED_KIND = "checkpoint_published"

# --- pinned series names (design doc §1 x-axes + §13 throughput wall) -------

SERIES_GAMES_COMPLETED = "games_completed"
SERIES_POSITIONS_EVALUATED = "positions_evaluated"
SERIES_SIMS_RUN = "sims_run"
SERIES_LEARNER_STEP = "learner_step"

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


def count_positions(
    fn: Callable[..., Any],
    counter: PositionCounter,
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
        counter: Accumulates every call's position count.
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
            single-counted segment time elapsed by the marker, under the same
            completed-segments-only rule as the top-level ``gpu_hours``.
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
    records do the same; ``segment_start``/``segment_end`` pairs (a LIFO
    stack of open starts -- the orchestrator is a single writer that never
    legitimately opens two segments at once, so this only matters for
    hand-built fixtures) accumulate completed-segment seconds; and each
    ``checkpoint_published`` marker snapshots the running
    positions-evaluated sum and completed-GPU-seconds *as of that point in
    the scan* -- which, by construction of the single forward pass, is
    exactly "summed over every record at or before the marker's position in
    run time order".

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
    open_segment_starts: list[float] = []
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
            open_segment_starts.append(ts)
        elif kind == KIND_SEGMENT_END:
            if open_segment_starts:  # an unmatched end (no open start) is ignored
                start_ts = open_segment_starts.pop()
                gpu_seconds += max(0.0, ts - start_ts)
        elif kind == CHECKPOINT_PUBLISHED_KIND:
            checkpoints[rec["model_version"]] = (
                rec["learner_step"],
                running_positions,
                gpu_seconds / _SECONDS_PER_HOUR,
            )

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
