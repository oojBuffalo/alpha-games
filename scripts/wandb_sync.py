"""Optional Weights & Biases mirror over the run-dir metrics contract (issue #90).

Mirror-only: this script reads through ``core.metrics``/``core.observability``/
``core.run_identity``'s existing, frozen on-disk contract and pushes it into a
W&B run -- it never writes to ``<run_dir>/metrics/`` and never changes
``core.actor``/``core.learner``'s write path. The run directory stays the
source of truth; the only file this script itself writes is its own sync-state
sidecar (:data:`SYNC_STATE_FILENAME`), which is how re-running it against the
same run dir stays idempotent.

``wandb`` is strictly opt-in (the ``pyproject.toml`` ``[wandb]`` extra): the
only import of it anywhere in this repo is the guarded one in
:func:`_require_wandb`. ``core/`` and ``games/`` never import it.

**Idempotency: durable write-ahead reconciliation, not pass-order numbering.**
An earlier version of this module numbered rows by their position within each
freshly computed batch (a plain per-run counter, incremented once per row).
That numbering is *not* stable across a crash-and-retry when the retry's
batch differs from the crashed pass's batch -- e.g. a new record becomes
eligible between the crash and the retry -- because the row at position N in
the new batch is not necessarily the same underlying event as the row at
position N in the old one: retrying could then re-derive *different* rows
under the *same* step numbers (silently dropped as "already seen") while
shifting a genuinely new row onto a step number the server had already
accepted for something else (silently duplicated). Step numbers must
therefore be tied to a persisted plan, never recomputed on retry.

Every sync pass (:func:`sync_once`) is now: (1) compute a batch as a fully
rendered plan -- every row's metric-name/value payload and its explicit step,
plus the resulting sync state once the batch is applied
(:func:`_compute_batch`) -- against a private working copy, so nothing about
the confirmed state changes yet; (2) persist that plan into
:attr:`SyncState.pending_plan` and save the sidecar (:func:`save_sync_state`,
atomic temp-file rename) *before* logging a single row -- a durable,
replayable record of exactly what this pass intends to do; (3) log every row
in the plan, in order, ``commit=True``; (4) apply the plan's resulting state
and clear ``pending_plan``, then save the sidecar again (:func:`_run_plan`).
A crash at any point between step 2 and the end of step 4 leaves
``pending_plan`` set on disk; the next invocation (:func:`sync_once`,
one-shot or ``--follow``) detects this and *replays that exact plan* -- the
same rows, the same steps, the same order -- before computing anything new.
Newly-eligible records are, by construction, absent from a persisted plan
(the plan was rendered before they existed), so they are picked up by the
fresh batch computed immediately afterward, with fresh step numbers layered
on top of the now-applied state -- never renumbering or colliding with the
replayed plan. Replaying an already-persisted plan is itself safe to repeat
(a second crash mid-replay): every row in the plan always carries the same
step it was given when first computed, so replaying it again is just
re-submitting the same ``(step, payload)`` pairs.

**Why an explicit step at all.** Every :func:`~wandb.Run.log` call this
script makes carries an explicit, deterministic ``step`` -- part of the plan
a row belongs to, assigned once when that plan is computed and never
recomputed -- and ``commit=True`` so each row lands as its own complete
history entry rather than accumulating into a later one. On
``wandb.init(id=..., resume="allow")``, W&B's client seeds its next-step
counter one past the highest step the server has already recorded for that
run id (verified against the ``wandb`` 0.28 SDK source,
``sdk/internal/sender.py``'s ``_resume_state.step = last_step + 1`` and
``sdk/internal/handler.py``'s ``handle_request_partial_history``, which drops
-- client-side, before ever forwarding the row -- any subsequently logged row
whose ``step`` is less than that counter, emitting a local warning rather
than an error). Because a plan's steps never change once persisted, replaying
it after a crash reassigns exactly the same steps to exactly the same rows
every time, and the already-accepted prefix of any replay is dropped by that
mechanism. **The guarantee is "a row is logged to W&B history at most
once,"** covering an in-progress run directory whose eligible-record set
grows between a crash and a retry -- not just a same-batch replay --
contingent on the assumptions above (one ``wandb`` run id per run dir, nobody
else logging to that run id, and this script never logging a real -- not
replayed -- row outside the plan mechanism this docstring names).

Every chart's x-axis is one of this script's own ``wandb.define_metric``
custom step fields (below), never W&B's own implicit step -- which is
exactly what leaves that implicit ``_step`` counter free to serve as the
replay-protection mechanism above without colliding with anything a chart
actually reads.

**Custom x-axes, not wandb's implicit step.** W&B allows exactly one
``step_metric`` per metric name, so covering more than one x-axis for the
same values means parallel metric names, each logging the same numbers
under its own name (:func:`_define_metrics` is the single source of this
list):

* Learner loss/replay-ratio gauges: ``learner/learner_step`` only (issue #90
  asks for one axis here).
* Actor throughput (``games_completed``, ``sims_run``): wall-clock
  (``actor/wall_clock_s``, the existing ``actor/*`` series) and positions
  (``throughput/positions_evaluated``, via the parallel
  ``throughput/games_vs_positions`` / ``throughput/sims_vs_positions``
  series -- ``actor/positions_evaluated`` itself keeps its existing
  wall-clock axis unchanged; a fresh name avoids shadowing that).
* ``checkpoint_published`` markers: learner-step (``checkpoint/learner_step``,
  the existing ``checkpoint/*`` series), positions
  (``checkpoint/positions_evaluated_axis``, via
  ``checkpoint/marker_vs_positions``), and GPU-hours
  (``checkpoint/gpu_hours_axis``, via ``checkpoint/marker_vs_gpu_hours``) --
  sourced from ``ReducedRun.checkpoints`` so M4 can later plot Elo against
  any of those coordinates.

**Trailing groups are held back by default; ``--finalize`` flushes them.**
A flush group still open at the end of the records this pass can see -- the
learner mid-writing a step's gauges, an actor mid-writing a game's deltas --
might simply be caught between two of its writer's own appends, not actually
finished; logging it now and consuming its records would strand whatever
that writer appends next (no group header to attach to, discarded on the
next sync). By default (one-shot backfill *and* every ``--follow`` pass,
including the one after Ctrl-C), such a trailing group is held back in full,
un-consumed, for a later sync to complete. Pass ``--finalize`` only when the
run directory is known to be complete (the process that was writing it has
exited) -- it flushes every trailing group, from every process, once.

**Multi-actor throughput is buffered and released behind a global watermark
of clamped timestamps.** ``actor/*`` and ``throughput/*`` are meant to read
as one coherent timeline across every actor, but each actor's own file only
proves its *own* chronology -- actor 0 having reached t=40 says nothing
about whether actor 1's next group lands at t=20 or t=41. Every actor's
newly finalized groups are therefore buffered per-process
(:attr:`SyncState.actor_buffer`) rather than logged immediately; a group is
released only once the *global watermark* -- the minimum, across every actor
process this run has ever had, of that process's own highest observed
``effective_ts`` (:attr:`SyncState.actor_watermarks`, below) -- has reached
or passed it.

**Clock-rollback clamp, not a monotonicity assumption.** An earlier version
of this module claimed ``core.actor``'s ``time.time()`` timestamps were
verified non-decreasing within one process's file and used them directly for
ordering. That claim does not hold: ``time.time()`` is wall-clock, not
monotonic, and can step backwards under NTP correction or a manual clock
change, which would let a later-arriving, earlier-stamped group release
*after* a later-timestamped one has already reached W&B -- violating the
non-regressing timeline this mechanism exists to guarantee. Every group's
ordering/axis coordinate is therefore ``effective_ts``, a *monotonized*
clamp of the raw timestamp: the running max of a process's own raw
timestamps in file-append order, seeded from that process's persisted
high-water mark (:func:`_plan_actors`). ``effective_ts`` -- never the raw
timestamp -- drives every place ordering or a chart axis is at stake: the
per-process high-water mark, the global watermark, the
``(effective_ts, proc)`` release sort, and the logged ``actor/wall_clock_s``
value itself (a chart axis must never regress; the raw pre-clamp value
would). Ordering is exact when clocks behave; under a rollback, every group
written until the clock recovers past the prior high-water mark is pinned at
that high-water mark -- a bounded, documented distortion (a burst of groups
reporting one wall-clock coordinate) rather than a silent regression or a
dropped/duplicated row. This clamp is computed purely from each process's
own file-append order plus its persisted high-water mark, so it is
deterministic across crashes/re-reads -- rows rendered by the write-ahead
plan above already carry their final, clamped axis value, so replaying a
pending plan never needs to (and never does) recompute it.

Released groups are logged in this global sorted order with cumulative
totals computed at release time, in that order -- never the order they
happened to arrive in across polls. **Known stall behavior:** one actor that
has gone quiet (crashed, or simply slower than its siblings) holds back
every other actor's throughput data at its own last-seen ``effective_ts``
indefinitely -- the data is buffered, not lost, and ``--finalize`` (once the
run is known complete) flushes every buffered group regardless of
watermark. A brand-new actor process whose metrics file exists but has not
yet completed its first flush contributes no watermark at all and blocks
release the same way, for the same reason: this script cannot distinguish
"about to report" from "already dead" without ``--finalize``.

**Honest granularity.** Actor deltas land at between-game flush boundaries
(``core.actor.ActorDriver._flush_game_metrics``), the same convention
``core.observability.reduce_run`` documents: the ``positions_evaluated``
figure attached to a ``checkpoint_published`` marker (via
``ReducedRun.checkpoints``, reused here rather than re-derived) is exact only
up to one actor flush period.

Usage::

    python3 scripts/wandb_sync.py runs/blokus_duo/<run-id> --project alpha-games
    python3 scripts/wandb_sync.py runs/blokus_duo/<run-id> --project alpha-games --follow
    # once the run is known complete (flush every trailing/buffered group):
    python3 scripts/wandb_sync.py runs/blokus_duo/<run-id> --project alpha-games --finalize
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.metrics import iter_epoch_records, list_procs  # noqa: E402
from core.observability import (  # noqa: E402
    CHECKPOINT_PUBLISHED_KIND,
    KIND_DELTA,
    KIND_GAUGE,
    KIND_TOTAL,
    SERIES_GAMES_COMPLETED,
    SERIES_LEARNER_STEP,
    SERIES_POSITIONS_EVALUATED,
    SERIES_SIMS_RUN,
    reduce_run,
)
from core.run_identity import (  # noqa: E402
    LaunchConfig,
    compute_config_hash,
    read_run_record,
    read_stored_config,
)
from games.registry import build_game_factory  # noqa: E402

SYNC_STATE_FILENAME = ".wandb_sync_state.json"

LEARNER_PROC = "learner"
_NON_ACTOR_PROCS = frozenset({LEARNER_PROC, "orchestrator"})

DEFAULT_POLL_INTERVAL = 5.0


# ==============================================================================
# The guarded wandb import
# ==============================================================================


def _require_wandb() -> Any:
    """Import and return the ``wandb`` module, or raise a clear, install-hinted error.

    The only place in this repository that imports ``wandb`` -- ``core/`` and
    ``games/`` never do. Uses :func:`importlib.import_module` (rather than a
    bare ``import wandb`` statement) so a test can simulate "not installed"
    by patching ``importlib.import_module`` without needing an environment
    that genuinely lacks the package.

    Returns:
        The imported ``wandb`` module.

    Raises:
        ImportError: If ``wandb`` is not installed.
    """
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise ImportError(
            "wandb is required for scripts/wandb_sync.py. Install the optional extra with: "
            "python3 -m pip install -e '.[wandb]'"
        ) from exc


# ==============================================================================
# Sync-state persistence
# ==============================================================================


@dataclass
class SyncState:
    """This script's own idempotency bookkeeping (never part of the run-dir contract).

    Attributes:
        proc_cursors: For each ``core.metrics`` process name, how many of its
            durably-appended records have already been consumed by this
            script (``core.metrics.iter_epoch_records`` yields a stable,
            append-only sequence, so a plain count is a safe resume point).
        checkpoint_versions_synced: ``model_version`` values already logged
            as a checkpoint summary point.
        actor_totals: This script's own running cumulative sums of the actor
            delta series (``games_completed``/``positions_evaluated``/
            ``sims_run``), used to log ``actor/*``/``throughput/*`` as a
            running total rather than a per-event increment. Advanced only
            when a group is *released* (module docstring's watermark rule),
            in release order.
        actor_buffer: Per-actor-process finalized flush groups not yet
            released to W&B, pending the cross-process watermark (module
            docstring). Each entry is ``{"timestamp", "effective_ts",
            "deltas"}`` -- ``effective_ts`` is the monotonized clamp
            (module docstring's clock-rollback rule, :func:`_plan_actors`)
            added at ingestion time; ``timestamp`` is the raw, unclamped
            value, kept only for inspection/debugging.
        actor_watermarks: Per-actor-process high-water mark: the highest
            ``effective_ts`` ever observed from that process, whether or not
            that group has been released yet. Monotonically non-decreasing
            per process *by construction* of the clamp (module docstring),
            regardless of whether the underlying raw timestamps are; the
            global release watermark is the minimum of these across every
            actor process this run has ever had.
        next_step: The next explicit W&B ``step`` this script will assign to
            a newly computed plan's rows -- a running count of every row
            ever logged for this run. Never reused once a plan has been
            persisted (module docstring's write-ahead mechanism); a
            replayed plan carries its own already-assigned steps and does
            not consult this counter.
        pending_plan: A plan from :func:`_compute_batch` that has been
            persisted (module docstring) but not yet fully applied -- set
            just before this script starts logging its rows, cleared just
            after every row is logged and the plan's resulting state is
            applied. ``None`` when no pass is mid-flight. A non-``None``
            value found at startup means a prior pass crashed between those
            two points and must be replayed verbatim before anything else.
    """

    proc_cursors: dict[str, int] = field(default_factory=dict)
    checkpoint_versions_synced: list[int] = field(default_factory=list)
    actor_totals: dict[str, float] = field(default_factory=dict)
    actor_buffer: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    actor_watermarks: dict[str, float] = field(default_factory=dict)
    next_step: int = 0
    pending_plan: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this state as a plain, JSON-serializable dict."""
        return {
            "proc_cursors": dict(self.proc_cursors),
            "checkpoint_versions_synced": list(self.checkpoint_versions_synced),
            "actor_totals": dict(self.actor_totals),
            "actor_buffer": {proc: list(groups) for proc, groups in self.actor_buffer.items()},
            "actor_watermarks": dict(self.actor_watermarks),
            "next_step": self.next_step,
            "pending_plan": self.pending_plan,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SyncState:
        """Build a :class:`SyncState` from a parsed JSON object.

        Args:
            raw: The parsed sync-state file content.

        Returns:
            The reconstructed state.
        """
        return cls(
            proc_cursors=dict(raw.get("proc_cursors", {})),
            checkpoint_versions_synced=list(raw.get("checkpoint_versions_synced", [])),
            actor_totals=dict(raw.get("actor_totals", {})),
            actor_buffer={
                proc: list(groups) for proc, groups in raw.get("actor_buffer", {}).items()
            },
            actor_watermarks=dict(raw.get("actor_watermarks", {})),
            next_step=int(raw.get("next_step", 0)),
            pending_plan=raw.get("pending_plan"),
        )


def sync_state_path(run_dir: Path | str) -> Path:
    """Return the sync-state sidecar path for one run directory.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``run_dir / SYNC_STATE_FILENAME``.
    """
    return Path(run_dir) / SYNC_STATE_FILENAME


def load_sync_state(run_dir: Path | str) -> SyncState:
    """Load a run dir's sync state, or a fresh empty one if never synced before.

    Args:
        run_dir: The run's root directory.

    Returns:
        The loaded (or fresh) state.
    """
    path = sync_state_path(run_dir)
    if not path.exists():
        return SyncState()
    return SyncState.from_dict(json.loads(path.read_text()))


def save_sync_state(run_dir: Path | str, state: SyncState) -> None:
    """Durably write a run dir's sync state, atomically.

    Args:
        run_dir: The run's root directory.
        state: The state to persist.
    """
    path = sync_state_path(run_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), sort_keys=True))
    tmp.replace(path)


# ==============================================================================
# Record splitting: raw core.metrics records -> per-flush groups
# ==============================================================================


class _FlushGroupSplitter:
    """Shared finalize-on-boundary state machine behind both process splitters.

    A **group** is one between-flush write burst from a single, single-writer
    process -- a leading boundary record followed by zero or more member
    records, always written contiguously in that order
    (:func:`_split_learner_records`'s and :func:`_split_actor_records`'s own
    docstrings name each process's exact burst). A group is *finalized* the
    moment a later record proves the writer is done producing it: a new
    group's own boundary record, an unrelated side-channel record
    (:meth:`close_current`), or end-of-input under ``finalize=True``
    (:meth:`finish`). An unfinalized trailing group -- one a subsequent poll
    might still extend -- is held back in full, along with its own raw
    records, out of the returned consumed count.

    This class owns only the group bookkeeping (the current open group, the
    index it started at, which raw records are safe to consider consumed,
    and finalize-on-close/finalize-on-end-of-input); each process's own
    record *classification* -- which record starts a group, which belongs to
    the open one, which is a side channel -- stays in that process's own
    splitter function below, since learner and actor records don't share a
    classification rule.
    """

    def __init__(self) -> None:
        self._groups: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._current_start = 0
        self.consumed = 0

    def start_group(self, idx: int, seed: dict[str, Any]) -> None:
        """Close whatever group is open, then open a new one anchored at ``idx``.

        Args:
            idx: This boundary record's index in the caller's record slice.
            seed: The new group's initial payload (mutated further by later
                :meth:`extend_current` calls); must not already contain the
                private ``_closed`` key this class manages.
        """
        self.close_current()
        seed["_closed"] = False
        self._current = seed
        self._current_start = idx
        self._groups.append(seed)
        self.consumed = idx + 1

    def extend_current(self, idx: int, update: Callable[[dict[str, Any]], None]) -> None:
        """Fold one member record into the open group, if any group is open.

        Args:
            idx: This member record's index in the caller's record slice.
            update: Mutates the open group in place with this record's data.
        """
        if self._current is not None:
            update(self._current)
            self.consumed = idx + 1

    def close_current(self) -> None:
        """Finalize whatever group is open (a later record has proven it done)."""
        if self._current is not None:
            self._current["_closed"] = True
            self._current = None

    def mark_consumed(self, idx: int) -> None:
        """Advance ``consumed`` past a record that doesn't touch group state.

        Args:
            idx: That record's index in the caller's record slice.
        """
        self.consumed = idx + 1

    def finish(self, *, finalize: bool) -> tuple[list[dict[str, Any]], int]:
        """Close out the stream: optionally finalize a still-open trailing group.

        Args:
            finalize: Whether to also finalize a still-open trailing group.

        Returns:
            ``(finalized_groups, consumed)`` -- ``consumed`` is rewound to
            the still-open trailing group's own start index when one exists
            and ``finalize`` is ``False``, holding it (and its raw records)
            back in full.
        """
        if self._current is not None and finalize:
            self._current["_closed"] = True
        if self._current is not None and not self._current["_closed"]:
            self.consumed = self._current_start
        finalized = [g for g in self._groups if g["_closed"]]
        for g in finalized:
            del g["_closed"]
        return finalized, self.consumed


def _split_learner_records(
    records: list[dict[str, Any]], *, finalize: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Split learner records into finalized flush groups, checkpoints, and a consumed count.

    One group per ``core.learner.LearnerDriver._flush_step_metrics`` call: a
    ``KIND_TOTAL`` ``learner_step`` record followed immediately by that
    step's ``KIND_GAUGE`` records (loss components, replay ratio) -- always
    written contiguously, in that order, by a single-writer process
    (``core.learner``'s own call sequence: ``_flush_step_metrics`` then
    ``_maybe_publish``). A ``checkpoint_published`` record is a side channel,
    not part of any group: it finalizes whatever group precedes it (proof
    the learner has moved on) and is collected separately, never folded into
    a group's own payload.

    Args:
        records: Records already sliced to start at the caller's cursor.
        finalize: Whether to also finalize a still-open trailing group (the
            single-shot backfill's last pass under ``--finalize``, and
            ``--follow``'s pass on exit when the caller knows the run is
            complete; ``False`` otherwise -- module docstring).

    Returns:
        ``(finalized_groups, checkpoint_records, records_consumed)`` --
        ``records_consumed`` is how many leading records of ``records`` are
        safe to advance a cursor past.
    """
    splitter = _FlushGroupSplitter()
    checkpoints: list[dict[str, Any]] = []

    for idx, rec in enumerate(records):
        kind = rec.get("kind")
        if kind == CHECKPOINT_PUBLISHED_KIND:
            splitter.close_current()
            checkpoints.append(rec)
            splitter.mark_consumed(idx)
        elif kind == KIND_TOTAL and rec.get("series") == SERIES_LEARNER_STEP:
            splitter.start_group(
                idx, {"learner_step": rec["value"], "timestamp": rec["timestamp"], "gauges": {}}
            )
        elif kind == KIND_GAUGE:
            splitter.extend_current(
                idx, lambda g, rec=rec: g["gauges"].__setitem__(rec["series"], rec["value"])
            )

    groups, consumed = splitter.finish(finalize=finalize)
    return groups, checkpoints, consumed


def _split_actor_records(
    records: list[dict[str, Any]], *, finalize: bool
) -> tuple[list[dict[str, Any]], int]:
    """Split one actor's records into finalized per-game flush groups and a consumed count.

    One group per ``core.actor.ActorDriver._flush_game_metrics`` call:
    ``games_completed`` (always first, always exactly ``1``) followed by
    ``sims_run`` and, when a position counter is wired,
    ``positions_evaluated`` -- mirrors :func:`_split_learner_records`'s
    finalize-on-evidence rule, keyed on the next ``games_completed`` record
    (there is no checkpoint-marker analogue in an actor's own file).

    Args:
        records: Records already sliced to start at the caller's cursor.
        finalize: Whether to also finalize a still-open trailing group.

    Returns:
        ``(finalized_groups, records_consumed)``, each group shaped
        ``{"timestamp", "deltas": {series: value}}``.
    """
    splitter = _FlushGroupSplitter()

    for idx, rec in enumerate(records):
        if rec.get("kind") != KIND_DELTA:
            splitter.mark_consumed(idx)
            continue
        series = rec.get("series")
        if series == SERIES_GAMES_COMPLETED:
            splitter.start_group(idx, {"timestamp": rec["timestamp"], "deltas": {}})
        splitter.extend_current(
            idx, lambda g, rec=rec, series=series: g["deltas"].__setitem__(series, rec["value"])
        )

    groups, consumed = splitter.finish(finalize=finalize)
    return groups, consumed


def _actor_procs(run_dir: Path | str) -> list[str]:
    """Return every actor process name under ``run_dir``, sorted for determinism.

    Args:
        run_dir: The run's root directory.

    Returns:
        Every ``core.metrics.list_procs`` entry except ``learner``/``orchestrator``.
    """
    return sorted(p for p in list_procs(run_dir) if p not in _NON_ACTOR_PROCS)


# ==============================================================================
# W&B run construction: identity, config, tags, summary seed, custom axes
# ==============================================================================


def _summary_seed(run_dir: Path | str, launch_config: LaunchConfig, run_id: str) -> dict[str, Any]:
    """Return the provenance fields logged once as W&B summary values.

    Args:
        run_dir: The run's root directory.
        launch_config: The run's stored, validated config.
        run_id: The run's identity.

    Returns:
        ``run_id``, the config hash, and the adapter's orientation-table hash
        (``None`` for a game that declares no orientation table).
    """
    game = build_game_factory(launch_config.run)()
    return {
        "run_id": run_id,
        "config_hash": compute_config_hash(launch_config),
        "orientation_hash": game.orientation_table_hash,
    }


def _tags(launch_config: LaunchConfig) -> list[str]:
    """Return this run's short, filterable W&B tags.

    Args:
        launch_config: The run's stored, validated config.

    Returns:
        ``["game:<name>", "game_config:<name>"]``.
    """
    return [f"game:{launch_config.run.game}", f"game_config:{launch_config.run.game_config}"]


def _define_metrics(run: Any) -> None:
    """Register every custom x-axis this script logs against (module docstring).

    Issue #90's required axes, each its own ``define_metric`` call since W&B
    allows exactly one ``step_metric`` per metric name: ``learner_step`` for
    the learner's loss/gauge series; wall-clock and positions for actor
    throughput; learner-step, positions, and GPU-hours for checkpoint
    markers. Covering more than one axis for the same values means parallel
    metric names logging the same numbers (:func:`_plan_actors`,
    :func:`_plan_checkpoints`) -- the fresh ``throughput/positions_evaluated``
    and ``checkpoint/*_axis`` names below exist only so a *second* axis can
    be defined without shadowing the first (``actor/positions_evaluated``
    and ``checkpoint/positions_evaluated``/``checkpoint/gpu_hours`` keep
    their existing wall-clock/learner-step axis via the ``actor/*``/
    ``checkpoint/*`` globs, unchanged).

    Args:
        run: The live W&B run (or any test double exposing ``define_metric``).
    """
    run.define_metric("learner/learner_step")
    run.define_metric("learner/*", step_metric="learner/learner_step")

    run.define_metric("actor/wall_clock_s")
    run.define_metric("actor/*", step_metric="actor/wall_clock_s")
    run.define_metric("throughput/positions_evaluated")
    run.define_metric("throughput/games_vs_positions", step_metric="throughput/positions_evaluated")
    run.define_metric("throughput/sims_vs_positions", step_metric="throughput/positions_evaluated")

    run.define_metric("checkpoint/learner_step")
    run.define_metric("checkpoint/*", step_metric="checkpoint/learner_step")
    run.define_metric("checkpoint/positions_evaluated_axis")
    run.define_metric(
        "checkpoint/marker_vs_positions", step_metric="checkpoint/positions_evaluated_axis"
    )
    run.define_metric("checkpoint/gpu_hours_axis")
    run.define_metric("checkpoint/marker_vs_gpu_hours", step_metric="checkpoint/gpu_hours_axis")


def _init_wandb_run(
    wandb: Any, run_dir: Path | str, *, project: str, entity: str | None = None
) -> Any:
    """Start (or resume) the W&B run mirroring ``run_dir``.

    The W&B run id is the run's own ``run_id`` (``core.run_identity``), with
    ``resume="allow"``: syncing the same run dir a second time continues the
    same W&B run rather than creating a duplicate.

    Args:
        wandb: The imported ``wandb`` module (:func:`_require_wandb`).
        run_dir: The run's root directory.
        project: The W&B project name.
        entity: The W&B entity/team, or ``None`` for the caller's default.

    Returns:
        The live W&B run, with ``define_metric`` x-axes and the provenance
        summary fields already set.
    """
    root = Path(run_dir)
    launch_config = read_stored_config(root)
    record = read_run_record(root)

    run = wandb.init(
        id=record.run_id,
        project=project,
        entity=entity,
        resume="allow",
        config=launch_config.to_dict(),
        tags=_tags(launch_config),
    )
    for key, value in _summary_seed(root, launch_config, record.run_id).items():
        run.summary[key] = value

    _define_metrics(run)
    return run


# ==============================================================================
# Plan computation: pure functions from (run_dir, confirmed state) -> a plan
# ==============================================================================


def _working_copy(state: SyncState) -> SyncState:
    """A snapshot copy of ``state``'s mutable fields for :func:`_compute_batch` to work against.

    Never shares a mutable container with ``state``: :func:`_compute_batch`
    mutates the returned copy freely while computing a plan, with zero risk
    of the confirmed ``state`` changing before the caller decides to persist
    and execute that plan (module docstring's write-ahead mechanism).

    Args:
        state: The state to snapshot. Its own ``pending_plan`` is not part
            of this copy -- plan bookkeeping is the caller's concern, not a
            plan-computation input.

    Returns:
        An independent copy of every field :func:`_compute_batch` reads or
        writes.
    """
    return SyncState(
        proc_cursors=dict(state.proc_cursors),
        checkpoint_versions_synced=list(state.checkpoint_versions_synced),
        actor_totals=dict(state.actor_totals),
        actor_buffer={
            proc: [dict(g) for g in groups] for proc, groups in state.actor_buffer.items()
        },
        actor_watermarks=dict(state.actor_watermarks),
        next_step=state.next_step,
    )


def _plan_checkpoints(
    run_dir: Path | str, working: SyncState, new_checkpoints: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[tuple[str, Any]]]:
    """Render a row (plus a summary update) for every not-yet-synced checkpoint marker.

    Sourced from ``core.observability.reduce_run(run_dir).checkpoints`` --
    reusing the canonical ``(learner_step, positions_evaluated, gpu_hours)``
    join rather than re-deriving it (module docstring's "honest granularity"
    note is exactly ``reduce_run``'s own documented bound). Also renders the
    marker's ``model_version`` a second and third time under
    ``checkpoint/marker_vs_positions``/``checkpoint/marker_vs_gpu_hours`` so
    the positions/GPU-hours axes (:func:`_define_metrics`) carry data too.

    Args:
        run_dir: The run's root directory.
        working: Mutated in place: ``checkpoint_versions_synced`` and
            ``next_step``.
        new_checkpoints: Newly observed ``checkpoint_published`` records
            (:func:`_split_learner_records`'s second return value).

    Returns:
        ``(rows, summary_updates)``.
    """
    rows: list[dict[str, Any]] = []
    summary_updates: list[tuple[str, Any]] = []
    pending = sorted(
        {
            rec["model_version"]
            for rec in new_checkpoints
            if rec["model_version"] not in working.checkpoint_versions_synced
        }
    )
    if not pending:
        return rows, summary_updates

    reduced = reduce_run(run_dir)
    for version in pending:
        if version not in reduced.checkpoints:
            continue  # not yet visible to reduce_run's own scan; retried next sync
        learner_step, positions_evaluated, gpu_hours = reduced.checkpoints[version]
        step = working.next_step
        working.next_step += 1
        rows.append(
            {
                "payload": {
                    "checkpoint/model_version": version,
                    "checkpoint/learner_step": learner_step,
                    "checkpoint/positions_evaluated": positions_evaluated,
                    "checkpoint/gpu_hours": gpu_hours,
                    "checkpoint/positions_evaluated_axis": positions_evaluated,
                    "checkpoint/gpu_hours_axis": gpu_hours,
                    "checkpoint/marker_vs_positions": version,
                    "checkpoint/marker_vs_gpu_hours": version,
                },
                "step": step,
            }
        )
        summary_updates.append(
            (
                f"checkpoint_{version}",
                {
                    "learner_step": learner_step,
                    "positions_evaluated": positions_evaluated,
                    "gpu_hours": gpu_hours,
                },
            )
        )
        working.checkpoint_versions_synced.append(version)
    return rows, summary_updates


def _plan_learner(
    run_dir: Path | str, working: SyncState, *, finalize: bool
) -> tuple[list[dict[str, Any]], list[tuple[str, Any]]]:
    """Render this pass's new learner flush-group rows plus any new checkpoint rows.

    Pure: mutates only ``working`` (a :func:`_working_copy`), never the live
    run.

    Args:
        run_dir: The run's root directory.
        working: Mutated in place: the learner cursor, ``next_step``, and
            (via :func:`_plan_checkpoints`) checkpoint bookkeeping.
        finalize: Forwarded to :func:`_split_learner_records`.

    Returns:
        ``(rows, summary_updates)`` -- learner rows followed by any
        checkpoint rows, in the order they must be logged.
    """
    rows: list[dict[str, Any]] = []
    records = list(iter_epoch_records(run_dir, LEARNER_PROC))
    cursor = working.proc_cursors.get(LEARNER_PROC, 0)
    groups, checkpoints, consumed = _split_learner_records(records[cursor:], finalize=finalize)

    for group in groups:
        payload = {"learner/learner_step": group["learner_step"]}
        for series, value in group["gauges"].items():
            payload[f"learner/{series}"] = value
        step = working.next_step
        working.next_step += 1
        rows.append({"payload": payload, "step": step})
    working.proc_cursors[LEARNER_PROC] = cursor + consumed

    checkpoint_rows, summary_updates = _plan_checkpoints(run_dir, working, checkpoints)
    rows.extend(checkpoint_rows)
    return rows, summary_updates


def _actor_watermark(state: SyncState, procs: list[str]) -> float:
    """Return the global cross-actor release watermark (module docstring).

    The minimum, across every currently known actor process, of that
    process's own highest observed ``effective_ts`` -- a process this run
    has never seen a group from yet contributes ``-inf`` (blocks release
    entirely, the documented stall behavior) rather than being silently
    excluded.

    Args:
        state: This tool's sync state (reads :attr:`SyncState.actor_watermarks`).
        procs: Every actor process name currently known for this run dir.

    Returns:
        The watermark, or ``+inf`` if ``procs`` is empty (nothing to gate).
    """
    if not procs:
        return float("inf")
    return min(state.actor_watermarks.get(proc, float("-inf")) for proc in procs)


def _plan_actors(
    run_dir: Path | str, working: SyncState, *, finalize: bool
) -> list[dict[str, Any]]:
    """Render this pass's actor throughput rows: ingest, clamp, buffer, then release.

    Every actor process's newly finalized groups are timestamp-clamped
    (module docstring's monotonized ``effective_ts``, below) and added to
    ``working.actor_buffer``; only groups at or below the current global
    watermark (:func:`_actor_watermark`) -- or, under ``finalize=True``,
    every buffered group regardless -- are then released: merged across
    every process in global ``(effective_ts, proc)`` order and rendered as
    one coherent cumulative throughput curve.

    **Clock-rollback clamp.** ``core.actor`` stamps each flush with
    ``time.time()``, which is not guaranteed monotonic. Every group's
    ``effective_ts`` is the running max of its own process's raw timestamps
    seen so far (seeded from that process's persisted high-water mark,
    ``working.actor_watermarks``), computed purely from file append order
    and therefore deterministic across crashes/re-reads. ``effective_ts`` --
    never the raw timestamp -- is used everywhere ordering or a chart axis
    is at stake (module docstring).

    Args:
        run_dir: The run's root directory.
        working: Mutated in place: per-actor cursors, the buffer,
            watermarks, the running ``actor_totals`` cumulative sums, and
            ``next_step``.
        finalize: Forwarded to :func:`_split_actor_records`; also bypasses
            the watermark gate entirely for release (module docstring).

    Returns:
        The rendered rows, in release order.
    """
    rows: list[dict[str, Any]] = []
    record = read_run_record(run_dir)
    run_start_ts = _parse_iso_utc(record.created_at)
    procs = _actor_procs(run_dir)

    for proc in procs:
        records = list(iter_epoch_records(run_dir, proc))
        cursor = working.proc_cursors.get(proc, 0)
        groups, consumed = _split_actor_records(records[cursor:], finalize=finalize)
        working.proc_cursors[proc] = cursor + consumed
        if groups:
            effective_ts = working.actor_watermarks.get(proc, float("-inf"))
            for group in groups:
                effective_ts = max(effective_ts, group["timestamp"])
                group["effective_ts"] = effective_ts
            working.actor_buffer.setdefault(proc, []).extend(groups)
            working.actor_watermarks[proc] = effective_ts

    watermark = float("inf") if finalize else _actor_watermark(working, procs)

    released: list[tuple[str, dict[str, Any]]] = []
    for proc in list(working.actor_buffer):
        buffered = working.actor_buffer[proc]
        ready = [g for g in buffered if g["effective_ts"] <= watermark]
        if not ready:
            continue
        held_back = [g for g in buffered if g["effective_ts"] > watermark]
        released.extend((proc, g) for g in ready)
        if held_back:
            working.actor_buffer[proc] = held_back
        else:
            del working.actor_buffer[proc]

    released.sort(key=lambda item: (item[1]["effective_ts"], item[0]))

    for proc, group in released:
        for series, value in group["deltas"].items():
            working.actor_totals[series] = working.actor_totals.get(series, 0.0) + value
        games_total = working.actor_totals.get(SERIES_GAMES_COMPLETED, 0.0)
        positions_total = working.actor_totals.get(SERIES_POSITIONS_EVALUATED, 0.0)
        sims_total = working.actor_totals.get(SERIES_SIMS_RUN, 0.0)
        step = working.next_step
        working.next_step += 1
        rows.append(
            {
                "payload": {
                    "actor/wall_clock_s": group["effective_ts"] - run_start_ts,
                    "actor/proc": proc,
                    "actor/games_completed": games_total,
                    "actor/positions_evaluated": positions_total,
                    "actor/sims_run": sims_total,
                    "throughput/positions_evaluated": positions_total,
                    "throughput/games_vs_positions": games_total,
                    "throughput/sims_vs_positions": sims_total,
                },
                "step": step,
            }
        )

    return rows


def _parse_iso_utc(stamp: str) -> float:
    """Parse a ``core.run_identity.iso_now`` timestamp back to epoch seconds.

    Args:
        stamp: A ``"YYYY-MM-DDTHH:MM:SSZ"`` string.

    Returns:
        Epoch seconds (UTC).
    """
    import calendar

    return float(calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))


def _compute_batch(run_dir: Path | str, state: SyncState, *, finalize: bool) -> dict[str, Any]:
    """Pure: compute the next batch of rows to log against a snapshot of ``state``.

    Never touches a live run and never mutates ``state`` -- returns
    everything the caller needs to persist as a recoverable pending plan and
    then execute it (:func:`_run_plan`, module docstring's write-ahead
    mechanism).

    Args:
        run_dir: The run's root directory.
        state: The current confirmed sync state (read-only; a working copy
            is computed against, per :func:`_working_copy`).
        finalize: Forwarded to the record splitters and the actor release
            gate.

    Returns:
        ``{"rows": [...], "summary_updates": [...], "post_state": {...}}``
        -- ``rows`` and ``summary_updates`` in the exact order they must be
        applied; ``post_state`` is the plain dict ``state`` should become
        once every row has been logged (:func:`_apply_plan`).
    """
    working = _working_copy(state)
    rows: list[dict[str, Any]] = []
    summary_updates: list[tuple[str, Any]] = []

    learner_rows, ckpt_updates = _plan_learner(run_dir, working, finalize=finalize)
    rows.extend(learner_rows)
    summary_updates.extend(ckpt_updates)

    rows.extend(_plan_actors(run_dir, working, finalize=finalize))

    return {
        "rows": rows,
        "summary_updates": summary_updates,
        "post_state": {
            "proc_cursors": working.proc_cursors,
            "checkpoint_versions_synced": working.checkpoint_versions_synced,
            "actor_totals": working.actor_totals,
            "actor_buffer": working.actor_buffer,
            "actor_watermarks": working.actor_watermarks,
            "next_step": working.next_step,
        },
    }


def _apply_plan(state: SyncState, plan: dict[str, Any]) -> None:
    """Advance ``state``'s cursors/buffers/totals/next_step to ``plan``'s resulting values.

    Args:
        state: Mutated in place.
        plan: A plan from :func:`_compute_batch`.
    """
    post = plan["post_state"]
    state.proc_cursors = dict(post["proc_cursors"])
    state.checkpoint_versions_synced = list(post["checkpoint_versions_synced"])
    state.actor_totals = dict(post["actor_totals"])
    state.actor_buffer = {proc: list(groups) for proc, groups in post["actor_buffer"].items()}
    state.actor_watermarks = dict(post["actor_watermarks"])
    state.next_step = post["next_step"]


def _run_plan(run: Any, run_dir: Path | str, state: SyncState, plan: dict[str, Any]) -> None:
    """Execute one row-logging plan under the write-ahead protocol (module docstring).

    Persists ``plan`` as ``state.pending_plan`` FIRST (durable before any
    ``run.log`` calls) so a crash at any point during logging leaves a
    recoverable, exact record of what to replay; only once every row has
    been (re-)logged does this apply the plan's resulting state and clear
    the pending marker, in one further atomic write. Replaying an
    already-persisted plan (this function called again with the same
    ``plan`` after a crash, via ``state.pending_plan``) is exactly this same
    call again -- logging every row a second time is safe under the
    explicit-step mechanism (module docstring): rows the server already
    accepted are dropped, rows it never saw are accepted for the first time.

    Args:
        run: The live W&B run.
        run_dir: The run's root directory.
        state: Mutated in place: ``pending_plan`` while executing, then the
            plan's ``post_state`` fields once every row is logged.
        plan: A plan from :func:`_compute_batch` (or ``state.pending_plan``
            itself, to replay one left over from a crashed pass). Must have
            a non-empty ``rows`` list -- callers with nothing to log should
            not call this at all.
    """
    state.pending_plan = plan
    save_sync_state(run_dir, state)

    for row in plan["rows"]:
        run.log(row["payload"], step=row["step"], commit=True)
    for key, value in plan["summary_updates"]:
        run.summary[key] = value

    _apply_plan(state, plan)
    state.pending_plan = None
    save_sync_state(run_dir, state)


def sync_once(run: Any, run_dir: Path | str, state: SyncState, *, finalize: bool = False) -> bool:
    """Run one full sync pass: replay any crashed pass's plan, then compute and run a new one.

    If ``state.pending_plan`` is set (a prior pass crashed after persisting
    a plan but before finishing it), that plan is replayed verbatim first --
    same rows, same steps, same order -- before anything else happens
    (module docstring). A fresh batch is then computed against the
    now-current state; if it has any rows, it is executed the same
    write-ahead-protected way. A batch with no rows still has its (possibly
    advanced) cursor/watermark bookkeeping applied and persisted, just
    without the plan-persist/replay dance a row-logging batch needs.

    Args:
        run: The live W&B run.
        run_dir: The run's root directory.
        state: This tool's sync-state, mutated in place.
        finalize: Whether to also finalize each process's still-open
            trailing flush group and release every buffered actor group
            regardless of the cross-process watermark -- ``True`` only under
            the CLI's ``--finalize`` flag (module docstring); ``False`` for
            every other pass, one-shot or ``--follow``.

    Returns:
        Whether anything new was logged.
    """
    changed = False
    if state.pending_plan is not None:
        _run_plan(run, run_dir, state, state.pending_plan)
        changed = True

    plan = _compute_batch(run_dir, state, finalize=finalize)
    if plan["rows"]:
        _run_plan(run, run_dir, state, plan)
        changed = True
    else:
        _apply_plan(state, plan)
        save_sync_state(run_dir, state)

    return changed


# ==============================================================================
# CLI
# ==============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="wandb_sync.py",
        description="Mirror a run directory's metrics into Weights & Biases (opt-in, read-only).",
    )
    parser.add_argument("run_dir", help="the run directory to sync (core.run_identity.run_root)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity/team (default: your default)")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="tail the run directory, syncing new records until interrupted (Ctrl-C)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f"seconds between polls in --follow mode (default: {DEFAULT_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "the run directory is known to be complete (its writer process has exited): "
            "flush every trailing partial flush group and every buffered actor group, "
            "regardless of the cross-process watermark. Omit for a run that might still be "
            "written to -- the default holds trailing/buffered groups back for a later sync."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """The CLI entrypoint.

    Args:
        argv: Argument vector, excluding the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The process exit code (always ``0``; a missing ``wandb`` install or a
        malformed run dir raises instead of exiting nonzero, matching this
        script's other exceptions propagating as a traceback).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    wandb = _require_wandb()
    run_dir = Path(args.run_dir)
    run = _init_wandb_run(wandb, run_dir, project=args.project, entity=args.entity)
    state = load_sync_state(run_dir)
    try:
        if args.follow:
            try:
                while True:
                    sync_once(run, run_dir, state, finalize=False)
                    save_sync_state(run_dir, state)
                    time.sleep(args.poll_interval)
            except KeyboardInterrupt:
                pass
        sync_once(run, run_dir, state, finalize=args.finalize)
        save_sync_state(run_dir, state)
    finally:
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
