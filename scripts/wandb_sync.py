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

**Idempotency: the precise mechanism and its guarantee.** The W&B run id is
derived from the run's own identity (``core.run_identity.RunRecord.run_id``),
so re-syncing resumes the same W&B run rather than creating a duplicate
(``resume="allow"``). Within that run, every :func:`~wandb.Run.log` call this
script makes carries an explicit, deterministic ``step`` -- this tool's own
monotonically increasing count of every row it has ever logged for this run
(:attr:`SyncState.next_step`), assigned in the fixed order one full
:func:`sync_once` pass visits records (learner flush groups, then
checkpoints, then released actor flush groups -- see below), and ``commit=True``
so each row lands as its own complete history entry rather than accumulating
into a later one. On ``wandb.init(id=..., resume="allow")``, W&B's client
seeds its next-step counter one past the highest step the server has already
recorded for that run id (verified against the ``wandb`` 0.28 SDK source,
``sdk/internal/sender.py``'s ``_resume_state.step = last_step + 1`` and
``sdk/internal/handler.py``'s ``handle_request_partial_history``, which drops
-- client-side, before ever forwarding the row -- any subsequently logged row
whose ``step`` is less than that counter, emitting a local warning rather
than an error). Since :attr:`SyncState.next_step` is only ever persisted
together with the cursors/buffers of the rows it was assigned to (the sidecar
is one JSON document, saved as a unit -- see :func:`save_sync_state`), a
crash between an accepted ``run.log`` and the next sidecar save reprocesses
those same not-yet-cursor-advanced records from an unchanged
``next_step`` baseline on the next invocation, reassigning them the exact
same step values -- which W&B then drops as already-seen. This makes replay
a true no-op, not merely a bounded duplicate window: **the guarantee is "a
row is logged to W&B history at most once," contingent on the assumptions
above (one ``wandb`` run id per run dir, nobody else logging to that run id,
and this script never logging a real -- not replayed -- row out of the fixed
per-pass order this docstring names).** Follow mode persists the sidecar
after every poll (not only at process exit), bounding how much a genuine
crash mid-poll ever has to replay to one poll's worth of records.

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

**Multi-actor throughput is buffered and released behind a global
watermark.** ``actor/*`` and ``throughput/*`` are meant to read as one
coherent timeline across every actor, but each actor's own file only proves
its *own* chronology -- actor 0 having reached t=40 says nothing about
whether actor 1's next group lands at t=20 or t=41. Every actor's newly
finalized groups are therefore buffered per-process
(:attr:`SyncState.actor_buffer`) rather than logged immediately; a group is
released only once the *global watermark* -- the minimum, across every actor
process this run has ever had, of that process's own highest observed group
timestamp (:attr:`SyncState.actor_watermarks`) -- has reached or passed it.
This relies on one assumption verified against the writer
(``core.actor.ActorDriver._flush_game_metrics`` stamps one ``time.time()``
per flush, strictly append-ordered within a single-writer process): a
process's own records are non-decreasing in timestamp, so "this process has
shown groups up to T" really does promise "no future group from it will be
earlier than T." Released groups are logged in global sorted
``(timestamp, proc)`` order with cumulative totals computed at release time,
in that order -- never the order they happened to arrive in across polls.
**Known stall behavior:** one actor that has gone quiet (crashed, or simply
slower than its siblings) holds back every other actor's throughput data at
its own last-seen timestamp indefinitely -- the data is buffered, not lost,
and ``--finalize`` (once the run is known complete) flushes every buffered
group regardless of watermark. A brand-new actor process whose metrics file
exists but has not yet completed its first flush contributes no watermark
at all and blocks release the same way, for the same reason: this script
cannot distinguish "about to report" from "already dead" without
``--finalize``.

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
            docstring). Each entry is ``{"timestamp", "deltas"}``, the same
            shape :func:`_split_actor_records` returns.
        actor_watermarks: Per-actor-process high-water mark: the highest
            group timestamp ever observed from that process, whether or not
            that group has been released yet. Monotonically non-decreasing
            per process (module docstring's within-file timestamp
            monotonicity assumption); the global release watermark is the
            minimum of these across every actor process this run has ever
            had.
        next_step: The next explicit W&B ``step`` this script will assign --
            a running count of every row it has ever logged for this run,
            in the fixed per-pass order the module docstring names. The
            idempotency mechanism this script relies on (module docstring).
    """

    proc_cursors: dict[str, int] = field(default_factory=dict)
    checkpoint_versions_synced: list[int] = field(default_factory=list)
    actor_totals: dict[str, float] = field(default_factory=dict)
    actor_buffer: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    actor_watermarks: dict[str, float] = field(default_factory=dict)
    next_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return this state as a plain, JSON-serializable dict."""
        return {
            "proc_cursors": dict(self.proc_cursors),
            "checkpoint_versions_synced": list(self.checkpoint_versions_synced),
            "actor_totals": dict(self.actor_totals),
            "actor_buffer": {proc: list(groups) for proc, groups in self.actor_buffer.items()},
            "actor_watermarks": dict(self.actor_watermarks),
            "next_step": self.next_step,
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
    metric names logging the same numbers (:func:`_sync_actors`,
    :func:`_sync_checkpoints`) -- the fresh ``throughput/positions_evaluated``
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
# The sync passes
# ==============================================================================


def _sync_learner(run: Any, run_dir: Path | str, state: SyncState, *, finalize: bool) -> bool:
    """Replay new learner flush groups into ``learner/*`` and sync any new checkpoints.

    Args:
        run: The live W&B run.
        run_dir: The run's root directory.
        state: Mutated in place: the learner cursor, :attr:`SyncState.next_step`,
            and (via :func:`_sync_checkpoints`) checkpoint bookkeeping.
        finalize: Forwarded to :func:`_split_learner_records`.

    Returns:
        Whether anything new was logged.
    """
    records = list(iter_epoch_records(run_dir, LEARNER_PROC))
    cursor = state.proc_cursors.get(LEARNER_PROC, 0)
    groups, checkpoints, consumed = _split_learner_records(records[cursor:], finalize=finalize)

    for group in groups:
        payload = {"learner/learner_step": group["learner_step"]}
        for series, value in group["gauges"].items():
            payload[f"learner/{series}"] = value
        step = state.next_step
        state.next_step += 1
        run.log(payload, step=step, commit=True)
    state.proc_cursors[LEARNER_PROC] = cursor + consumed

    checkpoints_changed = _sync_checkpoints(run, run_dir, state, checkpoints)
    return bool(groups) or checkpoints_changed


def _sync_checkpoints(
    run: Any, run_dir: Path | str, state: SyncState, new_checkpoints: list[dict[str, Any]]
) -> bool:
    """Log every not-yet-synced ``checkpoint_published`` marker as a summary point.

    Sourced from ``core.observability.reduce_run(run_dir).checkpoints`` --
    reusing the canonical ``(learner_step, positions_evaluated, gpu_hours)``
    join rather than re-deriving it (module docstring's "honest granularity"
    note is exactly ``reduce_run``'s own documented bound). Logs the marker's
    ``model_version`` a second and third time under
    ``checkpoint/marker_vs_positions``/``checkpoint/marker_vs_gpu_hours`` so
    the positions/GPU-hours axes (:func:`_define_metrics`) carry data too.

    Args:
        run: The live W&B run.
        run_dir: The run's root directory.
        state: Mutated in place: ``checkpoint_versions_synced`` and
            :attr:`SyncState.next_step`.
        new_checkpoints: Newly observed ``checkpoint_published`` records
            (:func:`_split_learner_records`'s second return value).

    Returns:
        Whether any new checkpoint was logged.
    """
    pending = sorted(
        {
            rec["model_version"]
            for rec in new_checkpoints
            if rec["model_version"] not in state.checkpoint_versions_synced
        }
    )
    if not pending:
        return False

    reduced = reduce_run(run_dir)
    for version in pending:
        if version not in reduced.checkpoints:
            continue  # not yet visible to reduce_run's own scan; retried next sync
        learner_step, positions_evaluated, gpu_hours = reduced.checkpoints[version]
        step = state.next_step
        state.next_step += 1
        run.log(
            {
                "checkpoint/model_version": version,
                "checkpoint/learner_step": learner_step,
                "checkpoint/positions_evaluated": positions_evaluated,
                "checkpoint/gpu_hours": gpu_hours,
                "checkpoint/positions_evaluated_axis": positions_evaluated,
                "checkpoint/gpu_hours_axis": gpu_hours,
                "checkpoint/marker_vs_positions": version,
                "checkpoint/marker_vs_gpu_hours": version,
            },
            step=step,
            commit=True,
        )
        run.summary[f"checkpoint_{version}"] = {
            "learner_step": learner_step,
            "positions_evaluated": positions_evaluated,
            "gpu_hours": gpu_hours,
        }
        state.checkpoint_versions_synced.append(version)
    return True


def _actor_watermark(state: SyncState, procs: list[str]) -> float:
    """Return the global cross-actor release watermark (module docstring).

    The minimum, across every currently known actor process, of that
    process's own highest observed group timestamp -- a process this run has
    never seen a group from yet contributes ``-inf`` (blocks release
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


def _sync_actors(run: Any, run_dir: Path | str, state: SyncState, *, finalize: bool) -> bool:
    """Buffer new actor flush groups, then release them behind the global watermark.

    Every actor process's newly finalized groups are added to
    :attr:`SyncState.actor_buffer` and that process's
    :attr:`SyncState.actor_watermarks` entry is raised to its latest group's
    timestamp; only groups at or below the current global watermark
    (:func:`_actor_watermark`) -- or, under ``finalize=True``, every buffered
    group regardless -- are then released: merged across every process in
    global ``(timestamp, proc)`` order and logged as one coherent cumulative
    throughput curve (module docstring).

    Args:
        run: The live W&B run.
        run_dir: The run's root directory.
        state: Mutated in place: per-actor cursors, the buffer, watermarks,
            the running ``actor_totals`` cumulative sums, and
            :attr:`SyncState.next_step`.
        finalize: Forwarded to :func:`_split_actor_records`; also bypasses
            the watermark gate entirely for release (module docstring).

    Returns:
        Whether anything new was logged.
    """
    record = read_run_record(run_dir)
    run_start_ts = _parse_iso_utc(record.created_at)
    procs = _actor_procs(run_dir)

    for proc in procs:
        records = list(iter_epoch_records(run_dir, proc))
        cursor = state.proc_cursors.get(proc, 0)
        groups, consumed = _split_actor_records(records[cursor:], finalize=finalize)
        state.proc_cursors[proc] = cursor + consumed
        if groups:
            state.actor_buffer.setdefault(proc, []).extend(groups)
            state.actor_watermarks[proc] = max(
                state.actor_watermarks.get(proc, float("-inf")), groups[-1]["timestamp"]
            )

    watermark = float("inf") if finalize else _actor_watermark(state, procs)

    released: list[tuple[str, dict[str, Any]]] = []
    for proc in list(state.actor_buffer):
        buffered = state.actor_buffer[proc]
        ready = [g for g in buffered if g["timestamp"] <= watermark]
        if not ready:
            continue
        held_back = [g for g in buffered if g["timestamp"] > watermark]
        released.extend((proc, g) for g in ready)
        if held_back:
            state.actor_buffer[proc] = held_back
        else:
            del state.actor_buffer[proc]

    released.sort(key=lambda item: (item[1]["timestamp"], item[0]))

    for proc, group in released:
        for series, value in group["deltas"].items():
            state.actor_totals[series] = state.actor_totals.get(series, 0.0) + value
        games_total = state.actor_totals.get(SERIES_GAMES_COMPLETED, 0.0)
        positions_total = state.actor_totals.get(SERIES_POSITIONS_EVALUATED, 0.0)
        sims_total = state.actor_totals.get(SERIES_SIMS_RUN, 0.0)
        step = state.next_step
        state.next_step += 1
        run.log(
            {
                "actor/wall_clock_s": group["timestamp"] - run_start_ts,
                "actor/proc": proc,
                "actor/games_completed": games_total,
                "actor/positions_evaluated": positions_total,
                "actor/sims_run": sims_total,
                "throughput/positions_evaluated": positions_total,
                "throughput/games_vs_positions": games_total,
                "throughput/sims_vs_positions": sims_total,
            },
            step=step,
            commit=True,
        )

    return bool(released)


def _parse_iso_utc(stamp: str) -> float:
    """Parse a ``core.run_identity.iso_now`` timestamp back to epoch seconds.

    Args:
        stamp: A ``"YYYY-MM-DDTHH:MM:SSZ"`` string.

    Returns:
        Epoch seconds (UTC).
    """
    import calendar

    return float(calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")))


def sync_once(run: Any, run_dir: Path | str, state: SyncState, *, finalize: bool = False) -> bool:
    """Run one full sync pass: new learner flushes, checkpoints, and actor throughput.

    Pure with respect to the run dir (never writes to it); mutates ``state``
    in place and calls ``run.log``/``run.summary`` for anything new. The
    caller owns persisting ``state`` (:func:`save_sync_state`).

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
    learner_changed = _sync_learner(run, run_dir, state, finalize=finalize)
    actor_changed = _sync_actors(run, run_dir, state, finalize=finalize)
    return learner_changed or actor_changed


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
