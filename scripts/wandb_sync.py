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

**Idempotency.** The W&B run id is derived from the run's own identity
(``core.run_identity.RunRecord.run_id``), so re-syncing resumes the same W&B
run rather than creating a duplicate. Within that run, this script never
re-logs a record it has already sent: a per-process cursor (how many of that
process's ``core.metrics`` records have already been consumed) and the set of
already-synced ``checkpoint_published`` versions are persisted to the sync
sidecar after every successful pass, so a second run of this script against
an unchanged run dir logs nothing new.

**Custom x-axes, not wandb's implicit step.** Every metric group is wired
through ``wandb.define_metric`` to its own step field rather than sharing
wandb's single monotonic step counter: the learner's loss/replay-ratio gauges
key on ``learner/learner_step``, actor throughput keys on
``actor/wall_clock_s`` (with ``actor/positions_evaluated`` also logged, so
either can be picked as the chart x-axis), and checkpoint markers key on
``checkpoint/learner_step``.

**Honest granularity.** Actor deltas land at between-game flush boundaries
(``core.actor.ActorDriver._flush_game_metrics``), the same convention
``core.observability.reduce_run`` documents: the ``positions_evaluated``
figure attached to a ``checkpoint_published`` marker (via
``ReducedRun.checkpoints``, reused here rather than re-derived) is exact only
up to one actor flush period.

Usage::

    python3 scripts/wandb_sync.py runs/blokus_duo/<run-id> --project alpha-games
    python3 scripts/wandb_sync.py runs/blokus_duo/<run-id> --project alpha-games --follow
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
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
            ``sims_run``), used to log ``actor/*`` as a running total rather
            than a per-event increment.
    """

    proc_cursors: dict[str, int] = field(default_factory=dict)
    checkpoint_versions_synced: list[int] = field(default_factory=list)
    actor_totals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this state as a plain, JSON-serializable dict."""
        return {
            "proc_cursors": dict(self.proc_cursors),
            "checkpoint_versions_synced": list(self.checkpoint_versions_synced),
            "actor_totals": dict(self.actor_totals),
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


def _split_learner_records(
    records: list[dict[str, Any]], *, finalize: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Split learner records into finalized flush groups, checkpoints, and a consumed count.

    One group per ``core.learner.LearnerDriver._flush_step_metrics`` call: a
    ``KIND_TOTAL`` ``learner_step`` record followed immediately by that
    step's ``KIND_GAUGE`` records (loss components, replay ratio) -- always
    written contiguously, in that order, by a single-writer process
    (``core.learner``'s own call sequence: ``_flush_step_metrics`` then
    ``_maybe_publish``). A group is "finalized" the moment any later record
    -- a new total, a ``checkpoint_published`` marker, or ``finalize=True``
    at the end of input -- proves the learner is done writing it; an
    unfinalized trailing group (still possibly receiving more gauges) is
    held back entirely, along with its own records, out of ``consumed``.

    Args:
        records: Records already sliced to start at the caller's cursor.
        finalize: Whether to also finalize a still-open trailing group (the
            single-shot backfill's last pass, and ``--follow``'s pass on
            exit; ``False`` for every other ``--follow`` tick).

    Returns:
        ``(finalized_groups, checkpoint_records, records_consumed)`` --
        ``records_consumed`` is how many leading records of ``records`` are
        safe to advance a cursor past.
    """
    groups: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_start = 0
    consumed = 0

    for idx, rec in enumerate(records):
        kind = rec.get("kind")
        if kind == CHECKPOINT_PUBLISHED_KIND:
            if current is not None:
                current["_closed"] = True
            checkpoints.append(rec)
            consumed = idx + 1
            continue
        if kind == KIND_TOTAL and rec.get("series") == SERIES_LEARNER_STEP:
            if current is not None:
                current["_closed"] = True
            current = {
                "learner_step": rec["value"],
                "timestamp": rec["timestamp"],
                "gauges": {},
                "_closed": False,
            }
            current_start = idx
            groups.append(current)
            consumed = idx + 1
        elif kind == KIND_GAUGE and current is not None:
            current["gauges"][rec["series"]] = rec["value"]
            consumed = idx + 1

    if current is not None and finalize:
        current["_closed"] = True
    if current is not None and not current["_closed"]:
        consumed = current_start

    finalized = [g for g in groups if g["_closed"]]
    for g in finalized:
        del g["_closed"]
    return finalized, checkpoints, consumed


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
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_start = 0
    consumed = 0

    for idx, rec in enumerate(records):
        if rec.get("kind") != KIND_DELTA:
            consumed = idx + 1
            continue
        series = rec.get("series")
        if series == SERIES_GAMES_COMPLETED:
            if current is not None:
                current["_closed"] = True
            current = {"timestamp": rec["timestamp"], "deltas": {}, "_closed": False}
            current_start = idx
            groups.append(current)
        if current is not None:
            current["deltas"][series] = rec["value"]
        consumed = idx + 1

    if current is not None and finalize:
        current["_closed"] = True
    if current is not None and not current["_closed"]:
        consumed = current_start

    finalized = [g for g in groups if g["_closed"]]
    for g in finalized:
        del g["_closed"]
    return finalized, consumed


def _actor_procs(run_dir: Path | str) -> list[str]:
    """Return every actor process name under ``run_dir``, sorted for determinism.

    Args:
        run_dir: The run's root directory.

    Returns:
        Every ``core.metrics.list_procs`` entry except ``learner``/``orchestrator``.
    """
    return sorted(p for p in list_procs(run_dir) if p not in _NON_ACTOR_PROCS)


# ==============================================================================
# W&B run construction: identity, config, tags, summary seed
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

    run.define_metric("learner/learner_step")
    run.define_metric("learner/*", step_metric="learner/learner_step")
    run.define_metric("actor/wall_clock_s")
    run.define_metric("actor/*", step_metric="actor/wall_clock_s")
    run.define_metric("checkpoint/learner_step")
    run.define_metric("checkpoint/*", step_metric="checkpoint/learner_step")
    return run


# ==============================================================================
# The sync passes
# ==============================================================================


def _sync_learner(run: Any, run_dir: Path | str, state: SyncState, *, finalize: bool) -> bool:
    """Replay new learner flush groups into ``learner/*`` and sync any new checkpoints.

    Args:
        run: The live W&B run.
        run_dir: The run's root directory.
        state: Mutated in place: the learner cursor and (via
            :func:`_sync_checkpoints`) checkpoint bookkeeping.
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
        run.log(payload)
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
    note is exactly ``reduce_run``'s own documented bound).

    Args:
        run: The live W&B run.
        run_dir: The run's root directory.
        state: Mutated in place: ``checkpoint_versions_synced``.
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
        run.log(
            {
                "checkpoint/model_version": version,
                "checkpoint/learner_step": learner_step,
                "checkpoint/positions_evaluated": positions_evaluated,
                "checkpoint/gpu_hours": gpu_hours,
            }
        )
        run.summary[f"checkpoint_{version}"] = {
            "learner_step": learner_step,
            "positions_evaluated": positions_evaluated,
            "gpu_hours": gpu_hours,
        }
        state.checkpoint_versions_synced.append(version)
    return True


def _sync_actors(run: Any, run_dir: Path | str, state: SyncState, *, finalize: bool) -> bool:
    """Replay new actor flush groups, merged in timestamp order, as a running cumulative.

    Every actor process's new groups are merged into one chronological
    stream (tie-broken by process name) so ``actor/*`` reads as one coherent
    throughput curve across every actor, rather than one series per process.

    Args:
        run: The live W&B run.
        run_dir: The run's root directory.
        state: Mutated in place: per-actor cursors and the running
            ``actor_totals`` cumulative sums.
        finalize: Forwarded to :func:`_split_actor_records`.

    Returns:
        Whether anything new was logged.
    """
    record = read_run_record(run_dir)
    run_start_ts = _parse_iso_utc(record.created_at)

    merged: list[tuple[str, dict[str, Any]]] = []
    new_cursors: dict[str, int] = {}
    for proc in _actor_procs(run_dir):
        records = list(iter_epoch_records(run_dir, proc))
        cursor = state.proc_cursors.get(proc, 0)
        groups, consumed = _split_actor_records(records[cursor:], finalize=finalize)
        new_cursors[proc] = cursor + consumed
        merged.extend((proc, group) for group in groups)

    merged.sort(key=lambda item: (item[1]["timestamp"], item[0]))

    for proc, group in merged:
        for series, value in group["deltas"].items():
            state.actor_totals[series] = state.actor_totals.get(series, 0.0) + value
        run.log(
            {
                "actor/wall_clock_s": group["timestamp"] - run_start_ts,
                "actor/proc": proc,
                "actor/games_completed": state.actor_totals.get(SERIES_GAMES_COMPLETED, 0.0),
                "actor/positions_evaluated": state.actor_totals.get(
                    SERIES_POSITIONS_EVALUATED, 0.0
                ),
                "actor/sims_run": state.actor_totals.get(SERIES_SIMS_RUN, 0.0),
            }
        )

    for proc, new_cursor in new_cursors.items():
        state.proc_cursors[proc] = new_cursor
    return bool(merged)


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
            trailing flush group -- ``True`` for a one-shot backfill and for
            ``--follow``'s pass on exit; ``False`` for every other
            ``--follow`` tick.

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
        sync_once(run, run_dir, state, finalize=True)
        save_sync_state(run_dir, state)
    finally:
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
