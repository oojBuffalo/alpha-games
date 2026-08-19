"""Per-process append-only epoch metrics files (design doc §12 M3, tasks/m3/011).

The naming/durability primitive every M3 process appends observability
records through -- **not** the full observability contract. Issue #62
(``tasks/m3/011-observability.md``) owns ``reduce_run``, the field-kind
taxonomy (``delta``/``gauge``/``total``) enforcement, and the
positions-evaluated counting wrapper; this module implements only the
generic writer/reader primitive that contract is built on top of, so #62 can
adopt it unchanged rather than re-deriving the file-naming convention.

**One writer per file, explicit reduction -- never a shared append target.**
Each process appends only its own ``<run_dir>/metrics/<proc>-<epoch>.jsonl``
(``proc`` e.g. ``"learner"``, ``"actor-<id>"``, ``"orchestrator"``; ``epoch``
increments on every (re)start of that process). Concurrent appends to one
shared file can interleave mid-write, and a restarted process resetting
cumulative counters under the same id would silently corrupt any sum a
reader computes over it -- per-process files plus epochs eliminate both by
construction: :class:`EpochMetricsWriter` picks the next unused epoch for
``proc`` by scanning ``run_dir/metrics/`` at construction time, so a fresh
instance (a process restart, by definition) always opens a brand-new file
rather than reopening a stale one.

**Records are flat JSON, one per line, durable before the call returns.**
:meth:`EpochMetricsWriter.append` opens the file in append mode, writes one
``json.dumps`` line, and ``flush``\\ es + ``fsync``\\ s before returning --
not the temp-name-then-``os.replace`` whole-file pattern
``core.replay_shard``/``core.checkpoint`` use for *replaceable* artifacts,
because an append-only log is never replaced, only grown; the durability
property this module needs is "once ``append`` returns, the record survives
a crash," which a synced append gives directly. This is also what makes
"exactly one marker per version" provable across a crash at any point: a
caller that checks :func:`iter_epoch_records` for an already-appended record
before appending again (:mod:`core.learner`'s publish path) can never
duplicate one, because a record that was ever durably appended is always
visible to that check afterward, restart or not.

Every documented record carries a ``kind`` field naming what it is (e.g.
``"checkpoint_published"`` -- :mod:`core.learner`); a numeric metrics series
(games played, sims run, loss components, ...) is additionally one of
``delta`` (increment since the writer's previous flush), ``gauge`` (a
latest-value observation), or ``total`` (a coordinator-owned exact
cumulative) per the design-doc taxonomy -- issue #62 defines and enforces
that taxonomy for its own series; this module does not interpret record
contents at all, only their durable storage and retrieval.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

_METRICS_DIRNAME = "metrics"
_EPOCH_PATTERN = re.compile(r"^(?P<proc>.+)-(?P<epoch>\d+)\.jsonl$")


def metrics_dir(run_dir: Path | str) -> Path:
    """Return the metrics directory for one run.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``run_dir / "metrics"``.
    """
    return Path(run_dir) / _METRICS_DIRNAME


def epoch_metrics_path(run_dir: Path | str, proc: str, epoch: int) -> Path:
    """Return one process epoch's file path.

    Args:
        run_dir: The run's root directory.
        proc: The process name, e.g. ``"learner"``.
        epoch: The epoch number (increments per process (re)start).

    Returns:
        ``run_dir / "metrics" / "<proc>-<epoch>.jsonl"``.
    """
    return metrics_dir(run_dir) / f"{proc}-{epoch}.jsonl"


def _existing_epochs(run_dir: Path | str, proc: str) -> list[int]:
    """Return every epoch number already on disk for ``proc``, unsorted.

    Args:
        run_dir: The run's root directory.
        proc: The process name.

    Returns:
        Epoch numbers found among ``<proc>-<digits>.jsonl`` filenames under
        ``run_dir/metrics/`` (empty if the directory does not exist yet, or
        no file matches).
    """
    directory = metrics_dir(run_dir)
    if not directory.exists():
        return []
    epochs = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        m = _EPOCH_PATTERN.match(p.name)
        if m is not None and m.group("proc") == proc:
            epochs.append(int(m.group("epoch")))
    return epochs


def next_epoch(run_dir: Path | str, proc: str) -> int:
    """Return the next unused epoch number for ``proc`` under ``run_dir``.

    Args:
        run_dir: The run's root directory.
        proc: The process name.

    Returns:
        ``0`` for a process that has never written under this ``run_dir``;
        otherwise one more than the highest epoch already on disk.
    """
    existing = _existing_epochs(run_dir, proc)
    return max(existing, default=-1) + 1


def iter_epoch_records(run_dir: Path | str, proc: str) -> Iterator[dict[str, Any]]:
    """Yield every record ``proc`` has ever durably appended, across all epochs.

    Reads every ``<proc>-<epoch>.jsonl`` file under ``run_dir/metrics/`` in
    ascending epoch order, then in on-disk (append) order within each file --
    the full durable history of one process's records, spanning every
    restart. Used both by a resuming writer to check "has this already been
    recorded" before appending (avoiding a duplicate across a crash) and,
    eventually, by issue #62's reducer.

    Args:
        run_dir: The run's root directory.
        proc: The process name.

    Yields:
        Each record, parsed from JSON, oldest first.
    """
    directory = metrics_dir(run_dir)
    if not directory.exists():
        return
    epoch_files = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        m = _EPOCH_PATTERN.match(p.name)
        if m is not None and m.group("proc") == proc:
            epoch_files.append((int(m.group("epoch")), p))
    for _, path in sorted(epoch_files):
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


class EpochMetricsWriter:
    """Append-only writer for one process's current-epoch metrics file.

    Args:
        run_dir: The run's root directory. Created if missing.
        proc: This process's name (e.g. ``"learner"``); constant for the
            life of one writer instance.

    Attributes:
        epoch: This instance's epoch number (:func:`next_epoch` at
            construction time) -- a fresh instance is, by definition, a
            process (re)start, so it always claims a new epoch rather than
            reopening a prior one.
        path: This instance's epoch file path.
    """

    def __init__(self, run_dir: Path | str, proc: str) -> None:
        self.run_dir = Path(run_dir)
        self.proc = proc
        self.epoch = next_epoch(self.run_dir, proc)
        self.path = epoch_metrics_path(self.run_dir, proc, self.epoch)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file up front, empty, so a directory scan sees this
        # epoch exists even before the first record is appended.
        self.path.touch(exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one record as a JSON line, durable before returning.

        Args:
            record: A flat, JSON-serializable mapping. Copied, not mutated.
        """
        line = json.dumps(dict(record), sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
