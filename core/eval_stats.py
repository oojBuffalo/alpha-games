"""Anchored full-ladder Elo fit over an eval snapshot, and the §1 x-axis join
(design doc §1, §9; tasks/m4/006).

Two pieces, deliberately factored so task 7 (the bootstrap/inference half this
task leaves room for) reuses both without a second fit implementation:

* **The fit.** :func:`snapshot_matches` aggregates every cell within an
  :class:`~core.eval_store.EvalSnapshot`'s *complete contiguous member
  prefix* -- never a hole-adjacent member's partial evidence, see
  ``EvalSnapshot.member_prefix``'s own docstring -- into ``core.elo.Match``
  aggregates via ``core.eval_store.records_to_match``; :func:`fit_snapshot_elo`
  feeds the whole set through the unmodified, virtual-draw-regularized
  ``core.elo.fit_elo``, anchored at :data:`ANCHOR_AGENT` (the M1.6 frozen
  ladder's rung-1 identity, ``core.agents.RandomAgent.name``). A missing or
  unplayed cell breaks the matchup graph's connectivity to the anchor, which
  surfaces here exactly as ``core.elo.fit_elo``'s own named-agent
  ``ValueError`` -- this module rebuilds no connectivity check of its own.
  Task 7's per-replicate warm-started fits call ``core.elo.fit_elo`` directly
  with ``initial_ratings`` over their own resampled match lists, reusing
  :func:`snapshot_matches`'s output as the population to resample from.
* **The join.** :func:`checkpoint_elo` extracts the rung-7 form's rating per
  member version, ordered by ``model_version`` (provenance order, never file
  mtime); :func:`elo_curve` pairs each with
  ``core.observability.reduce_run``'s per-version ``(learner_step,
  positions_evaluated, gpu_hours)`` x-axis coordinates -- the frozen
  ``checkpoint_published``-marker join -- and durably writes the design doc
  §1 plot series to ``<run_dir>/eval/elo_curve.json``
  (temp-name-then-``os.replace``, mirroring every other replaceable artifact
  in this codebase). Each row's ``net_evals`` is exact only up to one actor
  flush period -- ``reduce_run``'s own documented, honest granularity bound
  (the cumulative positions-evaluated sum as of the publish marker's position
  in run time order, never interpolated between flushes) -- restated here,
  not re-derived. matplotlib is deliberately not a dependency of this
  codebase; the provisional-vs-authoritative distinction over this series is
  task 7's concern, not this module's.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from core.elo import Match, fit_elo
from core.eval_store import EvalSnapshot, eval_dir, iter_cells, read_cell, records_to_match
from core.observability import reduce_run

#: The M1.6 frozen ladder's rung-1 anchor identity -- ``core.agents.RandomAgent.name``
#: verbatim (a literal mirror, like ``core.eval_protocol.EVAL_SIMS``'s cross-module
#: golden pattern: mirrored as one constant here rather than imported, so this module
#: does not need to reach into ``core.agents`` just to read a single string --
#: ``tests/test_eval_stats.py`` golds this literal against the live agent name).
ANCHOR_AGENT = "random"

#: A rung-7 (full policy-and-value MCTS) agent identity string, e.g. ``"rung7-v1-12"``
#: -- ``core.eval_agents.SearchAgent``'s ``f"rung{form}-v1-{model_version}"`` naming
#: convention for ``form == 7``, matched here as a plain string pattern rather than by
#: importing that module (which pulls in the checkpoint-loading machinery this module
#: has no other reason to depend on).
_RUNG7_IDENTITY = re.compile(r"^rung7-v1-(\d+)$")

_ELO_CURVE_NAME = "elo_curve.json"


def elo_curve_path(run_dir: Path | str) -> Path:
    """Return the §1 plot series' on-disk path for one run.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``<run_dir>/eval/elo_curve.json``.
    """
    return eval_dir(run_dir) / _ELO_CURVE_NAME


def snapshot_matches(snapshot: EvalSnapshot) -> list[Match]:
    """Aggregate an eval snapshot's in-scope cells into ``core.elo.Match`` aggregates.

    Reads every cell the snapshot marks complete (``core.eval_store.iter_cells``)
    but keeps only those belonging to the snapshot's *complete contiguous member
    prefix* (``snapshot.member_prefix``): a completed cell whose candidate version
    sits beyond that prefix is real evidence for a not-yet-fully-scored member
    (``EvalSnapshot.completed_cell_ids``'s own docstring notes such cells are
    visible there -- "per-checkpoint live reporting reads this set directly" --
    precisely because they sit *outside* the contiguous prefix), so it is excluded
    here rather than silently admitted into the §1 point estimate -- the "never
    partial data" analysis-snapshot convention this task's fit is pinned to. A
    cell that is merely scheduled (never completed at all) is already structurally
    absent from the snapshot and never reaches this function in the first place.

    Args:
        snapshot: A frozen snapshot from ``core.eval_store.load_snapshot``.

    Returns:
        One ``Match`` per in-scope cell, in sorted cell-id order. Order is
        immaterial to the fit itself (``core.elo.fit_elo`` collapses duplicate
        matchups and sums are order-independent) but kept deterministic for any
        caller inspecting this list directly, task 7's bootstrap resampling
        included.
    """
    matches: list[Match] = []
    for path in iter_cells(snapshot):
        header, records = read_cell(path)
        if header.cell_id.candidate_version > snapshot.member_prefix:
            continue
        matches.append(
            records_to_match(header.candidate_identity, header.opponent_identity, records)
        )
    return matches


def fit_snapshot_elo(
    snapshot: EvalSnapshot, initial_ratings: dict[str, float] | None = None
) -> dict[str, float]:
    """Fit the anchored full-ladder Elo point estimate over one eval snapshot.

    Args:
        snapshot: A frozen snapshot from ``core.eval_store.load_snapshot``.
        initial_ratings: Forwarded verbatim to ``core.elo.fit_elo``. ``None``
            (the default) is the cold zero-start this task's own point estimate
            uses; a caller (task 7's per-replicate warm start) may supply a
            dict of prior ratings instead -- see ``core.elo.fit_elo``'s
            docstring for the exact semantics.

    Returns:
        Every agent's rating reachable from :data:`ANCHOR_AGENT` in
        ``snapshot`` -- network-free ladder rungs, checkpoint forms, and
        historical opponents alike -- with ``result[ANCHOR_AGENT] == 0.0``.

    Raises:
        ValueError: If :func:`snapshot_matches` yields no matchup at all, or a
            matchup graph in which some scored agent cannot reach
            :data:`ANCHOR_AGENT` -- ``core.elo.fit_elo``'s own named-agent
            connectivity error (e.g. a required cell that was never played
            leaves its candidate or opponent identity unreachable).
    """
    return fit_elo(snapshot_matches(snapshot), anchor=ANCHOR_AGENT, initial_ratings=initial_ratings)


def checkpoint_elo(ratings: dict[str, float]) -> list[tuple[int, float]]:
    """Extract the rung-7 checkpoint curve from a fitted ratings dict.

    Args:
        ratings: A ``core.elo.fit_elo`` (or :func:`fit_snapshot_elo`) result.

    Returns:
        ``(model_version, elo)`` pairs for every rung-7 (``"rung7-v1-<v>"``)
        agent present in ``ratings``, ordered by ``model_version`` ascending --
        the §6.2 provenance ordering, never file mtime and never dict/insertion
        order.
    """
    versions: list[tuple[int, float]] = []
    for name, elo in ratings.items():
        match = _RUNG7_IDENTITY.match(name)
        if match is not None:
            versions.append((int(match.group(1)), elo))
    versions.sort(key=lambda pair: pair[0])
    return versions


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON to ``path``, durable and atomic.

    Temp-name-then-``os.replace`` -- the same primitive
    ``core.eval_store._atomic_write_manifest`` and ``core.checkpoint`` use for
    their own replaceable artifacts: a reader can never observe a partially
    written file.

    Args:
        path: Destination file path; its parent directory is created if
            missing.
        payload: A JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def elo_curve(run_dir: Path | str, snapshot: EvalSnapshot) -> dict[str, Any]:
    """Fit and join the design doc §1 plot series, writing it durably.

    One anchored Bradley-Terry fit (:func:`fit_snapshot_elo`) over
    ``snapshot``, joined member-by-member against
    ``core.observability.reduce_run(run_dir)``'s frozen ``checkpoints``
    contract -- the ``checkpoint_published``-marker x-axis coordinates
    (``learner_step``, cumulative ``positions_evaluated``, single-counted
    ``gpu_hours``) at each member's publish point in run time order. Every
    row's ``net_evals`` is therefore exact only up to one actor flush period:
    the cumulative positions-evaluated sum as of the publish marker's own
    position in the global run-time ordering, never interpolated between
    flushes -- ``reduce_run``'s own documented bound, restated here because
    this is where a plot consumer reads it, not re-derived.

    Writes ``<run_dir>/eval/elo_curve.json`` (:func:`_atomic_write_json`, so a
    reader never observes a partially written file) and returns the identical
    payload. matplotlib is deliberately not a dependency of this codebase --
    a plotting consumer reads this file directly. Distinguishing this
    (necessarily provisional, mid-run) series from an authoritative final one
    is task 7's concern, not this function's.

    Args:
        run_dir: The run's root directory -- both ``snapshot``'s own root (an
            eval-store snapshot is always read from one run) and the root
            ``core.observability.reduce_run`` aggregates metrics under.
        snapshot: A frozen snapshot from ``core.eval_store.load_snapshot``,
            covering this run.

    Returns:
        ``{"snapshot_fingerprint": str, "rows": [{"model_version", "elo",
        "learner_step", "net_evals", "gpu_hours"}, ...]}`` -- rows ordered by
        ``model_version`` ascending, covering exactly the snapshot's in-scope
        members (:func:`snapshot_matches`'s member-prefix scope).

    Raises:
        ValueError: If :func:`fit_snapshot_elo` raises (a disconnected
            agent), or if some member version :func:`checkpoint_elo` returns
            has no matching entry in ``reduce_run(run_dir).checkpoints`` --
            an eval-store/observability inconsistency (e.g. a candidate
            scored in the eval store whose learner never wrote a
            ``checkpoint_published`` marker for it) this function refuses to
            paper over.
    """
    ratings = fit_snapshot_elo(snapshot)
    reduced = reduce_run(run_dir)

    rows: list[dict[str, Any]] = []
    for model_version, elo in checkpoint_elo(ratings):
        coords = reduced.checkpoints.get(model_version)
        if coords is None:
            raise ValueError(
                f"member version {model_version} is scored in the eval snapshot but has no "
                f"checkpoint_published marker in {run_dir!r}'s reduced metrics -- "
                "eval-store/observability inconsistency"
            )
        learner_step, positions_evaluated, gpu_hours = coords
        rows.append(
            {
                "model_version": model_version,
                "elo": elo,
                "learner_step": learner_step,
                "net_evals": positions_evaluated,
                "gpu_hours": gpu_hours,
            }
        )

    payload = {"snapshot_fingerprint": snapshot.snapshot_fingerprint, "rows": rows}
    _atomic_write_json(elo_curve_path(run_dir), payload)
    return payload
