"""Anchored full-ladder Elo fit over an eval snapshot, the §1 x-axis join, the
within-cell paired-bootstrap resampler, the Delta/CI/Mann-Kendall inference layer
built on top of it, and the profiled-plateau detector built on top of *that*
(design doc §1, §9, §12 M4; tasks/m4/006, 007, 008).

Six pieces, deliberately factored so the bootstrap (task 7.1) reuses the fit
without a second fit implementation, the inference layer (task 7.2) reuses
the resampler without a second resampling implementation, the verdict
assembly (task 7.3) reuses every one of the above without re-deriving any of
them, and the plateau detector (task 8) reuses the inference layer's own
Mann-Kendall and order-statistic-CI machinery rather than a second copy:

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
* **The resample (task 7.1; task 1 pin 7).** :func:`bootstrap_seed` derives the
  run's one recorded bootstrap seed; :func:`bootstrap_replicate_matches`
  resamples every in-scope cell's pair records with replacement to the cell's
  own pair count and re-aggregates them -- the same in-scope cell population
  :func:`snapshot_matches` aggregates over (:func:`_snapshot_cell_records`),
  resampled at the per-pair-record grain within each cell rather than at
  ``Match``'s already-collapsed grain, since resampling *inside* a cell is the
  pinned unit (a resampled ``Match`` still names the same candidate/opponent
  pair, so the matchup graph's connectivity to the anchor never changes across
  replicates -- only each edge's score). :func:`bootstrap_replicate` refits
  that replicate through the unmodified ``core.elo.fit_elo``, warm-started from
  the point estimate via its ``initial_ratings`` parameter (task 6's
  extension) -- a pure speedup to the same fixed point, never a different
  answer. :func:`bootstrap_replicates` is the batch form: one point-estimate
  fit and one cell read, then ``B`` refits by index. Every replicate is
  reproducible from ``(bootstrap_seed, b)`` alone, independent of any other
  replicate ever having run.
* **The inference layer (task 7.2; task 1 pins 7-8).** :func:`delta_hat`
  computes §1's Delta contrast -- ``mean(Elo, final ceil(K/3)) - mean(Elo,
  first ceil(K/3))`` (:func:`delta_windows`) -- over one complete, contiguous
  ``1..K`` rung-7 curve; the identical function computes both the
  original-sample Delta-hat and, via :func:`replicate_deltas` mapping
  :func:`checkpoint_elo` over each of :func:`bootstrap_replicates`'s ratings
  dicts, every replicate's Delta_b. :func:`order_statistic_ci` is the one
  admissible-``B`` order-statistic CI rule (ranks ``(B+1)*0.025``/
  ``(B+1)*0.975``), used unchanged for both Delta's CI (:func:`delta_gate`
  reads its lower endpoint) and :func:`per_checkpoint_ci`'s per-version
  columns -- the same ``B`` replicates, no second resample. :func:`mann_kendall`
  is the pure-stdlib, tie/continuity-corrected trend test, reported but never
  gating.
* **``verdict.json`` assembly (task 7.3; task 1 pin 8; reviews P4/P6/P7).**
  :func:`build_verdict` is the one-shot orchestrator over every piece above:
  it reads the watched run's own stored ``config.json``
  (``core.run_identity.read_stored_config``) for ``k_target``
  (``training.checkpoint_count``) and the eval seed, loads the eval-store
  snapshot, and decides *whether* that snapshot's contiguous member prefix is
  the complete K-set eligible for a Delta at all -- the gate none of the
  functions above make on their own. Before the complete K-set exists, the
  artifact is provisional (``delta: null`` plus an explicit ``reason``, no
  gate anywhere); ``authoritative`` additionally requires ``B ==
  core.eval_protocol.BOOTSTRAP_B_PRODUCTION`` exactly. The artifact carries
  its own evidence fingerprint (the snapshot's :attr:`~core.eval_store.
  EvalSnapshot.snapshot_fingerprint`), a content-hash reference to the
  ``elo_curve.json`` derived artifact it refreshes alongside itself, the
  recorded :func:`bootstrap_seed`, and the full protocol-registry block
  (``core.eval_protocol.REGISTRY`` plus its fingerprint) -- everything a
  later reader needs to know exactly what evidence, seed, and protocol
  version produced this verdict, without re-deriving any of it.
* **The profiled-plateau detector (task 8; design doc §12 M4, §9 pins 2 and 9).**
  :func:`detect_plateau` is a pure, read-only reader over the task-5 snapshot
  and the task-7 ``elo_curve.json`` artifact already on disk -- it never
  writes anything and triggers nothing (M6 lever decisions stay human calls
  that *consume* its tri-state :class:`PlateauResult`, never the other way
  round). Six pinned constants (``core.eval_protocol.PLATEAU_WINDOW_M`` and
  friends) define one predicate evaluated over the newest
  ``PLATEAU_WINDOW_M`` evaluated member checkpoints: :func:`mann_kendall`
  restricted to that window's point-estimate Elo sequence (reused verbatim,
  same degenerate pins); a named windowed contrast Delta_window
  (:func:`_windowed_contrast`) -- explicitly distinct from :func:`delta_hat`,
  which is defined only over the complete K-set -- with its CI taken over the
  same :func:`bootstrap_replicates` draw and the same
  :func:`order_statistic_ci` rule the Delta CI uses; and a GPU-hour span read
  from the elo-curve join. The anti-flap confirmation clause requires this
  conjunction to hold at ``PLATEAU_CONFIRMATION_COUNT`` consecutive
  evaluated-member snapshots before reporting PLATEAU. §9 pin 9 pins only raw
  per-cell immutability -- a completed cell's own content never changes -- it
  says nothing about a *derived* rating being invariant to later evidence,
  and here it is not: §9 pin 2's one anchored Bradley-Terry fit connects every
  agent in play, including a rung-8 historical opponent, which keeps its own
  earlier rung-7 rating's identity (``core.eval_agents.historical_opponents``'s
  module note), so a later candidate's fresh matches generically shift an
  earlier checkpoint's fitted rating too. Slicing every confirmation window
  out of one fit computed over *all* currently-scored cells would therefore
  let the newest evidence retroactively repaint what is supposed to be an
  independent, earlier reading. Only the manifest read itself is single
  (``core.eval_store.load_snapshot``); every window but the newest is instead
  refit from that one read *truncated* to its own ``member_prefix``
  (mirroring the existing :func:`_snapshot_cell_records` truncation) --
  reconstructing bit-for-bit what a standalone snapshot taken back when that
  earlier member was newest would have shown, precisely because pin 9's
  immutability guarantee means the cells a smaller prefix admits are exactly
  the cells that already existed back then. ``insufficient_data`` is a
  first-class third outcome, never coerced into ``no_plateau`` or ``plateau``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from core.elo import Match, fit_elo
from core.eval_protocol import (
    BOOTSTRAP_B_ADMISSIBLE_MODULUS,
    BOOTSTRAP_B_ADMISSIBLE_REMAINDER,
    BOOTSTRAP_B_PRODUCTION,
    BOOTSTRAP_CI_LOWER_QUANTILE,
    BOOTSTRAP_CI_UPPER_QUANTILE,
    PLATEAU_CI_WIDTH_THRESHOLD_ELO,
    PLATEAU_CONFIRMATION_COUNT,
    PLATEAU_GPU_HOURS_MIN,
    PLATEAU_MK_ALPHA,
    PLATEAU_WINDOW_M,
    PROTOCOL_VERSION,
    REGISTRY,
    protocol_fingerprint,
)
from core.eval_store import (
    EvalSnapshot,
    PairRecord,
    eval_dir,
    iter_cells,
    load_snapshot,
    read_cell,
    records_to_match,
)
from core.observability import reduce_run
from core.run_identity import read_stored_config
from core.seeding import PURPOSE_BOOTSTRAP, derive_seed

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


def _snapshot_cell_records(
    snapshot: EvalSnapshot,
) -> list[tuple[str, str, list[PairRecord]]]:
    """Read every in-scope cell's identities and raw pair records.

    Shared by :func:`snapshot_matches` (whole-cell aggregation, the point-estimate
    population) and :func:`bootstrap_replicate_matches` (per-pair-record
    resampling within each cell) -- both need exactly the same cell scope and
    order, so the filter lives in one place rather than two copies that could
    silently drift apart.

    Reads every cell the snapshot marks complete (``core.eval_store.iter_cells``)
    but keeps only those belonging to the snapshot's *complete contiguous member
    prefix* (``snapshot.member_prefix``): a completed cell whose candidate version
    sits beyond that prefix is real evidence for a not-yet-fully-scored member
    (``EvalSnapshot.completed_cell_ids``'s own docstring notes such cells are
    visible there -- "per-checkpoint live reporting reads this set directly" --
    precisely because they sit *outside* the contiguous prefix), so it is excluded
    here rather than silently admitted into the §1 point estimate or the
    bootstrap -- the "never partial data" analysis-snapshot convention both are
    pinned to (task 1 pin 9, P2.2). A cell that is merely scheduled (never
    completed at all) is already structurally absent from the snapshot and never
    reaches this function in the first place.

    Args:
        snapshot: A frozen snapshot from ``core.eval_store.load_snapshot``.

    Returns:
        ``(candidate_identity, opponent_identity, records)`` per in-scope cell, in
        sorted cell-id order (``core.eval_store.iter_cells``'s own order) --
        immaterial to the point-estimate fit itself (``core.elo.fit_elo``
        collapses duplicate matchups and sums are order-independent) but pinned
        deterministic for the bootstrap, whose single shared per-replicate
        generator is threaded across cells in exactly this order (task 1 pin 7:
        "cells iterated in sorted cell-id order").
    """
    cells: list[tuple[str, str, list[PairRecord]]] = []
    for path in iter_cells(snapshot):
        header, records = read_cell(path)
        if header.cell_id.candidate_version > snapshot.member_prefix:
            continue
        cells.append((header.candidate_identity, header.opponent_identity, records))
    return cells


def snapshot_matches(snapshot: EvalSnapshot) -> list[Match]:
    """Aggregate an eval snapshot's in-scope cells into ``core.elo.Match`` aggregates.

    Args:
        snapshot: A frozen snapshot from ``core.eval_store.load_snapshot``.

    Returns:
        One ``Match`` per in-scope cell (:func:`_snapshot_cell_records`'s own
        scope), in sorted cell-id order.
    """
    return [
        records_to_match(candidate_identity, opponent_identity, records)
        for candidate_identity, opponent_identity, records in _snapshot_cell_records(snapshot)
    ]


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


# ---------------------------------------------------------------------------------
# The within-cell paired-bootstrap resampler (task 7.1; task 1 pin 7).
# ---------------------------------------------------------------------------------


def bootstrap_seed(eval_seed: int) -> int:
    """Derive the run's one recorded bootstrap seed (design doc §12 M4's "its own
    recorded seed"; task 1 pin 7).

    Derived once, from the run's evaluation seed, and recorded downstream (the
    verdict artifact) rather than re-derived from ``eval_seed`` at every call site
    -- every replicate's own stream is in turn derived from *this* seed (see
    :func:`bootstrap_replicate_matches`), never from ``eval_seed`` directly, so
    the whole bootstrap is reproducible from the one recorded integer.

    Args:
        eval_seed: The run's evaluation seed (``core.runconfig.EvalConfig.eval_seed``).

    Returns:
        ``derive_seed(eval_seed, core.seeding.PURPOSE_BOOTSTRAP)``.
    """
    return derive_seed(eval_seed, PURPOSE_BOOTSTRAP)


def _resample_cell(rng: random.Random, records: Sequence[PairRecord]) -> list[PairRecord]:
    """Resample one cell's pair records with replacement to its own pair count.

    Args:
        rng: The replicate's shared generator, consumed in the caller's fixed
            cell order (see :func:`_resample_matches`).
        records: The cell's stored pair records -- the population drawn from.

    Returns:
        ``len(records)`` records drawn independently and uniformly at random,
        with replacement, from ``records`` (task 1 pin 7's within-cell bootstrap
        unit: a cell never grows or shrinks under resampling). ``[]`` if
        ``records`` is empty.
    """
    if not records:
        return []
    return rng.choices(records, k=len(records))


def _resample_matches(
    cells: Sequence[tuple[str, str, Sequence[PairRecord]]], seed: int
) -> list[Match]:
    """Resample every cell once, under one seed, and re-aggregate into ``Match`` objects.

    One ``random.Random(seed)`` is threaded across every cell **in the order
    ``cells`` is given** -- callers must supply :func:`_snapshot_cell_records`'s
    own sorted-cell-id order, never a set or a re-sort here, so that a draw
    landing on one cell can never silently shift onto another (task 1 pin 7:
    "cells iterated in sorted cell-id order"). A resampled ``Match`` always names
    the same ``(candidate_identity, opponent_identity)`` pair as the cell it came
    from -- only its aggregate score and (in general) its per-pair composition
    change -- so the matchup graph's connectivity to the anchor is identical to
    the point estimate's across every replicate; resampling can never disconnect
    an agent that the point estimate reached.

    Args:
        cells: ``(candidate_identity, opponent_identity, records)`` per in-scope
            cell, in sorted cell-id order (:func:`_snapshot_cell_records`).
        seed: This replicate's own derived seed.

    Returns:
        One resampled ``Match`` per entry in ``cells``, in the same order.
    """
    rng = random.Random(seed)
    return [
        records_to_match(candidate_identity, opponent_identity, _resample_cell(rng, records))
        for candidate_identity, opponent_identity, records in cells
    ]


def bootstrap_replicate_matches(snapshot: EvalSnapshot, bootstrap_seed: int, b: int) -> list[Match]:
    """Resample replicate ``b``'s in-scope cells from ``snapshot`` into ``Match`` objects.

    Replicate ``b`` draws its own generator from ``derive_seed(bootstrap_seed,
    "replicate", b)`` (task 1 pin 7) -- independent of every other replicate's
    stream (a fresh ``random.Random`` per call, never a shared or advanced one),
    so replicate ``b`` is reproducible from ``(bootstrap_seed, b)`` alone, without
    any other replicate ever having been computed. Within each of ``snapshot``'s
    in-scope cells (:func:`_snapshot_cell_records`'s member-prefix scope --
    partial cells never enter, task 1 pin 9), the cell's pair records are
    resampled with replacement to the cell's own original pair count and
    re-aggregated (:func:`_resample_matches`).

    Args:
        snapshot: A frozen snapshot from ``core.eval_store.load_snapshot``.
        bootstrap_seed: The run's recorded bootstrap seed (:func:`bootstrap_seed`).
        b: The replicate's index (0-based).

    Returns:
        One resampled ``Match`` per in-scope cell, in sorted cell-id order.
    """
    seed = derive_seed(bootstrap_seed, "replicate", b)
    return _resample_matches(_snapshot_cell_records(snapshot), seed)


def bootstrap_replicate(
    snapshot: EvalSnapshot,
    bootstrap_seed: int,
    b: int,
    *,
    point_estimate: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fit replicate ``b``'s anchored Elo ratings, warm-started from the point estimate.

    Reproducible from ``(snapshot, bootstrap_seed, b)`` alone: a single replicate
    can be recomputed in isolation, with no other replicate ever having run, and
    matches that same replicate's ratings inside a full
    :func:`bootstrap_replicates` run bit-for-bit -- same seed derivation, same
    in-scope cells, same warm start.

    Args:
        snapshot: A frozen snapshot from ``core.eval_store.load_snapshot``.
        bootstrap_seed: The run's recorded bootstrap seed (:func:`bootstrap_seed`).
        b: The replicate's index (0-based).
        point_estimate: The snapshot's point-estimate ratings
            (:func:`fit_snapshot_elo`), forwarded to ``core.elo.fit_elo`` as
            ``initial_ratings`` (task 6's extension, S2.3) -- a pure speedup to
            the same anchored fixed point, never a different answer (``fit_elo``'s
            own docstring: the ascent's fixed point does not depend on where it
            starts). ``None`` (the default) computes the point estimate fresh
            from ``snapshot``, so this function is usable standing entirely on
            its own; a caller refitting many replicates
            (:func:`bootstrap_replicates`) fits the point estimate once and
            passes it through instead of repeating that fit ``B`` times.

    Returns:
        This replicate's fitted ratings dict -- same shape as
        :func:`fit_snapshot_elo`'s own return value.

    Raises:
        ValueError: Propagated from ``core.elo.fit_elo`` if ``snapshot`` has no
            matchup at all, or some agent is disconnected from
            :data:`ANCHOR_AGENT` -- resampling never changes which agents are
            connected (see :func:`_resample_matches`), so this can only fire here
            if the point estimate itself would also raise it.
    """
    if point_estimate is None:
        point_estimate = fit_snapshot_elo(snapshot)
    matches = bootstrap_replicate_matches(snapshot, bootstrap_seed, b)
    return fit_elo(matches, anchor=ANCHOR_AGENT, initial_ratings=point_estimate)


def bootstrap_replicates(
    snapshot: EvalSnapshot, bootstrap_seed: int, B: int
) -> Iterator[dict[str, float]]:
    """Yield ``B`` bootstrap replicates' anchored Elo ratings, deterministically, by index.

    Reads ``snapshot``'s in-scope cells and fits its point estimate exactly once
    (never re-read or re-fit per replicate), then yields replicate ``0`` through
    ``B - 1`` in order. Each yielded value is identical to what
    :func:`bootstrap_replicate` computes for that same index in isolation (same
    cells, same seed derivation, same warm start) -- this function exists purely
    to amortize the one-time snapshot read and point-estimate fit across many
    replicates, not to compute anything a caller could not get by calling
    :func:`bootstrap_replicate` ``B`` times.

    Args:
        snapshot: A frozen snapshot from ``core.eval_store.load_snapshot``.
        bootstrap_seed: The run's recorded bootstrap seed (:func:`bootstrap_seed`).
        B: How many replicates to yield. This function accepts any non-negative
            count; validating ``B`` against the pinned admissible-``B`` rule
            (task 1 pin 7: ``B ≡ 39 mod 40``) is task 7.2's concern, not this
            resampling engine's.

    Yields:
        Each replicate's fitted ratings dict (:func:`bootstrap_replicate`'s own
        return shape), for ``b`` in ``range(B)``, in increasing order.
    """
    point_estimate = fit_snapshot_elo(snapshot)
    cells = _snapshot_cell_records(snapshot)
    for b in range(B):
        seed = derive_seed(bootstrap_seed, "replicate", b)
        matches = _resample_matches(cells, seed)
        yield fit_elo(matches, anchor=ANCHOR_AGENT, initial_ratings=point_estimate)


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


# ---------------------------------------------------------------------------------
# Delta-hat, the admissible-B order-statistic CIs, and per-checkpoint CIs
# (task 7.2; design doc §1, §9 pin 7-8).
# ---------------------------------------------------------------------------------


def delta_windows(K: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return §1's first-third and final-third 1-indexed member-version windows.

    Defined only over a *complete* K-member series (task 1 pin 8: "no prefix
    Delta is ever computed") -- members are numbered ``1..K`` (``v0`` is
    excluded from the K-member set, §6.2's checkpoint/publish boundary rule),
    so both windows are plain slices of that contiguous range, never derived
    from anything a prefix snapshot could supply on its own.

    Args:
        K: The complete series length (member count).

    Returns:
        ``(first_versions, final_versions)``: the first ``ceil(K/3)`` member
        versions ``(1, ..., ceil(K/3))`` and the final ``ceil(K/3)`` member
        versions ``(K - ceil(K/3) + 1, ..., K)`` -- the two windows §1's Delta
        averages over. Documented boundary examples: ``K=3 -> ((1,), (3,))``;
        ``K=4 -> ((1, 2), (3, 4))``; ``K=5 -> ((1, 2), (4, 5))``.

    Raises:
        ValueError: If ``K < 1``.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    window = -(-K // 3)  # ceil(K / 3), pure-integer -- never a float rounding path.
    first_versions = tuple(range(1, window + 1))
    final_versions = tuple(range(K - window + 1, K + 1))
    return first_versions, final_versions


def delta_hat(curve: Sequence[tuple[int, float]]) -> float:
    """Delta-hat: the original-sample §1 contrast over one complete rung-7 Elo curve.

    The same formula computes both the reported point estimate (over the
    point-estimate curve) and every bootstrap replicate's Delta_b (over that
    replicate's own curve, task 1 pin 7) -- there is exactly one Delta
    computation in this module, never two.

    Args:
        curve: ``(model_version, elo)`` pairs -- exactly
            :func:`checkpoint_elo`'s own return shape -- which this function
            requires to cover member versions ``1..K`` contiguously (``K =
            len(curve)``; task 1 pin 8's "complete K-set" precondition). A
            caller must never invoke this over an incomplete prefix; passing
            one raises rather than silently averaging over the wrong window.

    Returns:
        ``mean(elo over the final ceil(K/3) versions) - mean(elo over the
        first ceil(K/3) versions)`` (:func:`delta_windows`).

    Raises:
        ValueError: If ``curve`` is empty, or its versions are not exactly
            ``{1, ..., len(curve)}`` (a non-contiguous or incomplete series).
    """
    if not curve:
        raise ValueError("delta_hat requires a non-empty checkpoint curve")
    K = len(curve)
    elo_by_version = dict(curve)
    if set(elo_by_version) != set(range(1, K + 1)):
        raise ValueError(
            f"delta_hat requires a complete, contiguous 1..{K} member series "
            f"(task 1 pin 8: no prefix Delta is ever computed); got versions "
            f"{sorted(elo_by_version)}"
        )
    first_versions, final_versions = delta_windows(K)
    first_mean = sum(elo_by_version[v] for v in first_versions) / len(first_versions)
    final_mean = sum(elo_by_version[v] for v in final_versions) / len(final_versions)
    return final_mean - first_mean


def replicate_deltas(replicate_ratings: Iterable[dict[str, float]]) -> list[float]:
    """Delta_b for every bootstrap replicate's fitted ratings.

    Args:
        replicate_ratings: One fitted ratings dict per replicate (e.g.
            :func:`bootstrap_replicates`'s own yield), each spanning the same
            complete K-member series as the point estimate.

    Returns:
        ``delta_hat(checkpoint_elo(ratings))`` per replicate, in the same
        order -- no second resample, only the already-fitted ratings each
        replicate carries.

    Raises:
        ValueError: Propagated from :func:`delta_hat` if some replicate's
            rung-7 curve is not a complete, contiguous ``1..K`` series.
    """
    return [delta_hat(checkpoint_elo(ratings)) for ratings in replicate_ratings]


def _validate_admissible_B(B: int) -> None:
    """Raise ``ValueError`` unless ``B`` is admissible under the pinned rank rule.

    Admissible means both order-statistic ranks
    ``(B+1)*BOOTSTRAP_CI_LOWER_QUANTILE`` and ``(B+1)*BOOTSTRAP_CI_UPPER_QUANTILE``
    land on an integer -- equivalently ``B % BOOTSTRAP_B_ADMISSIBLE_MODULUS ==
    BOOTSTRAP_B_ADMISSIBLE_REMAINDER`` (``B ≡ 39 mod 40``: 39, 79, ..., 1,999;
    task 1 pin 7).

    Args:
        B: The candidate replicate count.

    Raises:
        ValueError: If ``B`` is not a positive int admissible under the rule
            above.
    """
    if (
        isinstance(B, bool)
        or not isinstance(B, int)
        or B < BOOTSTRAP_B_ADMISSIBLE_REMAINDER
        or B % BOOTSTRAP_B_ADMISSIBLE_MODULUS != BOOTSTRAP_B_ADMISSIBLE_REMAINDER
    ):
        raise ValueError(
            f"B={B!r} is not admissible: the order-statistic CI rule requires both ranks "
            f"(B+1)*{BOOTSTRAP_CI_LOWER_QUANTILE} and (B+1)*{BOOTSTRAP_CI_UPPER_QUANTILE} to "
            f"be integral, i.e. B % {BOOTSTRAP_B_ADMISSIBLE_MODULUS} == "
            f"{BOOTSTRAP_B_ADMISSIBLE_REMAINDER} (B ≡ 39 mod 40: 39, 79, ..., 1999; "
            f"production default {BOOTSTRAP_B_PRODUCTION})"
        )


def order_statistic_ci(
    values: Sequence[float], B: int = BOOTSTRAP_B_PRODUCTION
) -> tuple[float, float]:
    """The pinned 95% order-statistic CI over ``B`` bootstrap replicate values.

    The one quantile convention used at every admissible ``B`` (task 1 pin 7):
    endpoints at ranks ``(B+1)*BOOTSTRAP_CI_LOWER_QUANTILE`` and
    ``(B+1)*BOOTSTRAP_CI_UPPER_QUANTILE`` -- 1-indexed order statistics of the
    sorted values -- never a second rule for a reduced-``B`` test. Used both
    for Delta's CI (over :func:`replicate_deltas`'s output) and, unchanged,
    for :func:`per_checkpoint_ci`'s per-version columns -- one CI rule, two
    call sites, never two formulas.

    Args:
        values: The replicate statistic values (Delta_b, or one checkpoint's
            per-replicate rating); must have exactly ``B`` entries.
        B: Replicate count; must be admissible (:func:`_validate_admissible_B`).
            Defaults to the pinned production value
            (``core.eval_protocol.BOOTSTRAP_B_PRODUCTION``).

    Returns:
        ``(lower, upper)``: the ``(B+1)*BOOTSTRAP_CI_LOWER_QUANTILE``-th and
        ``(B+1)*BOOTSTRAP_CI_UPPER_QUANTILE``-th order statistics (1-indexed)
        of ``sorted(values)``.

    Raises:
        ValueError: If ``B`` is not admissible, or ``len(values) != B``.
    """
    _validate_admissible_B(B)
    if len(values) != B:
        raise ValueError(f"order_statistic_ci requires exactly B={B} values, got {len(values)}")
    ordered = sorted(values)
    lower_rank = (B + 1) * BOOTSTRAP_CI_LOWER_QUANTILE
    upper_rank = (B + 1) * BOOTSTRAP_CI_UPPER_QUANTILE
    # _validate_admissible_B already guarantees both ranks are exact integers.
    lower = ordered[int(round(lower_rank)) - 1]
    upper = ordered[int(round(upper_rank)) - 1]
    return lower, upper


def delta_gate(ci: tuple[float, float]) -> bool:
    """Success = the Delta CI lies strictly above 0 (task 1's pre-registered gate).

    Args:
        ci: A :func:`order_statistic_ci` result over :func:`replicate_deltas`.

    Returns:
        ``ci[0] > 0.0`` -- the lower CI endpoint alone determines the gate,
        since ``order_statistic_ci`` already guarantees ``ci[0] <= ci[1]``.
    """
    return ci[0] > 0.0


def per_checkpoint_ci(
    replicate_ratings: Sequence[dict[str, float]], B: int = BOOTSTRAP_B_PRODUCTION
) -> list[tuple[int, tuple[float, float]]]:
    """Per-checkpoint (rung-7 model_version) 95% CIs from the same ``B`` replicates.

    The same :func:`order_statistic_ci` rule applied column-wise over each
    member version's ``B`` per-replicate ratings -- no second resample (task
    1 pin 7's "per-checkpoint CIs are the same rule over the same replicates'
    per-version ratings"). Resampling never disconnects an agent the point
    estimate reached (:func:`bootstrap_replicate_matches`'s own docstring), so
    every replicate names the identical set of rung-7 member versions; this
    function reads that version set from the union across replicates so a
    caller's contract violation (a replicate missing a version) surfaces as a
    ``KeyError`` naming the offending version rather than a silently
    truncated report.

    Args:
        replicate_ratings: Exactly ``B`` replicate fitted-ratings dicts (e.g.
            a materialized list of :func:`bootstrap_replicates`'s own yield).
        B: Replicate count; must be admissible. Defaults to the pinned
            production value.

    Returns:
        ``(model_version, (lower, upper))`` pairs, ordered by ``model_version``
        ascending.

    Raises:
        ValueError: If ``B`` is not admissible, or ``len(replicate_ratings) !=
            B``.
        KeyError: If some replicate's rung-7 curve is missing a member version
            another replicate carries.
    """
    if len(replicate_ratings) != B:
        raise ValueError(
            f"per_checkpoint_ci requires exactly B={B} replicates, got {len(replicate_ratings)}"
        )
    per_replicate_curves = [dict(checkpoint_elo(ratings)) for ratings in replicate_ratings]
    versions = sorted({version for curve in per_replicate_curves for version in curve})
    return [
        (version, order_statistic_ci([curve[version] for curve in per_replicate_curves], B))
        for version in versions
    ]


@dataclass(frozen=True)
class MannKendallResult:
    """One Mann-Kendall trend-test reading over a point-estimate Elo sequence.

    Reported, never gating (task 1 pin 7) -- a secondary, functional-form-free
    check on the evaluated prefix's checkpoint Elo sequence, never the §1
    pre-registered contrast itself.

    Attributes:
        n: The sequence length actually evaluated.
        insufficient_data: ``True`` iff ``n < 3`` -- the pinned degenerate
            case (no z/p, never a trend claim); ``s``/``z``/``p`` are all
            ``None`` exactly when this is ``True``.
        s: The classic Mann-Kendall S statistic, or ``None`` if
            ``insufficient_data``.
        z: The continuity-corrected z statistic, or ``None`` if
            ``insufficient_data``.
        p: The two-sided normal-approximation p-value, or ``None`` if
            ``insufficient_data``.
    """

    n: int
    insufficient_data: bool
    s: int | None
    z: float | None
    p: float | None


def mann_kendall(values: Sequence[float]) -> MannKendallResult:
    """Mann-Kendall trend test over a point-estimate Elo sequence, pure stdlib.

    Classic S (sum of pairwise signs), tie-corrected variance, continuity-
    corrected ``z = (S - sign(S)) / sigma``, two-sided normal-approximation p
    (task 1 pin 7). Two degenerate cases are pinned rather than left to the
    implementation: fewer than 3 points is insufficient data (no trend claim
    at all); every observation tied forces the tie-corrected variance to 0
    (and, with it, ``S = 0``, since every pairwise comparison is a tie) --
    handled explicitly so no division-by-zero path exists, returning ``z =
    0.0, p = 1.0`` directly rather than routing through the general formula.

    Args:
        values: The evaluated prefix's point-estimate Elo sequence, in
            checkpoint order (ascending ``model_version``).

    Returns:
        A :class:`MannKendallResult` -- ``insufficient_data=True`` (``s=z=p=
        None``) if ``len(values) < 3``; otherwise a populated ``(s, z, p)``.
    """
    n = len(values)
    if n < 3:
        return MannKendallResult(n=n, insufficient_data=True, s=None, z=None, p=None)

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = values[j] - values[i]
            s += (diff > 0.0) - (diff < 0.0)

    tie_counts = Counter(values)
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in tie_counts.values())
    variance_numerator = n * (n - 1) * (2 * n + 5) - tie_term
    if variance_numerator <= 0:
        # Every observation tied forces S = 0 (pin 7) -- no division-by-zero path.
        return MannKendallResult(n=n, insufficient_data=False, s=0, z=0.0, p=1.0)

    sigma = math.sqrt(variance_numerator / 18.0)
    sign_s = (s > 0) - (s < 0)
    z = (s - sign_s) / sigma
    p = math.erfc(abs(z) / math.sqrt(2.0))  # == 2 * (1 - Phi(|z|)), stabler near the tail.
    return MannKendallResult(n=n, insufficient_data=False, s=s, z=z, p=p)


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


# ---------------------------------------------------------------------------------
# verdict.json assembly (task 7.3; design doc §9/§12; task 1 pin 8; reviews
# P4/P6/P7; second pass P2.1/P2.2/S2.2).
# ---------------------------------------------------------------------------------

_VERDICT_NAME = "verdict.json"


def verdict_path(run_dir: Path | str) -> Path:
    """Return the §12 M5.5 verdict artifact's on-disk path for one run.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``<run_dir>/eval/verdict.json``.
    """
    return eval_dir(run_dir) / _VERDICT_NAME


def _file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file's raw on-disk bytes.

    Args:
        path: The file to hash.

    Returns:
        A 64-character lowercase hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_verdict(run_dir: Path | str, *, B: int = BOOTSTRAP_B_PRODUCTION) -> dict[str, Any]:
    """Assemble and durably write the §12 M5.5 verdict artifact.

    The one-shot orchestrator over every earlier piece of this module. Reads
    the watched run's own stored config
    (``core.run_identity.read_stored_config``) for ``k_target``
    (``training.checkpoint_count``) and the eval seed
    (``evaluation.eval_seed``); loads the run's eval-store snapshot
    (``core.eval_store.load_snapshot``); fits the point estimate
    (:func:`fit_snapshot_elo`) and refreshes the §1 plot series
    (:func:`elo_curve`) from that *same* snapshot object, so the verdict's
    evidence fingerprint and its ``elo_curve.json`` reference always describe
    the identical dataset; draws ``B`` bootstrap replicates
    (:func:`bootstrap_replicates`) exactly once, reused for both
    per-checkpoint CIs (:func:`per_checkpoint_ci`) and -- **iff** the
    snapshot's contiguous member prefix equals the complete ``k_target``
    -member set (task 1 pin 8) -- the Delta contrast (:func:`delta_hat`,
    :func:`replicate_deltas`, :func:`order_statistic_ci`,
    :func:`delta_gate`); and runs Mann-Kendall (:func:`mann_kendall`) over the
    evaluated prefix's point-estimate curve unconditionally (reported, never
    gating, task 1 pin 7).

    Before the complete K-set exists, the artifact is provisional: it carries
    no ``delta_hat``, no Delta CI, and no gate anywhere -- only
    ``delta: null`` plus an explicit ``reason`` string -- alongside the
    per-checkpoint CIs and Mann-Kendall result that *are* what live
    (incomplete-prefix) reporting consists of. ``authoritative`` requires
    both the complete K-set **and** ``B ==
    core.eval_protocol.BOOTSTRAP_B_PRODUCTION`` exactly -- a complete K-set
    evaluated at a smaller admissible ``B`` (e.g. a reduced-cost test run)
    still carries a full Delta/CI/gate, just never the ``authoritative`` flag.

    Args:
        run_dir: The run's root directory -- the eval snapshot's own root,
            the root ``core.run_identity.read_stored_config`` reads
            ``config.json`` from, and the root this writes
            ``eval/verdict.json`` (and refreshes ``eval/elo_curve.json``)
            under.
        B: The bootstrap replicate count. Must be admissible (task 1 pin 7:
            ``B ≡ 39 mod 40``; see :func:`order_statistic_ci`). Defaults to
            the pinned production value; the artifact always records the
            value actually used.

    Returns:
        The verdict payload -- the identical JSON-safe dict durably written
        (temp-name-then-``os.replace``, sorted keys, so the same records and
        seed always produce bit-identical bytes) to :func:`verdict_path`.

    Raises:
        ValueError: If ``B`` is not admissible, or propagated from
            :func:`fit_snapshot_elo`/:func:`elo_curve` (an empty snapshot, an
            agent disconnected from the anchor, or a scored member with no
            ``checkpoint_published`` marker).
        FileNotFoundError: If ``run_dir`` has no stored ``config.json``
            (``core.run_identity.read_stored_config``).
    """
    _validate_admissible_B(B)
    run_dir = Path(run_dir)

    snapshot = load_snapshot(run_dir)
    stored_config = read_stored_config(run_dir)
    k_target = stored_config.run.training.checkpoint_count
    eval_seed_value = stored_config.run.evaluation.eval_seed

    seed = bootstrap_seed(eval_seed_value)
    point_curve = checkpoint_elo(fit_snapshot_elo(snapshot))
    checkpoints_evaluated = snapshot.member_prefix
    is_complete_k_set = checkpoints_evaluated == k_target

    replicate_ratings = list(bootstrap_replicates(snapshot, seed, B))
    elo_by_version = dict(point_curve)
    per_checkpoint_payload = [
        {"model_version": version, "elo": elo_by_version[version], "ci": [lower, upper]}
        for version, (lower, upper) in per_checkpoint_ci(replicate_ratings, B)
    ]

    mk = mann_kendall([elo for _, elo in point_curve])
    mann_kendall_payload = {
        "n": mk.n,
        "insufficient_data": mk.insufficient_data,
        "s": mk.s,
        "z": mk.z,
        "p": mk.p,
    }

    if is_complete_k_set:
        delta_ci = order_statistic_ci(replicate_deltas(replicate_ratings), B)
        delta_payload: dict[str, Any] | None = {
            "delta_hat": delta_hat(point_curve),
            "ci": [delta_ci[0], delta_ci[1]],
            "gate": delta_gate(delta_ci),
        }
        reason = None
    else:
        delta_payload = None
        reason = (
            f"snapshot prefix covers {checkpoints_evaluated} of {k_target} required "
            "checkpoint(s) -- the Delta contrast is only ever computed over the "
            "complete K-set (task 1 pin 8); no prefix Delta exists, advisory or otherwise"
        )

    elo_curve(run_dir, snapshot)
    elo_curve_fingerprint = _file_sha256(elo_curve_path(run_dir))

    payload: dict[str, Any] = {
        "authoritative": is_complete_k_set and B == BOOTSTRAP_B_PRODUCTION,
        "checkpoints_evaluated": checkpoints_evaluated,
        "k_target": k_target,
        "bootstrap_b": B,
        "bootstrap_seed": seed,
        "evidence_fingerprint": snapshot.snapshot_fingerprint,
        "elo_curve_fingerprint": elo_curve_fingerprint,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint(),
        "protocol_constants": dict(REGISTRY),
        "per_checkpoint": per_checkpoint_payload,
        "mann_kendall": mann_kendall_payload,
        "delta": delta_payload,
        "reason": reason,
    }
    _atomic_write_json(verdict_path(run_dir), payload)
    return payload


# ---------------------------------------------------------------------------------
# The profiled-plateau detector (task 8; design doc §12 M4; §9 pins 2 and 9).
# ---------------------------------------------------------------------------------

#: Tri-state :attr:`PlateauResult.outcome` values -- deliberately plain strings
#: (never a bool, never a class implementing ``__bool__``) so a plateau claim can
#: only ever be read by comparing against one of these three named values, never
#: by accidental truthiness coercion.
PLATEAU_OUTCOME_PLATEAU = "plateau"
PLATEAU_OUTCOME_NO_PLATEAU = "no_plateau"
PLATEAU_OUTCOME_INSUFFICIENT_DATA = "insufficient_data"

#: The closed, three-member type of :attr:`PlateauResult.outcome` -- a type
#: checker rejects any fourth value at the call site, on top of the runtime
#: guarantee that :func:`detect_plateau` only ever returns one of the three
#: named constants above.
PlateauOutcome = Literal["plateau", "no_plateau", "insufficient_data"]


def _windowed_contrast(curve_by_version: dict[int, float], versions: Sequence[int]) -> float:
    """The named windowed contrast Delta_window over one window's Elo curve.

    The same half-window split (``core.eval_protocol.PLATEAU_HALF_WINDOW_RULE``)
    computes both the point-estimate value (over ``curve_by_version`` built from
    :func:`fit_snapshot_elo`) and every bootstrap replicate's own value (over a
    replicate's own curve) -- one formula, reused verbatim for both, mirroring how
    :func:`delta_hat` is the single formula behind both Delta-hat and every
    replicate's Delta_b.

    Args:
        curve_by_version: A rung-7 Elo curve as ``{model_version: elo}``, covering
            at least every version in ``versions``.
        versions: The window's member versions, ascending. Not required to be the
            whole evaluated series -- only this window's slice.

    Returns:
        ``mean(elo over the newest ceil(len(versions)/2) versions) - mean(elo
        over the oldest ceil(len(versions)/2) versions)``. For the pinned ``M`` =
        8 (even), the two halves are the non-overlapping first/last four members;
        the ``ceil`` generalizes correctly (with overlap) if ``M`` were ever
        odd, per the doc amendment's own formula.
    """
    half = -(-len(versions) // 2)  # ceil(len(versions) / 2), pure-integer.
    oldest = versions[:half]
    newest = versions[-half:]
    oldest_mean = sum(curve_by_version[v] for v in oldest) / len(oldest)
    newest_mean = sum(curve_by_version[v] for v in newest) / len(newest)
    return newest_mean - oldest_mean


@dataclass(frozen=True)
class WindowCondition:
    """One M-checkpoint window's profiled-plateau sub-conditions (design doc §12 M4).

    One instance is built per snapshot examined by the anti-flap confirmation
    clause -- the window ending at the newest evaluated member, and the window
    ending one member earlier -- each carrying every sub-condition's measured
    value against its pinned threshold, so a plateau (or non-plateau) claim is
    auditable rather than a bare bool.

    Attributes:
        newest_version: This window's newest evaluated member version -- what
            "the snapshot ending at this member" means for the confirmation
            clause (design doc: "two consecutive snapshots whose newest members
            are themselves consecutive versions").
        versions: The window's ``core.eval_protocol.PLATEAU_WINDOW_M`` member
            versions, ascending and contiguous (``newest_version - M + 1 ..
            newest_version``).
        mann_kendall: The windowed Mann-Kendall trend reading (:func:`mann_kendall`,
            reusing its own pinned degenerate cases verbatim) over this window's
            point-estimate Elo sequence, in ascending-version order.
        mk_non_significant: ``True`` iff ``mann_kendall`` is not
            insufficient-data, the window is not ``mk_all_tied``, and
            ``mann_kendall.p >= core.eval_protocol.PLATEAU_MK_ALPHA`` -- the
            trend sub-condition. Forced ``False`` in both degenerate cases
            (mirrors how ``gpu_span_sufficient`` is forced ``False`` rather
            than left ``None`` when ``gpu_hours_span`` is missing) so a
            degenerate reading can never masquerade as a legitimate
            non-significant trend.
        mk_all_tied: ``True`` iff every point-estimate Elo value in this
            window is bit-identical, i.e. ``mann_kendall`` hit its own pinned
            sigma=0 degenerate branch (``s=0, z=0.0, p=1.0`` -- see
            :func:`mann_kendall`'s docstring) while still reporting
            ``insufficient_data=False``. That branch exists to avoid a
            division-by-zero path, not to certify a real non-significant
            trend -- an all-tied window carries zero statistical information
            and must never stand as evidence of a plateau (design doc §12 M4's
            insufficient-data clause; tasks/m4/008 Test Strategy).
        contrast: The named windowed contrast Delta_window's point estimate
            (:func:`_windowed_contrast` over the point-estimate curve).
        contrast_ci: Delta_window's 95% CI, taken over the same bootstrap
            replicates and the same admissible-B order-statistic rank rule as
            the §1 Delta (:func:`order_statistic_ci`).
        contrast_ci_width: ``contrast_ci[1] - contrast_ci[0]``.
        ci_narrow: ``True`` iff ``contrast_ci_width <
            core.eval_protocol.PLATEAU_CI_WIDTH_THRESHOLD_ELO`` (strictly below).
        gpu_hours_span: This window's GPU-hour span -- the newest member's
            cumulative single-counted GPU-hours minus the oldest member's (the
            §1 x-axis join) -- or ``None`` if some window member has no
            GPU-hour coordinate in the elo-curve join read (the insufficient-data
            trigger; see :func:`detect_plateau`).
        gpu_span_sufficient: ``True`` iff ``gpu_hours_span`` is not ``None`` and
            ``>= core.eval_protocol.PLATEAU_GPU_HOURS_MIN``.
        satisfied: The full conjunction this window itself satisfies --
            ``mk_non_significant and ci_narrow and gpu_span_sufficient``. Two
            consecutive ``satisfied`` windows are what :func:`detect_plateau`
            requires before declaring PLATEAU.
    """

    newest_version: int
    versions: tuple[int, ...]
    mann_kendall: MannKendallResult
    mk_non_significant: bool
    mk_all_tied: bool
    contrast: float
    contrast_ci: tuple[float, float]
    contrast_ci_width: float
    gpu_hours_span: float | None
    gpu_span_sufficient: bool
    ci_narrow: bool
    satisfied: bool


def _window_condition(
    newest_version: int,
    versions: tuple[int, ...],
    point_curve_by_version: dict[int, float],
    replicate_curves_by_version: Sequence[dict[int, float]],
    gpu_hours_by_version: dict[int, float],
    B: int,
) -> WindowCondition:
    """Build one :class:`WindowCondition`, evaluating all three sub-conditions.

    Args:
        newest_version: This window's newest member version.
        versions: This window's ``M`` member versions, ascending.
        point_curve_by_version: The full evaluated series' point-estimate Elo
            curve, as ``{model_version: elo}`` (covers at least ``versions``).
        replicate_curves_by_version: Each bootstrap replicate's own full curve,
            in the same dict shape, in replicate-index order -- exactly ``B``
            entries (:func:`bootstrap_replicates`, reused unmodified).
        gpu_hours_by_version: The §1 x-axis join's GPU-hours coordinate per
            scored member version (:func:`detect_plateau`'s elo-curve-artifact
            read) -- may be missing a version entirely.
        B: The replicate count backing ``replicate_curves_by_version``
            (:func:`order_statistic_ci`'s own admissible-B rule).

    Returns:
        The populated :class:`WindowCondition`. ``mk_all_tied`` is checked
        directly against the window's own point-estimate values (bit-identical
        across every member) rather than inferred from ``mk.s``/``mk.z``/
        ``mk.p`` alone -- a genuine (non-degenerate) trend reading can also
        land on ``s=0, z=0.0, p=1.0`` when its positive and negative pairwise
        signs happen to cancel, so those fields alone cannot distinguish a
        real null result from the sigma=0 degenerate branch.
    """
    window_values = [point_curve_by_version[v] for v in versions]
    mk = mann_kendall(window_values)
    mk_all_tied = (not mk.insufficient_data) and len(set(window_values)) == 1
    mk_non_significant = (
        (not mk.insufficient_data) and (not mk_all_tied) and mk.p >= PLATEAU_MK_ALPHA
    )

    contrast = _windowed_contrast(point_curve_by_version, versions)
    replicate_contrasts = [
        _windowed_contrast(curve, versions) for curve in replicate_curves_by_version
    ]
    contrast_ci = order_statistic_ci(replicate_contrasts, B)
    contrast_ci_width = contrast_ci[1] - contrast_ci[0]
    ci_narrow = contrast_ci_width < PLATEAU_CI_WIDTH_THRESHOLD_ELO

    if all(v in gpu_hours_by_version for v in versions):
        gpu_hours_span: float | None = (
            gpu_hours_by_version[versions[-1]] - gpu_hours_by_version[versions[0]]
        )
        gpu_span_sufficient = gpu_hours_span >= PLATEAU_GPU_HOURS_MIN
    else:
        gpu_hours_span = None
        gpu_span_sufficient = False

    return WindowCondition(
        newest_version=newest_version,
        versions=versions,
        mann_kendall=mk,
        mk_non_significant=mk_non_significant,
        mk_all_tied=mk_all_tied,
        contrast=contrast,
        contrast_ci=contrast_ci,
        contrast_ci_width=contrast_ci_width,
        gpu_hours_span=gpu_hours_span,
        gpu_span_sufficient=gpu_span_sufficient,
        ci_narrow=ci_narrow,
        satisfied=mk_non_significant and ci_narrow and gpu_span_sufficient,
    )


@dataclass(frozen=True)
class PlateauResult:
    """The profiled-plateau detector's auditable tri-state verdict (task 8).

    Never coerce ``outcome`` to a bool -- it is one of
    :data:`PLATEAU_OUTCOME_PLATEAU`, :data:`PLATEAU_OUTCOME_NO_PLATEAU`, or
    :data:`PLATEAU_OUTCOME_INSUFFICIENT_DATA`, and this class deliberately
    implements no ``__bool__``: a plateau claim must be read by comparing
    ``outcome`` against a named value, never by truthiness.

    Attributes:
        outcome: The tri-state verdict.
        window_m: The pinned window length used (``core.eval_protocol.
            PLATEAU_WINDOW_M``), recorded for audit even though it never
            varies at a fixed protocol version.
        current: The window ending at the newest evaluated member -- an alias
            for ``windows[0]`` -- or ``None`` iff ``outcome ==
            PLATEAU_OUTCOME_INSUFFICIENT_DATA`` because fewer than
            ``window_m`` members are evaluated at all (there is no full
            window to examine, so ``windows == ()``).
        previous: The window ending one evaluated member earlier than
            ``current`` -- an alias for ``windows[1] if len(windows) > 1 else
            None``. It is refit from a snapshot truncated to *its own*
            ``member_prefix`` (:func:`detect_plateau`'s docstring), never
            sliced out of ``current``'s fit: §9 pin 2's shared Bradley-Terry
            graph means a later candidate's matches can shift an earlier
            checkpoint's rating too, so reusing one fit for both windows would
            let ``current``'s newer evidence quietly leak into what must read
            as an independent, earlier snapshot. (This is *not* a design-doc
            quotation -- no rating-invariance claim is pinned anywhere in the
            doc; §9 pin 9 pins only raw per-cell content immutability.)
            ``None`` when fewer than ``window_m + 1`` members are evaluated
            (no second window exists yet to confirm against) or when
            ``outcome == PLATEAU_OUTCOME_INSUFFICIENT_DATA`` before any window
            was built.
        windows: Every window the anti-flap loop actually built and examined,
            newest first -- the general form of ``current``/``previous``,
            which are just convenience aliases for ``windows[0]`` and
            ``windows[1] if len(windows) > 1 else None``. Holds ``0`` to
            ``core.eval_protocol.PLATEAU_CONFIRMATION_COUNT`` entries: fewer
            than the full count only when the evaluated series itself is too
            short yet for the full confirmation depth (never a hardcoded cap
            at two -- a future ``PLATEAU_CONFIRMATION_COUNT`` bump changes
            only ``len(windows)``, not this dataclass's shape).
        confirmation_count: How many of the newest consecutive ``windows``
            satisfy the full conjunction, counted from the newest backward and
            stopping at the first that does not (``0`` to
            ``core.eval_protocol.PLATEAU_CONFIRMATION_COUNT``) -- never a
            "confirmed later, unconfirmed at the newest" count, since the
            clause is defined over *consecutive* snapshots.
        confirmed_versions: The satisfying windows' ``newest_version`` values,
            ascending -- between ``0`` and ``PLATEAU_CONFIRMATION_COUNT``
            elements, naming exactly which consecutive member checkpoints
            confirmed the plateau (or are pending confirmation).
        reason: A human-readable explanation for ``no_plateau`` or
            ``insufficient_data``; ``None`` for ``plateau`` (mirrors
            ``build_verdict``'s own ``reason`` convention).
        snapshot_fingerprint: The task-5 :class:`~core.eval_store.EvalSnapshot`
            this verdict was computed over
            (``core.eval_store.EvalSnapshot.snapshot_fingerprint``).
        elo_curve_fingerprint: The sha256 of the on-disk ``elo_curve.json``
            artifact (:func:`elo_curve_path`) this verdict's GPU-hours join
            was read from, or ``None`` if that artifact does not exist yet --
            always the artifact actually read, never recomputed or rewritten
            here (:func:`detect_plateau` triggers no write of its own).
        protocol_fingerprint: ``core.eval_protocol.protocol_fingerprint()`` at
            the moment this verdict was computed -- the same registry stamp
            every cell header and verdict carries, including the six plateau
            constants themselves.
    """

    outcome: PlateauOutcome
    window_m: int
    current: WindowCondition | None
    previous: WindowCondition | None
    windows: tuple[WindowCondition, ...]
    confirmation_count: int
    confirmed_versions: tuple[int, ...]
    reason: str | None
    snapshot_fingerprint: str
    elo_curve_fingerprint: str | None
    protocol_fingerprint: str


def _read_elo_curve_gpu_hours(run_dir: Path) -> tuple[dict[int, float], str | None]:
    """Read the on-disk ``elo_curve.json`` artifact's per-version GPU-hours join.

    Read-only, over the artifact :func:`elo_curve` itself already wrote (task 7) --
    this function never calls :func:`elo_curve` and never writes anything, so
    :func:`detect_plateau` stays a pure reader (a partial cell already excluded
    from that artifact at write time can never leak in here either).

    Args:
        run_dir: The run's root directory.

    Returns:
        ``(gpu_hours_by_version, fingerprint)``: an empty dict and ``None`` if
        ``<run_dir>/eval/elo_curve.json`` does not exist yet (every window's
        GPU-hours lookup then misses, which :func:`_window_condition` already
        turns into the insufficient-data trigger); otherwise every row's
        ``(model_version, gpu_hours)`` pair and the file's sha256
        (:func:`_file_sha256`) -- the identical fingerprint
        ``build_verdict``'s own ``elo_curve_fingerprint`` field would record
        for the same file content.
    """
    path = elo_curve_path(run_dir)
    if not path.exists():
        return {}, None
    fingerprint = _file_sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    gpu_hours_by_version = {row["model_version"]: row["gpu_hours"] for row in payload["rows"]}
    return gpu_hours_by_version, fingerprint


def _snapshot_truncated_to(snapshot: EvalSnapshot, member_prefix: int) -> EvalSnapshot:
    """Return ``snapshot`` restricted to an earlier, smaller ``member_prefix``.

    A confirmation window ending at some earlier member ``v`` must be fit from
    exactly the evidence that existed back when ``v`` was the newest evaluated
    member -- never from evidence a later candidate's own matches contributed
    (§9 pin 2's shared Bradley-Terry graph means those matches can shift an
    earlier checkpoint's rating too; see :func:`detect_plateau`'s docstring).
    Reusing :func:`_snapshot_cell_records`'s own ``candidate_version >
    snapshot.member_prefix`` filter by simply handing it a copy of ``snapshot``
    with a smaller ``member_prefix`` reconstructs exactly that earlier
    evidence set: pin 9's raw-cell immutability guarantees every cell this
    smaller prefix admits has the identical content it had back then, and no
    cell it excludes could have existed yet either (a candidate's own cells
    cannot complete before that candidate itself was evaluated).

    Args:
        snapshot: A frozen snapshot from ``core.eval_store.load_snapshot`` (or
            an already-truncated one -- truncating is idempotent/composable).
        member_prefix: The smaller prefix to restrict to. Not validated against
            ``snapshot.member_prefix`` here -- every caller in this module only
            ever truncates to a member version already known to be within the
            snapshot's real, evaluated range.

    Returns:
        A copy of ``snapshot`` with ``member_prefix`` replaced; every other
        field (including ``completed_cell_ids``, left at its full breadth --
        the downstream ``candidate_version`` filter does the actual
        narrowing) is unchanged.
    """
    return replace(snapshot, member_prefix=member_prefix)


def _fit_checkpoint_curves(
    snapshot: EvalSnapshot, seed: int, B: int
) -> tuple[dict[int, float], list[dict[int, float]]]:
    """Fit one snapshot's point-estimate and bootstrap-replicate rung-7 Elo curves.

    Shared by every window :func:`detect_plateau` builds -- the newest window
    (over ``snapshot`` exactly as read) and every earlier confirmation window
    (over a copy truncated by :func:`_snapshot_truncated_to`) -- so both cases
    go through one fit-plus-resample implementation rather than two.

    Args:
        snapshot: A frozen snapshot (full or truncated).
        seed: The run's recorded bootstrap seed (:func:`bootstrap_seed`) -- the
            same seed regardless of which snapshot is passed, since pin 7
            records one seed per run, not one per analysis; only the in-scope
            cell population differs between a full and a truncated snapshot.
        B: The bootstrap replicate count.

    Returns:
        ``(point_curve_by_version, replicate_curves_by_version)``:
        :func:`checkpoint_elo`'s ``(model_version, elo)`` pairs from
        :func:`fit_snapshot_elo`, as a dict; and that same extraction applied
        to each of :func:`bootstrap_replicates`'s ``B`` ratings dicts, in
        replicate-index order.
    """
    point_curve_by_version = dict(checkpoint_elo(fit_snapshot_elo(snapshot)))
    replicate_curves_by_version = [
        dict(checkpoint_elo(ratings)) for ratings in bootstrap_replicates(snapshot, seed, B)
    ]
    return point_curve_by_version, replicate_curves_by_version


def detect_plateau(run_dir: Path | str, *, B: int = BOOTSTRAP_B_PRODUCTION) -> PlateauResult:
    """Detect a design doc §12 M4 "profiled plateau", read-only, over the run's evidence.

    Pure library detector: reads the task-5 analysis snapshot
    (``core.eval_store.load_snapshot`` -- never the live manifest, so an on-disk
    partial cell can never influence the result, §9 pin 9) and the task-7
    ``elo_curve.json`` artifact already on disk (:func:`_read_elo_curve_gpu_hours`
    -- never recomputed or rewritten here); triggers nothing, writes nothing, and
    schedules nothing -- the six pinned constants
    (``core.eval_protocol.PLATEAU_WINDOW_M`` / ``PLATEAU_MK_ALPHA`` /
    ``PLATEAU_CI_WIDTH_THRESHOLD_ELO`` / ``PLATEAU_GPU_HOURS_MIN`` /
    ``PLATEAU_CONFIRMATION_COUNT`` and the half-window split) only ever *report*
    a tri-state verdict; every M6 lever decision and the §13 ceiling declaration
    stay human calls that *consume* this report.

    **The anti-flap confirmation clause needs ``PLATEAU_CONFIRMATION_COUNT``
    independent, point-in-time readings, but this function opens the manifest
    only once.** §9 pin 9 pins raw per-cell immutability only -- a completed
    cell's own content never changes -- it says nothing about a *derived*
    rating being invariant to later evidence, and for this protocol it is not:
    §9 pin 2's one anchored Bradley-Terry fit connects every agent in play,
    including rung-8 historical opponents that keep their own earlier rung-7
    rating's identity (``core.eval_agents.historical_opponents``'s module
    note), so a later candidate's fresh matches generically shift an earlier
    checkpoint's fitted rating too. Slicing every confirmation window out of
    one fit computed over *all* currently-scored cells would therefore let the
    newest evidence retroactively repaint what is supposed to be an
    independent, earlier reading -- exactly the flap the confirmation clause
    exists to rule out.

    Instead: the window ending at the newest evaluated member (``current``) is
    fit from the snapshot exactly as read (:func:`_fit_checkpoint_curves`).
    Every earlier confirmation window is refit from that same one-time read,
    *truncated* to its own ``member_prefix`` (:func:`_snapshot_truncated_to`,
    mirroring :func:`_snapshot_cell_records`'s existing truncation) before its
    own point estimate and bootstrap replicates are computed from scratch --
    reconstructing bit-for-bit what a standalone snapshot taken back when that
    earlier member was newest would have shown, precisely because pin 9's
    immutability guarantee means the cells a smaller prefix admits are exactly
    the cells that already existed and were already complete back then. Up to
    ``core.eval_protocol.PLATEAU_CONFIRMATION_COUNT`` such windows are built
    this way (fewer only when the evaluated series is itself too short yet for
    the full confirmation depth -- never a hardcoded two), and *every one of
    them*, not just two, must satisfy the full conjunction before PLATEAU is
    declared.

    Sub-condition machinery is reused verbatim, never re-derived: :func:`mann_kendall`
    (with its own pinned degenerate cases) restricted to each window's
    point-estimate Elo sequence; :func:`bootstrap_replicates` plus
    :func:`order_statistic_ci` (the same admissible-B rank rule as the §1 Delta)
    for the windowed contrast's CI, drawn from the run's own ``bootstrap_seed``
    for every window examined -- the same top-level seed each time (pin 7
    records one seed per run, not one per analysis), applied to a different,
    narrower cell population per earlier window's own truncated snapshot.

    Args:
        run_dir: The run's root directory -- the eval snapshot's own root, the
            root ``core.run_identity.read_stored_config`` reads ``config.json``
            from (for the evaluation seed), and the root the on-disk
            ``eval/elo_curve.json`` artifact (if any) is read from.
        B: The bootstrap replicate count backing the windowed-contrast CIs.
            Must be admissible (``B ≡ 39 mod 40``; see :func:`order_statistic_ci`).
            Defaults to the pinned production value.

    Returns:
        A :class:`PlateauResult`. ``outcome`` is
        :data:`PLATEAU_OUTCOME_INSUFFICIENT_DATA` when: fewer than
        ``PLATEAU_WINDOW_M`` members are evaluated at all; some window's
        Mann-Kendall reading is itself insufficient-data (unreachable at the
        pinned ``M`` = 8 >= 3, but never assumed away); some window's
        point-estimate Elo sequence is all-tied (:attr:`WindowCondition.
        mk_all_tied` -- Mann-Kendall's pinned sigma=0 degenerate branch, which
        reports ``insufficient_data=False`` yet carries no real trend
        information and so must never stand in for a legitimate
        non-significant reading, design doc §12 M4's insufficient-data
        clause); or some examined window has a member version with no
        GPU-hours coordinate in the elo-curve join (including the whole join
        being absent, i.e. no ``elo_curve.json`` has been written yet).
        Otherwise ``outcome`` is
        :data:`PLATEAU_OUTCOME_PLATEAU` iff the full conjunction (Mann-Kendall
        non-significant AND CI width below threshold AND GPU-hour span at or
        above the minimum) holds at every one of the
        ``PLATEAU_CONFIRMATION_COUNT`` confirmation windows built (see
        ``windows``); :data:`PLATEAU_OUTCOME_NO_PLATEAU` otherwise (including
        when it holds only at a newest prefix of them, pending confirmation,
        or fewer than ``PLATEAU_CONFIRMATION_COUNT`` windows exist yet).

    Raises:
        ValueError: If ``B`` is not admissible, or the evaluated rung-7 curve's
            versions are not exactly ``{1, ..., n_evaluated}`` (a non-contiguous
            series -- an eval-store/protocol inconsistency this function refuses
            to window over silently), or propagated from :func:`fit_snapshot_elo`
            (an agent disconnected from the anchor).
        FileNotFoundError: If ``run_dir`` has no stored ``config.json``
            (``core.run_identity.read_stored_config``).
    """
    _validate_admissible_B(B)
    run_dir = Path(run_dir)
    M = PLATEAU_WINDOW_M

    snapshot = load_snapshot(run_dir)
    point_curve = checkpoint_elo(fit_snapshot_elo(snapshot))
    n_evaluated = len(point_curve)
    gpu_hours_by_version, elo_curve_fingerprint = _read_elo_curve_gpu_hours(run_dir)

    def _insufficient(reason: str, windows: tuple[WindowCondition, ...]) -> PlateauResult:
        return PlateauResult(
            outcome=PLATEAU_OUTCOME_INSUFFICIENT_DATA,
            window_m=M,
            current=windows[0] if windows else None,
            previous=windows[1] if len(windows) > 1 else None,
            windows=windows,
            confirmation_count=0,
            confirmed_versions=(),
            reason=reason,
            snapshot_fingerprint=snapshot.snapshot_fingerprint,
            elo_curve_fingerprint=elo_curve_fingerprint,
            protocol_fingerprint=protocol_fingerprint(),
        )

    if n_evaluated < M:
        return _insufficient(
            f"only {n_evaluated} evaluated member checkpoint(s) on the snapshot's "
            f"contiguous member prefix; the plateau window requires at least {M}",
            windows=(),
        )

    all_versions = [version for version, _ in point_curve]
    if set(all_versions) != set(range(1, n_evaluated + 1)):
        raise ValueError(
            f"detect_plateau requires a complete, contiguous rung-7 curve 1..{n_evaluated}; "
            f"got versions {sorted(all_versions)}"
        )

    point_curve_by_version = dict(point_curve)
    stored_config = read_stored_config(run_dir)
    seed = bootstrap_seed(stored_config.run.evaluation.eval_seed)
    replicate_curves = [
        dict(checkpoint_elo(ratings)) for ratings in bootstrap_replicates(snapshot, seed, B)
    ]

    # Build up to PLATEAU_CONFIRMATION_COUNT windows, newest first (i == 0 is
    # "current"). i == 0 reuses the fit already computed above; every i >= 1 is
    # refit from an independently truncated snapshot (see this function's own
    # docstring) so a later window's evidence can never leak into an earlier
    # one -- never hardcoded to exactly two, so a future PLATEAU_CONFIRMATION_COUNT
    # bump changes only how many windows this loop builds.
    max_windows = min(PLATEAU_CONFIRMATION_COUNT, n_evaluated - M + 1)
    windows: list[WindowCondition] = []
    for i in range(max_windows):
        end_index = n_evaluated - 1 - i
        versions = tuple(all_versions[end_index - M + 1 : end_index + 1])
        newest_version = all_versions[end_index]
        if i == 0:
            curve_by_version, curve_replicates = point_curve_by_version, replicate_curves
        else:
            earlier_snapshot = _snapshot_truncated_to(snapshot, newest_version)
            curve_by_version, curve_replicates = _fit_checkpoint_curves(earlier_snapshot, seed, B)
        windows.append(
            _window_condition(
                newest_version,
                versions,
                curve_by_version,
                curve_replicates,
                gpu_hours_by_version,
                B,
            )
        )

    for window in windows:
        if window.mann_kendall.insufficient_data:
            return _insufficient(
                f"Mann-Kendall over the window ending at member {window.newest_version} "
                "is itself insufficient-data",
                windows=tuple(windows),
            )
        if window.mk_all_tied:
            return _insufficient(
                f"the window ending at member {window.newest_version} (versions "
                f"{window.versions}) has an all-tied point-estimate Elo sequence -- "
                "Mann-Kendall's pinned sigma=0 degenerate branch reports a p-value with no "
                "real trend information and must never stand as evidence of a plateau",
                windows=tuple(windows),
            )
        if window.gpu_hours_span is None:
            return _insufficient(
                f"the window ending at member {window.newest_version} (versions "
                f"{window.versions}) is missing a GPU-hours coordinate for at least one "
                "member in the elo-curve join -- no elo_curve.json, or it has not been "
                "refreshed to cover this member yet",
                windows=tuple(windows),
            )

    confirmation_count = 0
    confirmed: list[int] = []
    for window in windows:  # newest first; stop at the first unsatisfied window.
        if not window.satisfied:
            break
        confirmation_count += 1
        confirmed.append(window.newest_version)
    confirmed_versions = tuple(reversed(confirmed))

    current = windows[0]
    previous = windows[1] if len(windows) > 1 else None

    if confirmation_count >= PLATEAU_CONFIRMATION_COUNT:
        outcome = PLATEAU_OUTCOME_PLATEAU
        reason = None
    else:
        outcome = PLATEAU_OUTCOME_NO_PLATEAU
        if confirmation_count == 0:
            reason = f"the conjunction does not hold at the newest member {current.newest_version}"
        else:
            reason = (
                f"the conjunction holds at the newest {confirmation_count} consecutive "
                f"member checkpoint(s) ({', '.join(str(v) for v in confirmed_versions)}) but is "
                f"pending confirmation at {PLATEAU_CONFIRMATION_COUNT - confirmation_count} more "
                "consecutive snapshot(s) before PLATEAU can be declared (anti-flap clause)"
            )

    return PlateauResult(
        outcome=outcome,
        window_m=M,
        current=current,
        previous=previous,
        windows=tuple(windows),
        confirmation_count=confirmation_count,
        confirmed_versions=confirmed_versions,
        reason=reason,
        snapshot_fingerprint=snapshot.snapshot_fingerprint,
        elo_curve_fingerprint=elo_curve_fingerprint,
        protocol_fingerprint=protocol_fingerprint(),
    )
