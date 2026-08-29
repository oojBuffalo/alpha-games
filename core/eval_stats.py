"""Anchored full-ladder Elo fit over an eval snapshot, the §1 x-axis join, the
within-cell paired-bootstrap resampler, and the Delta/CI/Mann-Kendall inference
layer built on top of it (design doc §1, §9; tasks/m4/006, 007).

Four pieces, deliberately factored so the bootstrap (task 7.1) reuses the fit
without a second fit implementation, and the inference layer (task 7.2) reuses
the resampler without a second resampling implementation:

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
  gating. None of this module decides *whether* a snapshot is the complete
  K-set eligible for a Delta at all -- that gate, the provisional-vs-
  authoritative distinction, and ``verdict.json`` assembly are task 7.3's.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.elo import Match, fit_elo
from core.eval_protocol import (
    BOOTSTRAP_B_ADMISSIBLE_MODULUS,
    BOOTSTRAP_B_ADMISSIBLE_REMAINDER,
    BOOTSTRAP_B_PRODUCTION,
    BOOTSTRAP_CI_LOWER_QUANTILE,
    BOOTSTRAP_CI_UPPER_QUANTILE,
)
from core.eval_store import (
    EvalSnapshot,
    PairRecord,
    eval_dir,
    iter_cells,
    read_cell,
    records_to_match,
)
from core.observability import reduce_run
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
