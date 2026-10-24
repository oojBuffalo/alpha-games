"""Publish evaluation curves and verdicts from the statistical inference core."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from core.eval_protocol import (
    BOOTSTRAP_B_PRODUCTION,
    EVAL_SIMS,
    PAIRS_PER_CELL,
    PROTOCOL_VERSION,
    REGISTRY,
    protocol_fingerprint,
)
from core.eval_stats import (
    _validate_admissible_B,
    bootstrap_replicates,
    bootstrap_seed,
    checkpoint_elo,
    delta_gate,
    delta_hat,
    fit_snapshot_elo,
    mann_kendall,
    order_statistic_ci,
    per_checkpoint_ci,
    replicate_deltas,
)
from core.eval_store import EvalSnapshot, eval_dir, iter_cells, load_snapshot, read_cell
from core.observability import reduce_run
from core.replay_shard import _atomic_write, _atomic_write_json
from core.run_identity import read_stored_config

_ELO_CURVE_NAME = "elo_curve.json"


def elo_curve_path(run_dir: Path | str) -> Path:
    """Return the §1 plot series' on-disk path for one run.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``<run_dir>/eval/elo_curve.json``.
    """
    return eval_dir(run_dir) / _ELO_CURVE_NAME


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
    is recorded by :func:`build_verdict`.

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
    payload = _curve_payload(run_dir, snapshot)
    _write_blob(elo_curve_path(run_dir), _json_blob(payload))
    return payload


def _curve_payload(run_dir: Path | str, snapshot: EvalSnapshot) -> dict[str, Any]:
    """Fit and join the §1 plot series without writing it (see :func:`elo_curve`)."""
    if Path(run_dir).resolve() != Path(snapshot.run_dir).resolve():
        raise ValueError("run_dir must identify the same run as snapshot.run_dir")
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

    return {"snapshot_fingerprint": snapshot.snapshot_fingerprint, "rows": rows}


def elo_curve_artifact_path(run_dir: Path | str, fingerprint: str) -> Path:
    """Return the immutable, content-addressed copy of one published curve.

    :func:`build_verdict` references the curve it fitted by this name rather
    than by the shared, replaceable :func:`elo_curve_path`, so a later
    publisher refreshing ``elo_curve.json`` can never substitute a different
    snapshot's curve under an already-written verdict's hash. One file exists
    per distinct published snapshot (at most one per member-prefix length
    plus the empty prefix), so the set stays bounded by ``K + 1``.

    Args:
        run_dir: The run's root directory.
        fingerprint: The sha256 hex digest of the curve file's exact bytes.

    Returns:
        ``<run_dir>/eval/elo_curve-<fingerprint>.json``.

    Raises:
        ValueError: If ``fingerprint`` is not a 64-character lowercase hex
            digest (it becomes a path component).
    """
    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("elo curve fingerprint must be a 64-character lowercase hex digest")
    return eval_dir(run_dir) / f"elo_curve-{fingerprint}.json"


def _json_blob(payload: dict[str, Any]) -> bytes:
    """Serialize ``payload`` with the same convention as :func:`_atomic_write_json`."""
    return json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")


def _write_blob(path: Path, blob: bytes) -> None:
    """Atomically write already-serialized bytes (temp name, then ``os.replace``)."""
    _atomic_write(path, lambda fh: fh.write(blob))


# Verdict publication (design doc §9/§12).

_VERDICT_NAME = "verdict.json"


def verdict_path(run_dir: Path | str) -> Path:
    """Return the §12 M5.5 verdict artifact's on-disk path for one run.

    Args:
        run_dir: The run's root directory.

    Returns:
        ``<run_dir>/eval/verdict.json``.
    """
    return eval_dir(run_dir) / _VERDICT_NAME


def _production_identity(identity: str) -> bool:
    """Require bare v1 network forms; frozen network-free names stay opaque."""
    if identity.startswith(("rung5-", "rung6-", "rung7-")):
        return re.fullmatch(r"rung[567]-v1-\d+", identity) is not None
    return True


def build_verdict(run_dir: Path | str, *, B: int = BOOTSTRAP_B_PRODUCTION) -> dict[str, Any]:
    """Assemble and durably write the §12 M5.5 verdict artifact.

    The publishing orchestrator over ``core.eval_stats`` inference. Reads
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
    -member set **and** the extracted rung-7 curve covers exactly versions
    ``1..k_target`` (task 1 pin 8) -- the Delta contrast (:func:`delta_hat`,
    :func:`replicate_deltas`, :func:`order_statistic_ci`,
    :func:`delta_gate`); and runs Mann-Kendall (:func:`mann_kendall`) over the
    evaluated prefix's point-estimate curve unconditionally (reported, never
    gating, task 1 pin 7).

    Before the complete K-set exists, the artifact is provisional: it carries
    no ``delta_hat``, no Delta CI, and no gate anywhere -- only
    ``delta: null`` plus an explicit ``reason`` string -- alongside the
    per-checkpoint CIs and Mann-Kendall result for the evaluated prefix.
    An empty prefix skips fitting and bootstrap, writes an empty curve and
    checkpoint list, and reports insufficient data for Mann-Kendall.
    ``authoritative`` requires production cell settings and identities,
    the complete K-set, and ``B ==
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

    The curve is serialized once; ``elo_curve_fingerprint`` is the sha256 of
    exactly those bytes, which are written both to an immutable
    content-addressed copy (:func:`elo_curve_artifact_path`, named by
    ``elo_curve_artifact``) and to the shared, replaceable
    :func:`elo_curve_path`. A reader verifies the verdict against the
    content-addressed copy; a later publisher refreshing the shared path can
    never make a different snapshot's curve verify under this verdict's hash.

    Returns:
        The verdict payload -- the identical JSON-safe dict durably written
        (temp-name-then-``os.replace``, sorted keys, so the same records and
        seed always produce bit-identical bytes) to :func:`verdict_path`.

    Raises:
        ValueError: If ``B`` is not admissible, or propagated from
            :func:`fit_snapshot_elo`/:func:`elo_curve` (an
            agent disconnected from the anchor, or a scored member with no
            ``checkpoint_published`` marker).
        ProtocolMismatchError: If a completed cell carries non-current protocol
            stamps; propagated from :func:`load_snapshot` before publication.
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
    point_ratings = fit_snapshot_elo(snapshot) if snapshot.member_prefix else {}
    point_curve = checkpoint_elo(point_ratings)
    production_identities = all(_production_identity(name) for name in point_ratings)
    production_cells = production_identities and all(
        (
            _production_identity(header.candidate_identity)
            and _production_identity(header.opponent_identity)
            and header.eval_config.get("pairs_per_cell") == PAIRS_PER_CELL
            and header.eval_config.get("eval_sims") == EVAL_SIMS
        )
        for header, _ in (read_cell(path) for path in iter_cells(snapshot))
    )
    checkpoints_evaluated = snapshot.member_prefix
    # The member prefix alone does not prove a rung-7 rating exists for every
    # member (a member's registered cell set need not include a rung-7 cell),
    # and ``delta_hat`` takes K from ``len(curve)``. Pin 8's complete K-set
    # therefore also requires the extracted curve to be exactly 1..k_target.
    curve_versions = [version for version, _ in point_curve]
    expected_versions = list(range(1, k_target + 1))
    missing_rung7 = sorted(set(range(1, checkpoints_evaluated + 1)) - set(curve_versions))
    is_complete_k_set = checkpoints_evaluated == k_target and curve_versions == expected_versions

    replicate_ratings = (
        list(bootstrap_replicates(snapshot, seed, B)) if snapshot.member_prefix else []
    )
    elo_by_version = dict(point_curve)
    per_checkpoint_payload = [
        {"model_version": version, "elo": elo_by_version[version], "ci": [lower, upper]}
        for version, (lower, upper) in (
            per_checkpoint_ci(replicate_ratings, B) if snapshot.member_prefix else []
        )
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
        reason = (
            None if production_cells else "non-production evaluation cell settings or identities"
        )
    elif checkpoints_evaluated == k_target:
        delta_payload = None
        reason = (
            f"snapshot prefix covers all {k_target} required checkpoint(s) but the rung-7 "
            f"curve has versions {curve_versions}, not 1..{k_target} (no rung-7 rating for "
            f"member(s) {missing_rung7}) -- the Delta contrast is only ever computed over "
            "the complete K-set (task 1 pin 8)"
        )
    else:
        delta_payload = None
        reason = (
            f"snapshot prefix covers {checkpoints_evaluated} of {k_target} required "
            "checkpoint(s) -- the Delta contrast is only ever computed over the "
            "complete K-set (task 1 pin 8); no prefix Delta exists, advisory or otherwise"
        )

    curve_payload = (
        _curve_payload(run_dir, snapshot)
        if snapshot.member_prefix
        else {"snapshot_fingerprint": snapshot.snapshot_fingerprint, "rows": []}
    )
    # Hash the bytes this invocation serializes -- never a re-read of the shared
    # path, which another publisher may replace in between -- and reference an
    # immutable content-addressed copy, so the verdict's curve hash always
    # describes this verdict's own snapshot (pin 9).
    curve_blob = _json_blob(curve_payload)
    elo_curve_fingerprint = hashlib.sha256(curve_blob).hexdigest()
    curve_artifact = elo_curve_artifact_path(run_dir, elo_curve_fingerprint)
    _write_blob(curve_artifact, curve_blob)
    _write_blob(elo_curve_path(run_dir), curve_blob)

    payload: dict[str, Any] = {
        "authoritative": (is_complete_k_set and B == BOOTSTRAP_B_PRODUCTION and production_cells),
        "checkpoints_evaluated": checkpoints_evaluated,
        "k_target": k_target,
        "bootstrap_b": B,
        "bootstrap_seed": seed,
        "evidence_fingerprint": snapshot.snapshot_fingerprint,
        "elo_curve_fingerprint": elo_curve_fingerprint,
        "elo_curve_artifact": curve_artifact.name,
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
