"""Tests for the anchored full-ladder Elo fit and §1 x-axis join (tasks/m4/006).

Four layers:

1. **The fit's ``core.elo.fit_elo(initial_ratings=...)`` extension in isolation** --
   the default-preserving, bit-for-bit-with-None golden this task owns in
   ``core/elo.py`` (S2.3). ``tests/test_elo.py`` is never imported or modified here.
2. **``snapshot_matches``/``fit_snapshot_elo`` over hand-built eval-store fixtures** --
   a two-checkpoint, three-rung hand-computed Bradley-Terry golden (mirroring
   ``tests/test_ladder_integration.py``'s wiring and ``tests/test_elo.py``'s
   closed-form verification style), complete separation, a disconnected-agent
   fixture surfacing ``core.elo.fit_elo``'s own named error, and the
   "analysis snapshot only" scoping rule (P2.2).
3. **``checkpoint_elo``** -- provenance ordering under a shuffled ratings dict.
4. **``elo_curve``** -- the real-shape metrics-fixture join golden, round-trip
   through ``elo_curve.json``, and ordering under shuffled build order.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import re
from pathlib import Path

import pytest

from core import eval_protocol
from core.agents import RandomAgent
from core.elo import fit_elo
from core.eval_protocol import BOOTSTRAP_B_PRODUCTION
from core.eval_stats import (
    ANCHOR_AGENT,
    PLATEAU_OUTCOME_INSUFFICIENT_DATA,
    PLATEAU_OUTCOME_NO_PLATEAU,
    PLATEAU_OUTCOME_PLATEAU,
    MannKendallResult,
    PlateauResult,
    WindowCondition,
    bootstrap_replicate,
    bootstrap_replicate_matches,
    bootstrap_replicates,
    bootstrap_seed,
    build_verdict,
    checkpoint_elo,
    delta_gate,
    delta_hat,
    delta_windows,
    detect_plateau,
    elo_curve,
    elo_curve_path,
    fit_snapshot_elo,
    mann_kendall,
    order_statistic_ci,
    per_checkpoint_ci,
    replicate_deltas,
    snapshot_matches,
    verdict_path,
)
from core.eval_store import (
    CellId,
    GameRecordSnapshot,
    PairRecord,
    append_pair_record,
    build_header,
    cell_path,
    complete_cell,
    load_snapshot,
    open_cell_for_write,
    register_member,
)
from core.metrics import EpochMetricsWriter
from core.observability import (
    CHECKPOINT_PUBLISHED_KIND,
    delta_record,
    segment_end_record,
    segment_start_record,
)
from core.run_identity import (
    ENTRY_CONDITION,
    LAUNCH_SCHEMA_VERSION,
    LaunchConfig,
    RunRecord,
    write_provenance,
)
from core.runconfig import MICRO_RUN_CONFIG_PATH
from core.seeding import PURPOSE_BOOTSTRAP, derive_seed

# ---------------------------------------------------------------------------------
# Fixture helpers -- a minimal, self-contained eval-store builder (mirrors
# tests/test_eval_store.py's own helpers, kept local rather than imported so this
# file depends only on the public eval_store API tasks/m4/006 is scoped to consume).
# ---------------------------------------------------------------------------------


def _header(*, candidate_version: int, rung: int, opponent_id: str, n_pairs: int):
    return build_header(
        run_id="run",
        cell_id=CellId(candidate_version, rung, opponent_id),
        candidate_identity=f"rung{rung}-v1-{candidate_version}",
        opponent_identity=opponent_id,
        eval_config={"pairs_per_cell": n_pairs},
        candidate_fingerprint={"orientation_table_hash": "test"},
    )


def _pair_record(pair_index: int, score_a: float) -> PairRecord:
    return PairRecord(
        pair_index=pair_index,
        pair_seed=pair_index,
        score_a=score_a,
        games=(GameRecordSnapshot(plies=1, opening=0), GameRecordSnapshot(plies=1, opening=0)),
    )


def _fill(run_dir, header, pair_scores):
    path = cell_path(run_dir, header.cell_id.to_string())
    next_index = open_cell_for_write(run_dir, header)
    for i, score in enumerate(pair_scores, start=next_index):
        append_pair_record(path, _pair_record(i, score))
    return path


def _write_member(run_dir, member_version, cells):
    """Register and fully complete one member's cells.

    Args:
        run_dir: The run directory.
        member_version: The member's checkpoint version.
        cells: ``[(rung, opponent_id, pair_scores), ...]`` -- this member's
            complete required-cell set.
    """
    headers = [
        _header(candidate_version=member_version, rung=rung, opponent_id=opp, n_pairs=len(scores))
        for rung, opp, scores in cells
    ]
    register_member(run_dir, member_version, [h.cell_id.to_string() for h in headers])
    for header, (_, _, scores) in zip(headers, cells, strict=True):
        _fill(run_dir, header, scores)
        complete_cell(run_dir, header.cell_id.to_string())


def _closed_form(p: float) -> float:
    """Elo difference whose expected score is exactly ``p`` (mirrors tests/test_elo.py)."""
    return 400.0 * math.log10(p / (1.0 - p))


def _write_run_config(run_dir, *, checkpoint_count: int, eval_seed: int, run_id: str = "test-run"):
    """Write a valid ``config.json`` a ``build_verdict`` call can read K/eval_seed from.

    Starts from the real pinned micro-Blokus config (already game/schema-valid)
    and overrides only ``training.checkpoint_count`` and ``evaluation.eval_seed``
    -- the two scalars ``build_verdict`` itself reads -- plus the launcher-only
    fields ``LaunchConfig`` requires. Never touches ``core/run_identity.py``;
    only exercises its already-public ``write_provenance``/``LaunchConfig`` API.
    """
    raw = json.loads(MICRO_RUN_CONFIG_PATH.read_text())
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    raw["training"] = dict(raw["training"])
    raw["training"]["checkpoint_count"] = checkpoint_count
    raw["evaluation"] = dict(raw["evaluation"])
    raw["evaluation"]["eval_seed"] = eval_seed
    raw["num_actors"] = 1
    raw["device"] = "cpu"
    raw["schema_version"] = LAUNCH_SCHEMA_VERSION
    raw["runtime"] = {
        "refresh_poll_interval": 1.0,
        "pacing_poll_interval": 1.0,
        "ceiling_poll_interval": 1.0,
    }
    launch_config = LaunchConfig.from_dict(raw)
    record = RunRecord(
        run_id=run_id, created_at="2026-01-01T00:00:00Z", entry_condition=ENTRY_CONDITION
    )
    write_provenance(run_dir, launch_config, record)


def _write_checkpoint_markers(run_dir, versions):
    """Write one ``checkpoint_published`` marker per member version.

    The minimal metrics fixture ``elo_curve``/``build_verdict`` need:
    ``core.observability.reduce_run`` requires no GPU segments or actor deltas
    to compute ``net_evals``/``gpu_hours`` -- both default to ``0.0`` with none
    on disk -- only a marker per scored member version.
    """
    learner = EpochMetricsWriter(run_dir, "learner")
    for i, version in enumerate(sorted(versions), start=1):
        _append_checkpoint_published(
            learner, version=version, learner_step=10 * i, timestamp=float(i)
        )


def _find_key(obj, key: str) -> bool:
    """Recursively search a JSON-shaped ``dict``/``list`` tree for ``key``.

    A plain key search, never a substring search over serialized text (which
    would false-positive on an unrelated word like "aggregate").
    """
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_find_key(value, key) for value in obj.values())
    if isinstance(obj, list):
        return any(_find_key(item, key) for item in obj)
    return False


# ==============================================================================
# 1. core.elo.fit_elo(initial_ratings=...) in isolation (S2.3)
# ==============================================================================


def test_initial_ratings_none_is_bit_for_bit_identical_to_the_old_zero_start_path():
    matches = [("a", "b", 30.0, 40), ("b", "c", 12.0, 40)]
    agents = sorted({name for m in matches for name in (m[0], m[1])})

    omitted = fit_elo(matches, anchor="c")
    explicit_none = fit_elo(matches, anchor="c", initial_ratings=None)
    explicit_zero_dict = fit_elo(matches, anchor="c", initial_ratings=dict.fromkeys(agents, 0.0))

    assert omitted == explicit_none == explicit_zero_dict  # exact, not approximate


def test_initial_ratings_warm_start_converges_to_the_zero_start_fixed_point():
    matches = [("a", "b", 30.0, 40), ("b", "c", 12.0, 40)]
    cold = fit_elo(matches, anchor="c")
    warm = fit_elo(matches, anchor="c", initial_ratings={"a": 900.0, "b": -450.0, "c": 200.0})
    assert warm.keys() == cold.keys()
    for name in cold:
        assert warm[name] == pytest.approx(cold[name], abs=1e-6)
    assert warm["c"] == 0.0  # anchor pinned regardless of the (ignored) warm value


def test_initial_ratings_defaults_missing_agents_to_zero():
    matches = [("a", "b", 30.0, 40), ("b", "c", 12.0, 40)]
    cold = fit_elo(matches, anchor="c")
    partially_warm = fit_elo(matches, anchor="c", initial_ratings={"a": 900.0})
    for name in cold:
        assert partially_warm[name] == pytest.approx(cold[name], abs=1e-6)


def test_initial_ratings_ignores_keys_outside_the_matchup_graph():
    matches = [("a", "b", 30.0, 40)]
    ratings = fit_elo(matches, anchor="b", initial_ratings={"nonexistent-agent": 12345.0})
    assert ratings == fit_elo(matches, anchor="b")


def test_initial_ratings_ignores_an_invalid_anchor_value():
    # The docstring promises the anchor's supplied value is "always forced to 0.0
    # regardless of what (if anything) initial_ratings supplies for it" -- a
    # non-finite or non-numeric anchor value must be silently discarded, not
    # validated, since it is never actually used.
    matches = [("a", "b", 30.0, 40)]
    cold = fit_elo(matches, anchor="b")

    nan_anchor = fit_elo(matches, anchor="b", initial_ratings={"b": float("nan")})
    assert nan_anchor == cold
    assert nan_anchor["b"] == 0.0

    inf_anchor = fit_elo(matches, anchor="b", initial_ratings={"b": float("inf")})
    assert inf_anchor == cold

    non_numeric_anchor = fit_elo(matches, anchor="b", initial_ratings={"b": "not-a-number"})
    assert non_numeric_anchor == cold


def test_initial_ratings_rejects_non_finite_values():
    matches = [("a", "b", 30.0, 40)]
    with pytest.raises(ValueError):
        fit_elo(matches, anchor="b", initial_ratings={"a": float("nan")})
    with pytest.raises(ValueError):
        fit_elo(matches, anchor="b", initial_ratings={"a": float("inf")})


def test_initial_ratings_rejects_non_numeric_values():
    matches = [("a", "b", 30.0, 40)]
    with pytest.raises(ValueError):
        fit_elo(matches, anchor="b", initial_ratings={"a": "500"})
    with pytest.raises(ValueError):
        fit_elo(matches, anchor="b", initial_ratings={"a": True})


# ==============================================================================
# 2. snapshot_matches / fit_snapshot_elo over eval-store fixtures
# ==============================================================================


def test_anchor_agent_matches_the_frozen_ladder_rung_1_identity():
    assert ANCHOR_AGENT == RandomAgent(seed=0).name


def test_two_checkpoint_three_rung_fixture_matches_hand_computed_bt_ratings(tmp_path):
    """Each of two checkpoints gets 3 rung cells (5, 6, 7), all vs the anchor only --
    a pure star graph, so every leaf's rating is exactly the 2-agent closed form
    (virtual draw included), independent of every other leaf. Mirrors
    tests/test_ladder_integration.py's real-runner-through-anchored-fit wiring, but
    through the eval-store cell path this task adds.
    """
    # v1: rung5 all-wins (score 6/6), rung6 all-draws (score 2/4), rung7 mixed (2.5/4)
    _write_member(
        tmp_path,
        1,
        [
            (5, "random", [2.0, 2.0, 2.0]),
            (6, "random", [1.0, 1.0]),
            (7, "random", [1.5, 1.0]),
        ],
    )
    # v2: rung5 all-losses (0/4), rung6 all-wins (8/8), rung7 a single draw (1/2)
    _write_member(
        tmp_path,
        2,
        [
            (5, "random", [0.0, 0.0]),
            (6, "random", [2.0, 2.0, 2.0, 2.0]),
            (7, "random", [1.0]),
        ],
    )

    snapshot = load_snapshot(tmp_path)
    assert snapshot.member_prefix == 2
    ratings = fit_snapshot_elo(snapshot)

    assert ratings[ANCHOR_AGENT] == 0.0
    assert ratings["rung5-v1-1"] == pytest.approx(_closed_form(6.5 / 7.0), abs=1e-4)
    assert ratings["rung6-v1-1"] == pytest.approx(0.0, abs=1e-6)
    assert ratings["rung7-v1-1"] == pytest.approx(_closed_form(3.0 / 5.0), abs=1e-4)
    assert ratings["rung5-v1-2"] == pytest.approx(_closed_form(0.5 / 5.0), abs=1e-4)
    assert ratings["rung6-v1-2"] == pytest.approx(_closed_form(8.5 / 9.0), abs=1e-4)
    assert ratings["rung7-v1-2"] == pytest.approx(0.0, abs=1e-6)

    assert checkpoint_elo(ratings) == [
        (1, pytest.approx(ratings["rung7-v1-1"])),
        (2, pytest.approx(ratings["rung7-v1-2"])),
    ]


def test_all_wins_cell_is_complete_separation_and_fits_finite(tmp_path):
    _write_member(tmp_path, 1, [(7, "random", [2.0] * 12)])  # 24/24, a shutout
    snapshot = load_snapshot(tmp_path)

    ratings = fit_snapshot_elo(snapshot)

    assert math.isfinite(ratings["rung7-v1-1"])
    assert ratings["rung7-v1-1"] == pytest.approx(_closed_form(24.5 / 25.0), abs=1e-4)
    assert ratings["rung7-v1-1"] > 400.0  # a real, large, but finite separation


def test_one_unplayed_cell_raises_naming_the_disconnected_agent(tmp_path):
    # v1 connects to the anchor directly.
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.0])])
    # v2's only played cell is vs "largest-piece" -- the vs-anchor cell for v2 was
    # simply never scheduled/played, so v2 and largest-piece form an isolated
    # component with no path back to "random".
    _write_member(tmp_path, 2, [(7, "largest-piece", [1.0, 1.0])])

    snapshot = load_snapshot(tmp_path)
    with pytest.raises(ValueError, match="not connected to the anchor") as exc_info:
        fit_snapshot_elo(snapshot)
    message = str(exc_info.value)
    assert "largest-piece" in message
    assert "rung7-v1-2" in message


def test_snapshot_matches_excludes_cells_outside_the_contiguous_member_prefix(tmp_path):
    # Member 1: complete.
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.0])])

    # Member 2: a hole -- one of its two required cells never completes.
    header_random = _header(candidate_version=2, rung=7, opponent_id="random", n_pairs=2)
    header_lp = _header(candidate_version=2, rung=7, opponent_id="largest-piece", n_pairs=2)
    register_member(tmp_path, 2, [header_random.cell_id.to_string(), header_lp.cell_id.to_string()])
    _fill(tmp_path, header_random, [1.0, 1.0])  # opened, appended, never completed
    _fill(tmp_path, header_lp, [1.0, 1.0])
    complete_cell(tmp_path, header_lp.cell_id.to_string())  # only this one completes

    # Member 3: fully complete in its own right (plays v1's historical rung-7 form),
    # despite the hole at member 2 -- real, completed evidence, but outside the
    # contiguous authoritative prefix (EvalSnapshot.member_prefix's own docstring).
    _write_member(tmp_path, 3, [(7, "rung7-v1-1", [1.0, 1.0])])
    member3_cid = _header(
        candidate_version=3, rung=7, opponent_id="rung7-v1-1", n_pairs=2
    ).cell_id.to_string()

    snapshot = load_snapshot(tmp_path)
    assert snapshot.member_prefix == 1
    # Member 2's largest-piece cell and member 3's cell are both real, completed
    # evidence, visible in completed_cell_ids despite sitting outside the prefix...
    assert header_lp.cell_id.to_string() in snapshot.completed_cell_ids
    assert member3_cid in snapshot.completed_cell_ids
    assert header_random.cell_id.to_string() not in snapshot.completed_cell_ids

    matches = snapshot_matches(snapshot)
    agents = {name for m in matches for name in (m[0], m[1])}
    # ...but neither perturbs the fit: only member 1's cell is in scope.
    assert agents == {ANCHOR_AGENT, "rung7-v1-1"}

    ratings = fit_snapshot_elo(snapshot)
    assert set(ratings) == {ANCHOR_AGENT, "rung7-v1-1"}


def test_an_unregistered_on_disk_cell_file_perturbs_nothing(tmp_path):
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.0])])
    baseline = fit_snapshot_elo(load_snapshot(tmp_path))

    # A stray cell file written directly to disk, never registered in the manifest --
    # structurally invisible to load_snapshot (P2.2), so it must not reach the fit.
    stray_header = _header(candidate_version=9, rung=7, opponent_id="mobility", n_pairs=2)
    _fill(tmp_path, stray_header, [2.0, 2.0])  # never register_member'd, never completed

    after = fit_snapshot_elo(load_snapshot(tmp_path))
    assert after == baseline


# ==============================================================================
# 3. checkpoint_elo: provenance ordering
# ==============================================================================


def test_checkpoint_elo_orders_by_model_version_under_a_shuffled_ratings_dict():
    ratings = {
        "rung7-v1-10": 50.0,
        ANCHOR_AGENT: 0.0,
        "rung7-v1-2": -30.0,
        "largest-piece": 12.0,
        "rung5-v1-2": 5.0,  # not rung-7 -- excluded
        "rung7-v1-1": 10.0,
    }
    assert checkpoint_elo(ratings) == [
        (1, 10.0),
        (2, -30.0),
        (10, 50.0),  # numeric order, never lexical ("10" before "2" would be wrong)
    ]


def test_checkpoint_elo_is_empty_when_no_rung_7_agent_is_present():
    assert checkpoint_elo({ANCHOR_AGENT: 0.0, "largest-piece": 5.0}) == []


# ==============================================================================
# 4. elo_curve: the real-shape metrics-fixture join, round-trip, ordering
# ==============================================================================


def _append_checkpoint_published(writer, *, version, learner_step, timestamp):
    writer.append(
        {
            "kind": CHECKPOINT_PUBLISHED_KIND,
            "model_version": version,
            "learner_step": learner_step,
            "timestamp": timestamp,
        }
    )


def _build_real_shape_metrics_fixture(run_dir):
    """Per-process epoch files, asynchronous actor flushes before/after each
    publish, and an epoch restart mid-series -- built the way
    tests/test_observability.py's own golden builds its fixture, not a
    pre-reduced synthetic table.

    Hand arithmetic (mirrors tests/test_observability.py's inline comments):
      v1 marker @t=2.0: positions before it = 10(t=1, actor-0) + 7(t=1.5, actor-1) = 17
                        gpu segment #1 (0.0-2.5) not yet closed at t=2.0 -> 0.0h
      v2 marker @t=5.0: positions before it = 17 + 15(t=3, actor-0 ep0)
                        + 5(t=4, actor-0 ep1, after its restart) + 3(t=4.5, actor-1) = 40
                        gpu segment #1 closed at t=2.5 -> 2.5/3600 h
    """
    orch = EpochMetricsWriter(run_dir, "orchestrator")
    a0 = EpochMetricsWriter(run_dir, "actor-0")
    a1 = EpochMetricsWriter(run_dir, "actor-1")
    learner = EpochMetricsWriter(run_dir, "learner")

    orch.append(segment_start_record(device="cpu", timestamp=0.0))
    a0.append(delta_record("positions_evaluated", 10, timestamp=1.0))
    a1.append(delta_record("positions_evaluated", 7, timestamp=1.5))
    _append_checkpoint_published(learner, version=1, learner_step=5, timestamp=2.0)
    orch.append(segment_end_record(timestamp=2.5))
    a0.append(delta_record("positions_evaluated", 15, timestamp=3.0))
    a0_restarted = EpochMetricsWriter(run_dir, "actor-0")  # a crash + restart mid-series
    assert a0_restarted.epoch == 1
    a0_restarted.append(delta_record("positions_evaluated", 5, timestamp=4.0))
    a1.append(delta_record("positions_evaluated", 3, timestamp=4.5))
    _append_checkpoint_published(learner, version=2, learner_step=9, timestamp=5.0)


def test_elo_curve_joins_correct_cumulative_x_values_on_a_real_shape_metrics_fixture(tmp_path):
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.0])])
    _write_member(tmp_path, 2, [(7, "random", [1.0, 1.0])])
    _build_real_shape_metrics_fixture(tmp_path)

    snapshot = load_snapshot(tmp_path)
    ratings = fit_snapshot_elo(snapshot)
    result = elo_curve(tmp_path, snapshot)

    assert result["snapshot_fingerprint"] == snapshot.snapshot_fingerprint
    rows = result["rows"]
    assert [r["model_version"] for r in rows] == [1, 2]
    assert rows[0]["elo"] == ratings["rung7-v1-1"]
    assert rows[1]["elo"] == ratings["rung7-v1-2"]

    assert rows[0]["learner_step"] == 5
    assert rows[0]["net_evals"] == pytest.approx(17.0)
    assert rows[0]["gpu_hours"] == pytest.approx(0.0)

    assert rows[1]["learner_step"] == 9
    assert rows[1]["net_evals"] == pytest.approx(40.0)
    assert rows[1]["gpu_hours"] == pytest.approx(2.5 / 3600.0)


def test_elo_curve_raises_when_a_scored_member_has_no_publication_marker(tmp_path):
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.0])])
    # No metrics/ directory at all -- reduce_run's checkpoints mapping is empty.
    snapshot = load_snapshot(tmp_path)
    with pytest.raises(ValueError, match="checkpoint_published marker"):
        elo_curve(tmp_path, snapshot)


def test_elo_curve_json_round_trips_and_is_durably_written(tmp_path):
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.0])])
    _write_member(tmp_path, 2, [(7, "random", [1.0, 1.0])])
    _build_real_shape_metrics_fixture(tmp_path)
    snapshot = load_snapshot(tmp_path)

    result = elo_curve(tmp_path, snapshot)

    on_disk = json.loads(elo_curve_path(tmp_path).read_text(encoding="utf-8"))
    assert on_disk == result
    assert elo_curve_path(tmp_path) == tmp_path / "eval" / "elo_curve.json"


def test_elo_curve_orders_by_model_version_regardless_of_build_order(tmp_path):
    # Build member 2 before member 1 -- ordering must still come out ascending.
    _write_member(tmp_path, 2, [(7, "random", [1.0, 1.0])])
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.0])])
    _build_real_shape_metrics_fixture(tmp_path)

    snapshot = load_snapshot(tmp_path)
    result = elo_curve(tmp_path, snapshot)

    assert [row["model_version"] for row in result["rows"]] == [1, 2]


# ==============================================================================
# 5. The within-cell paired-bootstrap resampler (tasks/m4/007, subtask 7.1)
# ==============================================================================


def _write_bootstrap_fixture(tmp_path):
    """A small multi-cell, non-uniform-pair-count eval-store fixture.

    Deliberately gives cells different original pair counts (4, 3, 4) so
    "resample with replacement to the cell's own count" is exercised
    meaningfully rather than vacuously with a single shared count everywhere.
    """
    _write_member(
        tmp_path,
        1,
        [
            (5, "random", [2.0, 1.0, 0.0, 2.0]),
            (7, "random", [1.5, 1.0, 2.0]),
        ],
    )
    _write_member(
        tmp_path,
        2,
        [
            (7, "random", [1.0, 0.5, 2.0, 1.5]),
        ],
    )


def test_bootstrap_seed_matches_the_pinned_derivation():
    assert bootstrap_seed(4242) == derive_seed(4242, PURPOSE_BOOTSTRAP)
    assert bootstrap_seed(4242) != bootstrap_seed(4243)


def test_replicate_reproduced_in_isolation_matches_the_full_run_exactly(tmp_path):
    """Task 7.1's headline reproducibility guarantee: replicate b is recoverable
    from (bootstrap_seed, b) alone, with no other replicate having ever run, and
    is bit-for-bit identical to that same index inside a full B-replicate batch.
    """
    _write_bootstrap_fixture(tmp_path)
    snapshot = load_snapshot(tmp_path)
    seed = bootstrap_seed(4242)

    full_run = list(bootstrap_replicates(snapshot, seed, 5))

    for b in range(5):
        isolated = bootstrap_replicate(snapshot, seed, b)
        assert isolated == full_run[b]  # exact, not approximate


def test_different_replicate_indices_give_different_resamples(tmp_path):
    _write_bootstrap_fixture(tmp_path)
    snapshot = load_snapshot(tmp_path)
    seed = bootstrap_seed(11)

    matches_by_b = [bootstrap_replicate_matches(snapshot, seed, b) for b in range(8)]
    # Each replicate names the identical set of matchups (resampling never
    # changes which candidate/opponent pairs appear -- only their scores) --
    # but at least one replicate's aggregate scores must differ from another's,
    # or the resampler is silently ignoring the seed.
    agent_pairs = {(m[0], m[1]) for matches in matches_by_b for m in matches}
    assert len(agent_pairs) == 3  # the fixture's 3 cells: rung5-v1-1, rung7-v1-1, rung7-v1-2
    assert len({tuple(sorted(m[2] for m in matches)) for matches in matches_by_b}) > 1


def test_warm_start_replicate_refit_matches_cold_start_within_fit_tolerance(tmp_path):
    """Warm-starting from the point estimate (task 6's `initial_ratings`) must be a
    pure speedup to the same anchored fixed point, never a different answer
    (`core.elo.fit_elo`'s own docstring; verified directly for `fit_elo` in
    `tests/test_eval_stats.py`'s own `initial_ratings` suite). Exact bit-for-bit
    equality is deliberately not asserted here: cold and warm starts drive
    `fit_elo`'s bisection from different initial `(lo, hi)` brackets, so each
    per-agent search takes a different sequence of floating-point midpoints
    before its per-sweep move drops below `fit_elo`'s own convergence tolerance
    (`tol=1e-9`) -- the two fits converge on the same value to within roughly
    that tolerance, not through the identical arithmetic path that would be
    needed for bit-identical output.
    """
    _write_bootstrap_fixture(tmp_path)
    snapshot = load_snapshot(tmp_path)
    seed = bootstrap_seed(777)
    b = 2

    matches = bootstrap_replicate_matches(snapshot, seed, b)
    cold = fit_elo(matches, anchor=ANCHOR_AGENT)
    warm = bootstrap_replicate(snapshot, seed, b)

    assert warm.keys() == cold.keys()
    assert warm[ANCHOR_AGENT] == 0.0 == cold[ANCHOR_AGENT]
    for name in cold:
        assert warm[name] == pytest.approx(cold[name], abs=1e-6)


def test_same_store_and_seed_give_bit_identical_replicate_ratings_across_two_runs(tmp_path):
    _write_bootstrap_fixture(tmp_path)
    seed = bootstrap_seed(99)

    run_a = list(bootstrap_replicates(load_snapshot(tmp_path), seed, 7))
    run_b = list(bootstrap_replicates(load_snapshot(tmp_path), seed, 7))

    assert run_a == run_b  # exact, not approximate -- two independent snapshot loads


# ==============================================================================
# 6. Delta-hat, the admissible-B order-statistic CIs, per-checkpoint CIs, and
#    Mann-Kendall (tasks/m4/007, subtask 7.2)
# ==============================================================================


# --- delta_windows: the ceil(K/3) boundary sets ------------------------------------


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (3, ((1,), (3,))),
        (4, ((1, 2), (3, 4))),
        (5, ((1, 2), (4, 5))),
    ],
)
def test_delta_windows_documented_boundary_sets(k, expected):
    assert delta_windows(k) == expected


def test_delta_windows_rejects_non_positive_k():
    with pytest.raises(ValueError):
        delta_windows(0)
    with pytest.raises(ValueError):
        delta_windows(-1)


# --- delta_hat: the original-sample statistic --------------------------------------


def test_delta_hat_matches_hand_computed_window_means():
    # K=5 -> windows (1,2) and (4,5); mean(40,50) - mean(10,20) = 45 - 15 = 30.
    curve = [(1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0), (5, 50.0)]
    assert delta_hat(curve) == pytest.approx(30.0)


def test_delta_hat_is_indifferent_to_curve_order():
    shuffled = [(3, 30.0), (1, 10.0), (5, 50.0), (2, 20.0), (4, 40.0)]
    assert delta_hat(shuffled) == pytest.approx(30.0)


def test_delta_hat_rejects_empty_curve():
    with pytest.raises(ValueError):
        delta_hat([])


def test_delta_hat_rejects_a_non_contiguous_or_incomplete_series():
    # Only versions {1, 3} present -- not the complete, contiguous 1..K set
    # (task 1 pin 8: no prefix Delta is ever computed).
    with pytest.raises(ValueError, match="complete, contiguous"):
        delta_hat([(1, 10.0), (3, 30.0)])


# --- replicate_deltas: Delta_b mapped over each replicate's own curve --------------


def test_replicate_deltas_maps_delta_hat_over_each_replicates_ratings():
    replicate_ratings = [
        {
            ANCHOR_AGENT: 0.0,
            "rung7-v1-1": 0.0,
            "rung7-v1-2": 10.0,
            "rung7-v1-3": 20.0,
            "rung7-v1-4": 30.0,
            "rung7-v1-5": 40.0,
        },
        {
            ANCHOR_AGENT: 0.0,
            "rung7-v1-1": 100.0,
            "rung7-v1-2": 100.0,
            "rung7-v1-3": 100.0,
            "rung7-v1-4": 100.0,
            "rung7-v1-5": 100.0,
        },
    ]
    # Replicate 0: windows (1,2)/(4,5) -> mean(30,40) - mean(0,10) = 35 - 5 = 30.
    # Replicate 1: every version tied at 100 -> Delta = 0.
    assert replicate_deltas(replicate_ratings) == [pytest.approx(30.0), pytest.approx(0.0)]


# --- order_statistic_ci: the admissible-B rank rule --------------------------------


def test_order_statistic_ci_at_b39_picks_ranks_1_and_39_with_ties():
    # 39 values: five ties at the minimum (0), then 1..34 -- deliberately
    # unsorted on input. Ranks (B+1)*0.025=1 and (B+1)*0.975=39 are exactly the
    # overall min and max of the 39-element sample.
    values = list(range(1, 35)) + [0.0] * 5
    assert len(values) == 39
    assert order_statistic_ci(values, 39) == (0.0, 34.0)


def test_order_statistic_ci_rejects_non_admissible_b():
    with pytest.raises(ValueError):
        order_statistic_ci(list(range(100)), 100)
    with pytest.raises(ValueError):
        order_statistic_ci(list(range(2000)), 2000)


def test_order_statistic_ci_rejects_wrong_length_input():
    with pytest.raises(ValueError):
        order_statistic_ci(list(range(10)), 39)


def test_order_statistic_ci_defaults_to_the_pinned_production_b():
    default_b = inspect.signature(order_statistic_ci).parameters["B"].default
    assert default_b == BOOTSTRAP_B_PRODUCTION == 1999


def test_1999_is_admissible_and_2000_is_not():
    # Production B: ranks (1999+1)*0.025=50, (1999+1)*0.975=1950 -- both integral.
    order_statistic_ci([0.0] * 1999, 1999)  # must not raise
    with pytest.raises(ValueError):
        order_statistic_ci([0.0] * 2000, 2000)


# --- delta_gate: strictly-above-0 -----------------------------------------------


@pytest.mark.parametrize(
    ("ci", "expected"),
    [
        ((0.5, 10.0), True),
        ((-0.1, 10.0), False),
        ((0.0, 5.0), False),  # exactly 0 is not "strictly above"
    ],
)
def test_delta_gate(ci, expected):
    assert delta_gate(ci) is expected


# --- per_checkpoint_ci: the same rule, same replicates, no second resample --------


def test_per_checkpoint_ci_applies_the_same_rule_column_wise_no_second_resample():
    replicate_ratings = [
        {ANCHOR_AGENT: 0.0, "rung7-v1-2": float(2 * b), "rung7-v1-1": float(b)} for b in range(39)
    ]
    result = per_checkpoint_ci(replicate_ratings, 39)
    # v1's column is 0..38 (min/max at ranks 1/39); v2's is 0,2,...,76.
    assert result == [(1, (0.0, 38.0)), (2, (0.0, 76.0))]


def test_per_checkpoint_ci_rejects_wrong_replicate_count():
    replicate_ratings = [{ANCHOR_AGENT: 0.0, "rung7-v1-1": 0.0}] * 10
    with pytest.raises(ValueError):
        per_checkpoint_ci(replicate_ratings, 39)


def test_per_checkpoint_ci_defaults_to_the_pinned_production_b():
    default_b = inspect.signature(per_checkpoint_ci).parameters["B"].default
    assert default_b == BOOTSTRAP_B_PRODUCTION


# --- mann_kendall: classic S, tie/continuity-corrected z, two-sided p -------------


def test_mann_kendall_golden_without_ties():
    # values [1, 3, 2, 5]: pairwise signs +,+,+,-,+,+ -> S = 4.
    # variance = (4*3*13)/18 = 8.666...7; sigma = 2.943920288775949.
    # z = (4 - 1)/sigma = 1.0190493307301363; p = erfc(z/sqrt(2)).
    result = mann_kendall([1.0, 3.0, 2.0, 5.0])
    assert result.insufficient_data is False
    assert result.n == 4
    assert result.s == 4
    assert result.z == pytest.approx(1.0190493307301363)
    assert result.p == pytest.approx(0.308179547467054)


def test_mann_kendall_golden_with_ties():
    # values [1, 2, 2, 3]: pairwise signs +,+,+,0,+,+ -> S = 5.
    # tie term = 2*1*9 = 18; variance = (156 - 18)/18 = 7.666...7.
    result = mann_kendall([1.0, 2.0, 2.0, 3.0])
    assert result.insufficient_data is False
    assert result.n == 4
    assert result.s == 5
    assert result.z == pytest.approx(1.4446302370292303)
    assert result.p == pytest.approx(0.1485617748918687)


def test_mann_kendall_golden_with_ties_and_a_downward_trend():
    # The mirror image of the previous case: S and z flip sign, p is unchanged.
    result = mann_kendall([3.0, 2.0, 2.0, 1.0])
    assert result.s == -5
    assert result.z == pytest.approx(-1.4446302370292303)
    assert result.p == pytest.approx(0.1485617748918687)


def test_mann_kendall_all_tied_forces_s_zero_z_zero_p_one():
    result = mann_kendall([7.0, 7.0, 7.0, 7.0, 7.0])
    assert result.insufficient_data is False
    assert result.n == 5
    assert result.s == 0
    assert result.z == 0.0
    assert result.p == 1.0


def test_mann_kendall_below_three_points_is_insufficient_data():
    result = mann_kendall([1.0, 2.0])
    assert result == MannKendallResult(n=2, insufficient_data=True, s=None, z=None, p=None)


def test_mann_kendall_never_claims_a_trend_when_insufficient():
    result = mann_kendall([1.0])
    assert result.insufficient_data is True
    assert result.s is None
    assert result.z is None
    assert result.p is None


# ==============================================================================
# 7. verdict.json assembly (tasks/m4/007, subtask 7.3)
# ==============================================================================


def test_build_verdict_is_bit_identical_across_two_independently_built_runs(tmp_path):
    """Same records + seed -> bit-identical verdict.json bytes (task 1 pin 7's
    determinism discipline, at the whole-artifact grain)."""

    def _build(root):
        for version in (1, 2, 3):
            _write_member(root, version, [(7, "random", [1.0, 1.5, 0.5, 2.0])])
        _write_checkpoint_markers(root, [1, 2, 3])
        _write_run_config(root, checkpoint_count=3, eval_seed=4242)
        return build_verdict(root, B=39)

    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    payload_a = _build(root_a)
    payload_b = _build(root_b)

    assert payload_a == payload_b
    assert verdict_path(root_a).read_bytes() == verdict_path(root_b).read_bytes()


def test_build_verdict_writes_exactly_what_it_returns(tmp_path):
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.0])])
    _write_checkpoint_markers(tmp_path, [1])
    _write_run_config(tmp_path, checkpoint_count=1, eval_seed=5)

    payload = build_verdict(tmp_path, B=39)

    assert verdict_path(tmp_path) == tmp_path / "eval" / "verdict.json"
    on_disk = json.loads(verdict_path(tmp_path).read_text(encoding="utf-8"))
    assert on_disk == payload


def test_build_verdict_partial_k_snapshot_yields_null_delta_and_no_gate_key(tmp_path):
    """The no-prefix-Delta golden (task 1 pin 8): a partial-K snapshot carries
    per-checkpoint CIs and MK, but delta is null with a reason, and the "gate"
    key exists nowhere in the artifact -- not nested, not advisory."""
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.5, 0.5, 2.0])])
    _write_member(tmp_path, 2, [(7, "random", [1.5, 1.0, 2.0, 0.5])])
    _write_checkpoint_markers(tmp_path, [1, 2])
    _write_run_config(tmp_path, checkpoint_count=3, eval_seed=99)  # K=3, only 2 scored

    payload = build_verdict(tmp_path, B=39)

    assert payload["checkpoints_evaluated"] == 2
    assert payload["k_target"] == 3
    assert payload["authoritative"] is False
    assert payload["delta"] is None
    assert isinstance(payload["reason"], str) and payload["reason"]
    assert [row["model_version"] for row in payload["per_checkpoint"]] == [1, 2]
    assert payload["mann_kendall"]["insufficient_data"] is True  # n=2 < 3
    assert not _find_key(payload, "gate")


def test_build_verdict_complete_k_set_carries_delta_ci_and_gate(tmp_path):
    _write_member(tmp_path, 1, [(7, "random", [0.0, 0.0, 0.5, 0.0])])
    _write_member(tmp_path, 2, [(7, "random", [2.0, 2.0, 1.5, 2.0])])
    _write_checkpoint_markers(tmp_path, [1, 2])
    _write_run_config(tmp_path, checkpoint_count=2, eval_seed=17)

    payload = build_verdict(tmp_path, B=39)

    assert payload["checkpoints_evaluated"] == payload["k_target"] == 2
    assert payload["delta"] is not None
    assert set(payload["delta"]) == {"delta_hat", "ci", "gate"}
    assert payload["delta"]["ci"][0] <= payload["delta"]["delta_hat"] <= payload["delta"]["ci"][1]
    assert payload["reason"] is None


def test_build_verdict_on_disk_partial_cell_perturbs_nothing(tmp_path):
    """The bootstrap consumes only snapshot cells: a cell that is merely opened
    and appended -- never completed -- must change nothing (task 1 pin 9 /
    P2.2), including at the whole-verdict grain."""
    _write_member(tmp_path, 1, [(7, "random", [1.0, 1.5, 0.5, 2.0])])
    _write_member(tmp_path, 2, [(7, "random", [1.5, 1.0, 2.0, 0.5])])
    _write_checkpoint_markers(tmp_path, [1, 2])
    _write_run_config(tmp_path, checkpoint_count=2, eval_seed=123)

    build_verdict(tmp_path, B=39)
    baseline_bytes = verdict_path(tmp_path).read_bytes()

    # A stray, never-completed member-3 cell -- structurally "scheduled" but not
    # yet complete -- sits entirely outside every in-scope cell set.
    header = _header(candidate_version=3, rung=7, opponent_id="random", n_pairs=2)
    register_member(tmp_path, 3, [header.cell_id.to_string()])
    _fill(tmp_path, header, [1.0, 1.0])  # opened + appended, never completed

    build_verdict(tmp_path, B=39)
    assert verdict_path(tmp_path).read_bytes() == baseline_bytes


def test_build_verdict_authoritative_requires_complete_k_set_and_b_1999(tmp_path):
    """authoritative flips only when the last required cell of the last member
    completes, and only at B=1999: a complete K-set at B=39 is non-authoritative
    but still carries Delta and CI; an incomplete K-set at B=1999 stays
    non-authoritative regardless."""
    run_complete = tmp_path / "complete"
    _write_member(run_complete, 1, [(7, "random", [1.0, 1.0])])
    _write_checkpoint_markers(run_complete, [1])
    _write_run_config(run_complete, checkpoint_count=1, eval_seed=7)

    at_39 = build_verdict(run_complete, B=39)
    assert at_39["authoritative"] is False
    assert at_39["delta"] is not None

    at_1999 = build_verdict(run_complete, B=1999)
    assert at_1999["authoritative"] is True
    assert at_1999["delta"] is not None

    run_partial = tmp_path / "partial"
    _write_member(run_partial, 1, [(7, "random", [1.0, 1.0])])
    _write_checkpoint_markers(run_partial, [1])
    _write_run_config(run_partial, checkpoint_count=2, eval_seed=7)  # K=2, only 1 scored

    partial_at_1999 = build_verdict(run_partial, B=1999)
    assert partial_at_1999["authoritative"] is False
    assert partial_at_1999["delta"] is None


def _null_pair_scores(rng: random.Random, n_pairs: int) -> list[float]:
    """``n_pairs`` i.i.d. pair scores with no built-in trend (mean 1.0/2, symmetric).

    Each of the pair's two games independently scores 0.0/0.5/1.0 with
    probabilities 0.45/0.10/0.45 -- a fair, mildly-drawish coin, identical for
    every member, so the population Delta is exactly 0.
    """

    def _game() -> float:
        draw = rng.random()
        if draw < 0.45:
            return 0.0
        if draw < 0.55:
            return 0.5
        return 1.0

    return [_game() + _game() for _ in range(n_pairs)]


@pytest.mark.slow
def test_bootstrap_gate_false_at_approximately_the_nominal_rate_under_a_true_null(tmp_path):
    """A synthetic no-improvement fixture (exchangeable pair scores, task 1 pin
    7/8) over a complete tiny K-set: with no true Elo separation across
    checkpoints, repeated independent data draws must gate True only rarely --
    B=39's conservative min/max-based order-statistic CI (task 1 pin 7) is far
    more conservative than the asymptotic 95% two-sided rate, so the empirical
    false-positive rate should sit well below the nominal one-sided 2.5%.
    """
    k = 3
    n_pairs = 6
    n_reps = 60
    b = 39

    gates = []
    for rep in range(n_reps):
        root = tmp_path / f"null-{rep}"
        rng = random.Random(1_000_000 + rep)
        for version in range(1, k + 1):
            _write_member(root, version, [(7, "random", _null_pair_scores(rng, n_pairs))])
        snapshot = load_snapshot(root)
        seed = bootstrap_seed(2_000_000 + rep)
        replicate_ratings = list(bootstrap_replicates(snapshot, seed, b))
        ci = order_statistic_ci(replicate_deltas(replicate_ratings), b)
        gates.append(delta_gate(ci))

    false_positive_rate = sum(gates) / n_reps
    assert false_positive_rate <= 0.15, (
        f"gate=True on {sum(gates)}/{n_reps} genuinely-null repetitions "
        f"({false_positive_rate:.1%}) -- expected well below the nominal 2.5% "
        "one-sided rate"
    )


def test_bootstrap_gate_true_under_a_strong_monotone_trend(tmp_path):
    """A strong-trend fixture -- win rate climbing from 10% to 90% across a
    complete K-set -- must reliably gate True."""
    k = 4
    n_pairs = 20
    b = 39

    def _biased_pair_scores(rng: random.Random, p_win: float) -> list[float]:
        def _game() -> float:
            return 1.0 if rng.random() < p_win else 0.0

        return [_game() + _game() for _ in range(n_pairs)]

    for rep in range(5):
        root = tmp_path / f"trend-{rep}"
        rng = random.Random(5_000_000 + rep)
        for version in range(1, k + 1):
            p_win = 0.1 + 0.8 * (version - 1) / (k - 1)  # 0.1 .. 0.9, strictly increasing
            _write_member(root, version, [(7, "random", _biased_pair_scores(rng, p_win))])
        snapshot = load_snapshot(root)
        seed = bootstrap_seed(6_000_000 + rep)
        replicate_ratings = list(bootstrap_replicates(snapshot, seed, b))
        ci = order_statistic_ci(replicate_deltas(replicate_ratings), b)
        assert delta_gate(ci) is True


# ==============================================================================
# 8. Doc <-> protocol-registry golden (tasks/m4/007, subtask 7.3)
# ==============================================================================

_DESIGN_DOC_PATH = (
    Path(__file__).resolve().parent.parent / "metadocs" / "blokus-duo-az-design-v0_5.md"
)
_DOC_AMENDMENT_BRANCH = "docs/m4-pin-eval-protocol"
# The bolded lead-in of the actual section-9 block. The bare phrase also appears
# in the status-header changelog ("section 9 gains a ... block"), so anchoring on
# the bold form keeps the parse scoped to the pins themselves.
_PINNED_PROTOCOL_HEADING = "**Pre-registered protocol (M4 pins).**"


def test_protocol_registry_matches_the_literal_pinned_values():
    """Always asserted, independent of the doc amendment's merge status -- the
    exact values tasks/m4/001's amendment pins, mirrored as module constants
    (core.eval_protocol) so the protocol cannot silently drift once production
    games exist."""
    assert eval_protocol.PROTOCOL_VERSION == 1
    assert eval_protocol.PAIRS_PER_CELL == 24
    assert eval_protocol.EVAL_SIMS == 512
    assert eval_protocol.RUNG8_LAG_DIVISOR == 4
    assert eval_protocol.RUNG8_EARLIEST_VERSION == 1
    assert eval_protocol.BOOTSTRAP_B_PRODUCTION == 1999
    assert eval_protocol.BOOTSTRAP_B_ADMISSIBLE_MODULUS == 40
    assert eval_protocol.BOOTSTRAP_B_ADMISSIBLE_REMAINDER == 39
    assert eval_protocol.BOOTSTRAP_CI_LOWER_QUANTILE == 0.025
    assert eval_protocol.BOOTSTRAP_CI_UPPER_QUANTILE == 0.975
    # The rank rule at the pinned production B: (1999+1)*0.025=50, (1999+1)*0.975=1950.
    b_plus_one = eval_protocol.BOOTSTRAP_B_PRODUCTION + 1
    lower_rank = b_plus_one * eval_protocol.BOOTSTRAP_CI_LOWER_QUANTILE
    upper_rank = b_plus_one * eval_protocol.BOOTSTRAP_CI_UPPER_QUANTILE
    assert (lower_rank, upper_rank) == (50.0, 1950.0)


def _extract_number(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).replace(",", "")


def test_protocol_registry_matches_the_amended_design_doc_section_9_pins():
    """The doc<->constants golden: parses the pinned §9 'Pre-registered protocol
    (M4 pins)' block and compares every parsed value against this module's own
    constants. Arms itself automatically once tasks/m4/001's design-doc
    amendment lands -- it currently lives on the not-yet-merged
    docs/m4-pin-eval-protocol branch (core.eval_protocol's own module
    docstring), so until that block exists in this tree, this test has nothing
    to compare against and explicitly skips rather than failing on a doc
    section that was never written here.
    """
    doc_text = _DESIGN_DOC_PATH.read_text(encoding="utf-8")
    if _PINNED_PROTOCOL_HEADING not in doc_text:
        pytest.skip(
            "design doc has no section-9 'Pre-registered protocol (M4 pins)' block yet "
            f"-- the amendment lives on the not-yet-merged {_DOC_AMENDMENT_BRANCH} branch "
            "(see core.eval_protocol's module docstring)"
        )

    # Scope the search to the block itself: from the heading to the next
    # section boundary ('---' or the next '## ' heading) -- never the whole doc.
    start = doc_text.index(_PINNED_PROTOCOL_HEADING)
    rest = doc_text[start:]
    end_match = re.search(r"\n(?:---|## )", rest)
    block = rest[: end_match.start()] if end_match else rest

    pairs_per_cell = _extract_number(r"pairs[- ]per[- ]cell[^0-9]{0,20}(\d[\d,]*)", block)
    assert pairs_per_cell is not None, f"could not find a pairs-per-cell pin in: {block!r}"
    assert int(pairs_per_cell) == eval_protocol.PAIRS_PER_CELL

    eval_sims = _extract_number(r"\bS\s*=\s*(\d[\d,]*)", block)
    assert eval_sims is not None, f"could not find the eval-sims (S) pin in: {block!r}"
    assert int(eval_sims) == eval_protocol.EVAL_SIMS

    bootstrap_b = _extract_number(r"`?\bB`?\s*=\s*(\d[\d,]*)", block)
    assert bootstrap_b is not None, f"could not find the bootstrap B pin in: {block!r}"
    assert int(bootstrap_b) == eval_protocol.BOOTSTRAP_B_PRODUCTION

    assert "0.025" in block and "0.975" in block, (
        f"could not find the order-statistic rank-rule's quantiles in: {block!r}"
    )

    rung8_lag = _extract_number(r"K`?\s*/\s*(\d+)", block) or _extract_number(
        r"lag[^0-9]{0,20}(\d+)", block
    )
    assert rung8_lag is not None, f"could not find the rung-8 lag divisor in: {block!r}"
    assert int(rung8_lag) == eval_protocol.RUNG8_LAG_DIVISOR


# ==============================================================================
# 9. detect_plateau: the profiled-plateau rule (tasks/m4/008; design doc §12 M4)
# ==============================================================================
#
# Fixture-building helpers below are deliberately additional to (never a rewrite
# of) section 4's `_write_checkpoint_markers`/`_build_real_shape_metrics_fixture`:
# the plateau rule needs GPU-hour *segments* (§1's x-axis join's `gpu_hours`
# coordinate), which those two never exercise (both leave gpu_hours at 0.0).
#
# Every randomized fixture below fixes its own `random.Random(seed)` and was
# picked by direct search over (seed, n_pairs) for the one sub-condition it is
# named after to land on the intended side of its pinned threshold while every
# other sub-condition lands on the *other* side -- so each test isolates the one
# clause it claims to. This mirrors the module's own existing style
# (`_null_pair_scores`/`_biased_pair_scores` above): a deterministic seed is not
# a fragile magic number here, since core.eval_stats's whole pipeline (Bradley-
# Terry coordinate ascent, Mann-Kendall, the order-statistic CI) is pure and
# reproducible -- the same seed always reproduces the same booleans on any
# machine.


def _win_rate_pair_scores(rng: random.Random, p_win: float, n_pairs: int) -> list[float]:
    """``n_pairs`` i.i.d. pair scores at win probability ``p_win`` (mirrors the
    module's own ``_biased_pair_scores``/``_null_pair_scores`` inline helpers,
    pulled to module scope since several plateau fixtures below share it)."""

    def _game() -> float:
        return 1.0 if rng.random() < p_win else 0.0

    return [_game() + _game() for _ in range(n_pairs)]


def _write_gpu_segmented_checkpoint_markers(
    run_dir, versions: list[int], *, seconds_per_version: float
) -> None:
    """One ``checkpoint_published`` marker per version, each preceded by a closed
    GPU segment of ``seconds_per_version`` seconds -- so ``reduce_run``'s (and
    therefore ``elo_curve``'s) cumulative ``gpu_hours`` coordinate advances by a
    known, controllable amount between consecutive members. Unlike section 4's
    ``_write_checkpoint_markers`` (deliberately left at ``gpu_hours = 0.0``
    everywhere, since the Delta/CI/MK tests it serves never read that column),
    the plateau rule's GPU-hour-span sub-condition is the whole point here.

    Args:
        run_dir: The run directory.
        versions: Member versions to publish, in the order to space them.
        seconds_per_version: Wall-clock seconds of GPU segment time inserted
            before each marker -- ``len(versions) - 1`` such gaps separate the
            first and last marker, so a window's GPU-hour span is exactly
            ``(len(window) - 1) * seconds_per_version / 3600``.
    """
    orch = EpochMetricsWriter(run_dir, "orchestrator")
    learner = EpochMetricsWriter(run_dir, "learner")
    t = 0.0
    orch.append(segment_start_record(device="cpu", timestamp=t))
    for i, version in enumerate(versions, start=1):
        t += seconds_per_version
        orch.append(segment_end_record(timestamp=t))
        _append_checkpoint_published(learner, version=version, learner_step=10 * i, timestamp=t)
        t += 1.0
        orch.append(segment_start_record(device="cpu", timestamp=t))
    orch.append(segment_end_record(timestamp=t + 1.0))


def _build_plateau_fixture(
    run_dir,
    scores_by_version: dict,
    *,
    seconds_per_version: float = 4200.0,
    k_target: int = 30,
    eval_seed: int = 4242,
    refresh_elo_curve: bool = True,
):
    """Build a complete eval-store + config + (optionally) ``elo_curve.json``
    fixture for one member per ``scores_by_version`` key, each a single rung-7
    vs. ``"random"`` cell -- everything :func:`detect_plateau` reads.

    Args:
        run_dir: The run directory.
        scores_by_version: ``{model_version: pair_scores}``.
        seconds_per_version: Forwarded to
            :func:`_write_gpu_segmented_checkpoint_markers`.
        k_target: ``training.checkpoint_count`` written to ``config.json``.
        eval_seed: ``evaluation.eval_seed`` written to ``config.json`` (also
            what :func:`detect_plateau` derives its ``bootstrap_seed`` from).
        refresh_elo_curve: If ``True`` (the default), write
            ``eval/elo_curve.json`` from the resulting snapshot -- the task-7
            artifact :func:`detect_plateau` reads its GPU-hours join from.
            ``False`` leaves it unwritten, for the missing-artifact
            insufficient-data tests.

    Returns:
        The loaded :class:`~core.eval_store.EvalSnapshot`.
    """
    for version, scores in scores_by_version.items():
        _write_member(run_dir, version, [(7, "random", scores)])
    _write_gpu_segmented_checkpoint_markers(
        run_dir, sorted(scores_by_version), seconds_per_version=seconds_per_version
    )
    _write_run_config(run_dir, checkpoint_count=k_target, eval_seed=eval_seed)
    snapshot = load_snapshot(run_dir)
    if refresh_elo_curve:
        elo_curve(run_dir, snapshot)
    return snapshot


# --- registry: the six plateau constants' literal pinned values -------------------


def test_plateau_registry_constants_match_the_literal_pinned_values():
    assert eval_protocol.PLATEAU_WINDOW_M == 8
    assert eval_protocol.PLATEAU_MK_ALPHA == 0.05
    assert eval_protocol.PLATEAU_CI_WIDTH_THRESHOLD_ELO == 75.0
    assert eval_protocol.PLATEAU_GPU_HOURS_MIN == 8.0
    assert eval_protocol.PLATEAU_CONFIRMATION_COUNT == 2
    assert "plateau_window_m" in eval_protocol.REGISTRY
    assert "plateau_half_window_rule" in eval_protocol.REGISTRY


def test_plateau_registry_constants_are_covered_by_the_protocol_fingerprint(monkeypatch):
    before = eval_protocol.protocol_fingerprint()
    monkeypatch.setitem(eval_protocol.REGISTRY, "plateau_window_m", 999)
    after = eval_protocol.protocol_fingerprint()
    assert before != after


# --- doc <-> constants golden: the amended §12 M4 plateau bullet ------------------

_PLATEAU_BULLET_HEADING = "**Plateau-detection rule (operationalize the M6 gate)"
_PLATEAU_BULLET_END = "\n  - **Bootstrap seed"


def test_protocol_registry_matches_the_amended_design_doc_plateau_bullet():
    """The doc<->constants golden for task 8's amendment: this tree already
    carries the committed plateau rule (unlike task 7's pins golden above, this
    one is not conditionally skipped) -- parses §12 M4's plateau-detection-rule
    bullet and compares every one of its numeric constants against
    ``core.eval_protocol``'s own module constants. Tolerant of prose: anchored
    on the bolded bullet heading and its own sub-bullet labels, not exact
    phrasing elsewhere in the paragraph.
    """
    doc_text = _DESIGN_DOC_PATH.read_text(encoding="utf-8")
    assert _PLATEAU_BULLET_HEADING in doc_text, (
        "design doc §12 M4 has no 'Plateau-detection rule' bullet -- expected the "
        "tasks/m4/008 doc-first amendment to already be committed in this tree"
    )
    start = doc_text.index(_PLATEAU_BULLET_HEADING)
    rest = doc_text[start:]
    end_match = re.search(re.escape(_PLATEAU_BULLET_END), rest)
    block = rest[: end_match.start()] if end_match else rest

    window_m = _extract_number(r"`M`\s*=\s*(\d[\d,]*)\s*evaluated member", block)
    assert window_m is not None, f"could not find the window-length M pin in: {block!r}"
    assert int(window_m) == eval_protocol.PLATEAU_WINDOW_M

    alpha = _extract_number(r"α`\s*=\s*([\d.]+)", block)
    assert alpha is not None, f"could not find the Mann-Kendall alpha pin in: {block!r}"
    assert float(alpha) == eval_protocol.PLATEAU_MK_ALPHA

    threshold = _extract_number(r"strictly below\s*(\d[\d,]*)\s*Elo points", block)
    assert threshold is not None, f"could not find the CI-width threshold pin in: {block!r}"
    assert float(threshold) == eval_protocol.PLATEAU_CI_WIDTH_THRESHOLD_ELO

    gpu_hours = _extract_number(r"span\s*≥\s*(\d[\d,]*)\s*GPU-hours", block)
    assert gpu_hours is not None, f"could not find the GPU-hour window pin in: {block!r}"
    assert float(gpu_hours) == eval_protocol.PLATEAU_GPU_HOURS_MIN

    assert re.search(r"\btwo\b consecutive", block, re.IGNORECASE), (
        f"could not find the anti-flap confirmation-count pin in: {block!r}"
    )
    assert eval_protocol.PLATEAU_CONFIRMATION_COUNT == 2

    # The amendment must retire the "Δ-CI width" sketch's ambiguity by naming a
    # windowed contrast explicitly distinct from §1's Delta (tasks/m4/008 detail).
    assert "not" in block and "§1" in block, (
        f"could not find the 'explicitly not §1's Delta' disambiguation in: {block!r}"
    )
    # The insufficient-data clause must be present as its own named case.
    assert "insufficient-data" in block.lower() or "INSUFFICIENT-DATA" in block, (
        f"could not find the insufficient-data clause in: {block!r}"
    )


# --- tri-state structural contract: never a bool, never coerced -------------------


def test_plateau_outcome_values_are_the_three_named_strings():
    outcomes = {
        PLATEAU_OUTCOME_PLATEAU,
        PLATEAU_OUTCOME_NO_PLATEAU,
        PLATEAU_OUTCOME_INSUFFICIENT_DATA,
    }
    assert outcomes == {"plateau", "no_plateau", "insufficient_data"}


def test_plateau_result_implements_no_dunder_bool():
    """The tri-state outcome must only ever be read by comparing ``outcome``
    against a named value -- never by accidental truthiness coercion."""
    assert "__bool__" not in PlateauResult.__dict__
    assert "__bool__" not in WindowCondition.__dict__


# --- insufficient-data: a window shorter than M ------------------------------------


def test_window_shorter_than_m_is_insufficient_data(tmp_path):
    for version in range(1, 8):  # 7 < PLATEAU_WINDOW_M (8) -- no config/elo_curve needed.
        _write_member(tmp_path, version, [(7, "random", [1.0, 1.0])])

    result = detect_plateau(tmp_path, B=39)

    assert result.outcome == PLATEAU_OUTCOME_INSUFFICIENT_DATA
    assert result.current is None
    assert result.previous is None
    assert result.confirmation_count == 0
    assert result.confirmed_versions == ()
    assert "7" in result.reason and "8" in result.reason


def test_a_two_point_series_is_insufficient_data(tmp_path):
    """The degenerate low end of the same "short window" gate -- with only 2
    evaluated members, a from-scratch Mann-Kendall over the whole series would
    itself be insufficient-data (n < 3); the window-length gate fires first
    either way, so both routes collapse onto the identical tri-state outcome,
    never miscoerced into plateau or no-plateau."""
    for version in (1, 2):
        _write_member(tmp_path, version, [(7, "random", [1.0, 1.0])])

    result = detect_plateau(tmp_path, B=39)

    assert result.outcome == PLATEAU_OUTCOME_INSUFFICIENT_DATA
    assert result.current is None


# --- insufficient-data: missing GPU-hour coordinate --------------------------------


def test_missing_elo_curve_artifact_is_insufficient_data(tmp_path):
    """No ``elo_curve.json`` has ever been written -- every window member is
    missing its GPU-hours coordinate. Scores vary by version (never all-tied)
    so this isolates the missing-GPU-hours trigger alone from the separate
    all-tied-window trigger below."""
    rng = random.Random(2)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 40) for v in range(1, 9)}
    _build_plateau_fixture(tmp_path, scores, refresh_elo_curve=False)

    result = detect_plateau(tmp_path, B=39)

    assert result.outcome == PLATEAU_OUTCOME_INSUFFICIENT_DATA
    assert result.current is not None  # sub-conditions still computed for audit
    assert result.current.gpu_hours_span is None
    assert result.current.gpu_span_sufficient is False
    assert result.elo_curve_fingerprint is None
    assert "GPU-hours coordinate" in result.reason


def test_stale_elo_curve_missing_the_newest_member_is_insufficient_data(tmp_path):
    """``elo_curve.json`` exists but was written before the newest member was
    scored -- a realistic staleness case (the harness has not yet re-run
    ``elo_curve``/``build_verdict`` since the snapshot advanced), distinct from
    the artifact never existing at all."""
    scores = {v: [1.0, 1.0] for v in range(1, 8)}  # members 1..7 only, refreshed now
    _build_plateau_fixture(tmp_path, scores, refresh_elo_curve=True)
    stale_fingerprint = elo_curve_path(tmp_path).read_bytes()

    # Member 8 completes afterward, without ever refreshing elo_curve.json again.
    _write_member(tmp_path, 8, [(7, "random", [1.0, 1.0])])

    result = detect_plateau(tmp_path, B=39)

    assert result.outcome == PLATEAU_OUTCOME_INSUFFICIENT_DATA
    assert result.current.versions == (1, 2, 3, 4, 5, 6, 7, 8)
    assert result.current.gpu_hours_span is None  # version 8 has no row in the stale file
    assert result.elo_curve_fingerprint == hashlib.sha256(stale_fingerprint).hexdigest()


def test_an_insufficient_data_windows_mann_kendall_reading_also_blocks(tmp_path):
    """Symmetric with the missing-GPU-coordinate trigger: if a window's own
    Mann-Kendall reading were ever insufficient-data (unreachable at the pinned
    M=8 >= 3, but never assumed away), the overall verdict must be
    insufficient-data too, not silently treated as "non-significant"."""
    scores = {v: [1.0, 1.0] for v in range(1, 9)}
    _build_plateau_fixture(tmp_path, scores)
    result = detect_plateau(tmp_path, B=39)
    # At M=8 this path is not reachable -- assert the structural guarantee instead:
    # every real window's own n always equals the pinned M, which is >= 3.
    assert result.current.mann_kendall.insufficient_data is False
    assert result.current.mann_kendall.n == eval_protocol.PLATEAU_WINDOW_M


# --- insufficient-data: an all-tied window (P2.5; tasks/m4/008 Test Strategy) ------


def test_all_tied_window_is_insufficient_data_not_plateau(tmp_path):
    """An all-tied window hits Mann-Kendall's own pinned sigma=0 degenerate
    branch (``insufficient_data=False, s=0, z=0.0, p=1.0``) rather than its
    literal ``insufficient_data`` flag -- distinct from the unreachable n<3
    case exercised above. That branch carries zero real trend information,
    and the windowed contrast's CI collapses to width 0 for the same reason
    (every replicate curve is identical too), so the full conjunction would
    otherwise be trivially satisfied on a window with no statistical content.
    The design doc §12 M4 insufficient-data clause and tasks/m4/008's Test
    Strategy both require this to read as INSUFFICIENT-DATA, never PLATEAU."""
    scores = {v: [1.0, 1.0] for v in range(1, 11)}  # every member ties exactly
    _build_plateau_fixture(tmp_path, scores)

    result = detect_plateau(tmp_path, B=39)

    assert result.outcome == PLATEAU_OUTCOME_INSUFFICIENT_DATA
    assert result.outcome != PLATEAU_OUTCOME_PLATEAU
    assert result.current.mann_kendall.insufficient_data is False  # not the n<3 branch
    assert result.current.mann_kendall.p == 1.0
    assert result.current.mk_all_tied is True
    assert result.current.mk_non_significant is False  # forced False, never "legitimate"
    assert result.current.contrast_ci_width == 0.0  # the degenerate CI collapse too
    assert result.current.satisfied is False
    assert result.confirmation_count == 0
    assert result.confirmed_versions == ()
    assert "all-tied" in result.reason


def test_a_window_shorter_than_m_and_an_all_tied_window_are_both_never_plateau(tmp_path):
    """Both P2.5 triggers side by side, over the same two candidate window
    lengths one below and one at ``M``: neither ever reaches PLATEAU or
    NO_PLATEAU, only the explicit tri-state INSUFFICIENT-DATA."""
    short_scores = {v: [1.0, 0.0] for v in range(1, 8)}  # 7 < M -- too short outright
    tied_scores = {v: [1.0, 1.0] for v in range(1, 9)}  # exactly M, but all-tied

    for scores in (short_scores, tied_scores):
        root = tmp_path / f"run-{len(scores)}"
        root.mkdir()
        _build_plateau_fixture(root, scores)
        result = detect_plateau(root, B=39)
        assert result.outcome == PLATEAU_OUTCOME_INSUFFICIENT_DATA
        assert result.outcome not in (PLATEAU_OUTCOME_PLATEAU, PLATEAU_OUTCOME_NO_PLATEAU)


# --- each sub-condition independently flips the outcome ----------------------------


def test_mann_kendall_significant_trend_blocks_plateau_alone(tmp_path):
    rng = random.Random(1000)
    scores = {v: _win_rate_pair_scores(rng, 0.15 + 0.7 * (v - 1) / 8, 300) for v in range(1, 9)}
    _build_plateau_fixture(tmp_path, scores)

    result = detect_plateau(tmp_path, B=39)

    c = result.current
    assert c.mk_non_significant is False  # the one blocking sub-condition
    assert c.ci_narrow is True
    assert c.gpu_span_sufficient is True
    assert c.satisfied is False
    assert result.outcome == PLATEAU_OUTCOME_NO_PLATEAU
    assert result.confirmation_count == 0


def test_wide_ci_blocks_plateau_alone(tmp_path):
    rng = random.Random(1)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 50) for v in range(1, 9)}
    _build_plateau_fixture(tmp_path, scores)

    result = detect_plateau(tmp_path, B=39)

    c = result.current
    assert c.mk_non_significant is True
    assert c.ci_narrow is False  # the one blocking sub-condition
    assert c.contrast_ci_width >= eval_protocol.PLATEAU_CI_WIDTH_THRESHOLD_ELO
    assert c.gpu_span_sufficient is True
    assert c.satisfied is False
    assert result.outcome == PLATEAU_OUTCOME_NO_PLATEAU


def test_thin_gpu_hour_span_blocks_plateau_alone(tmp_path):
    rng = random.Random(0)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 200) for v in range(1, 9)}
    _build_plateau_fixture(tmp_path, scores, seconds_per_version=500.0)

    result = detect_plateau(tmp_path, B=39)

    c = result.current
    assert c.mk_non_significant is True
    assert c.ci_narrow is True
    assert c.gpu_span_sufficient is False  # the one blocking sub-condition
    assert c.gpu_hours_span == pytest.approx(7 * 500.0 / 3600.0)
    assert c.satisfied is False
    assert result.outcome == PLATEAU_OUTCOME_NO_PLATEAU


# --- the anti-flap confirmation clause ---------------------------------------------


def test_conjunction_holding_only_at_the_newest_member_is_no_plateau_pending_confirmation(
    tmp_path,
):
    rng = random.Random(1)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 100) for v in range(1, 10)}  # 9 members
    _build_plateau_fixture(tmp_path, scores)

    result = detect_plateau(tmp_path, B=39)

    assert result.current.newest_version == 9
    assert result.previous.newest_version == 8
    assert result.current.satisfied is True
    assert result.previous.satisfied is False  # not yet confirmed one snapshot earlier
    assert result.confirmation_count == 1
    assert result.confirmed_versions == (9,)
    assert result.outcome == PLATEAU_OUTCOME_NO_PLATEAU
    assert "pending confirmation" in result.reason


def test_conjunction_holding_at_two_consecutive_snapshots_is_plateau(tmp_path):
    rng = random.Random(0)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 120) for v in range(1, 11)}  # 10 members
    _build_plateau_fixture(tmp_path, scores)

    result = detect_plateau(tmp_path, B=39)

    assert result.current.newest_version == 10
    assert result.previous.newest_version == 9
    assert result.current.satisfied is True
    assert result.previous.satisfied is True
    assert result.confirmation_count == 2
    assert result.confirmed_versions == (9, 10)
    assert result.outcome == PLATEAU_OUTCOME_PLATEAU
    assert result.reason is None


# --- non-star-graph regression: `previous` must not absorb later evidence --------


def test_previous_window_is_unaffected_by_a_later_rung8_game_into_its_own_span(tmp_path):
    """Regression for the truncated-refit fix (tasks/m4/008 review finding #1).

    Every other plateau fixture in this module is a degenerate star graph -- each
    member only ever plays the fixed anchor "random" -- the one case where slicing
    both confirmation windows out of a single, fully-informed fit happens to give
    the same answer as two genuinely independent, contemporaneous reads. Real runs
    are not star graphs: §9 pin 2's one anchored Bradley-Terry fit links every
    member through the rungs they share, and a rung-8 cell keeps a historical
    opponent's *own* earlier rung-7 identity (``core.eval_agents.
    historical_opponents``'s module note), so member 9's own rung-8 game against
    member 1 pulls on member 1's fitted rating too -- and member 1 sits inside the
    ``previous`` window's span (versions 1..8). This builds that exact shape and
    checks ``previous`` reads identically to a store that only ever had members
    1..8 -- i.e. that it is genuinely order-independent, never contaminated by
    member 9's rung-8 evidence arriving after member 8 was already the newest.
    """
    root_full = tmp_path / "full"
    root_truncated = tmp_path / "truncated"
    root_full.mkdir()
    root_truncated.mkdir()

    for version in range(1, 9):
        _write_member(root_full, version, [(7, "random", [1.0, 1.0])])
        _write_member(root_truncated, version, [(7, "random", [1.0, 1.0])])
    # Member 9 exists only in the "full" store -- and, beyond its ordinary
    # rung-7-vs-random cell, plays a rung-8 game against member 1's own rung-7
    # identity, lopsided enough to move member 1's fitted rating.
    _write_member(
        root_full,
        9,
        [(7, "random", [1.0, 1.0]), (7, "rung7-v1-1", [2.0, 2.0])],
    )

    _write_gpu_segmented_checkpoint_markers(
        root_full, list(range(1, 10)), seconds_per_version=4200.0
    )
    _write_gpu_segmented_checkpoint_markers(
        root_truncated, list(range(1, 9)), seconds_per_version=4200.0
    )
    _write_run_config(root_full, checkpoint_count=30, eval_seed=4242)
    _write_run_config(root_truncated, checkpoint_count=30, eval_seed=4242)

    snapshot_full = load_snapshot(root_full)
    snapshot_truncated = load_snapshot(root_truncated)
    elo_curve(root_full, snapshot_full)
    elo_curve(root_truncated, snapshot_truncated)

    # Sanity check the fixture actually exercises the coupling this regression
    # targets: member 1's rating really does move once member 9's rung-8 game
    # against it is folded into one joint fit -- otherwise this fixture would not
    # be testing anything beyond the star-graph case every other test already covers.
    full_v1_elo = fit_snapshot_elo(snapshot_full)["rung7-v1-1"]
    truncated_v1_elo = fit_snapshot_elo(snapshot_truncated)["rung7-v1-1"]
    assert full_v1_elo != pytest.approx(truncated_v1_elo)

    result_full = detect_plateau(root_full, B=39)
    result_truncated = detect_plateau(root_truncated, B=39)

    assert result_full.previous.newest_version == 8
    assert result_truncated.current.newest_version == 8
    assert result_truncated.previous is None  # only 8 members -- no second window yet

    # The fix: `previous`'s own sub-conditions read exactly as a standalone
    # snapshot that never had member 9 (or its rung-8 game) would have -- never
    # perturbed by evidence that did not exist yet when member 8 was newest.
    assert result_full.previous.mann_kendall == result_truncated.current.mann_kendall
    assert result_full.previous.contrast == pytest.approx(result_truncated.current.contrast)
    assert result_full.previous.contrast_ci == pytest.approx(result_truncated.current.contrast_ci)
    assert result_full.previous.gpu_hours_span == pytest.approx(
        result_truncated.current.gpu_hours_span
    )
    assert result_full.previous.satisfied == result_truncated.current.satisfied


# --- hand-computed windowed-contrast value -----------------------------------------


def test_windowed_contrast_matches_hand_computed_half_window_means(tmp_path):
    """A deterministic (non-random) star fixture: versions 1-4 score exactly 2.0
    of 4 games (50%) against the anchor and versions 5-8 score exactly 3.0 of 4
    (75%) -- each closed-form-verifiable once §9 pin 6's one-virtual-draw
    regularizer is folded in (``_closed_form`` mirrors ``tests/test_elo.py``'s
    own convention: effective rate = ``(score + 0.5) / (games + 1)``), the same
    style ``test_two_checkpoint_three_rung_fixture_matches_hand_computed_bt_
    ratings`` above already uses. A 50% raw rate is unmoved by the (symmetric)
    virtual draw (elo stays exactly 0.0); 75% becomes ``(3.0 + 0.5) / 5 = 0.7``.
    Delta_window = mean(elo, 5..8) - mean(elo, 1..4) is then hand-verifiable,
    mirroring ``test_delta_hat_matches_hand_computed_window_means``'s style for
    the §1 Delta. No ``elo_curve.json`` is written -- irrelevant to this
    assertion, since a window's sub-conditions are computed (and exposed for
    audit) before the GPU-hours check ever runs.
    """
    scores = {v: [1.0, 1.0] for v in range(1, 5)}  # sum=2.0/4 games = 0.5 win rate
    scores.update({v: [1.5, 1.5] for v in range(5, 9)})  # sum=3.0/4 games = 0.75
    _build_plateau_fixture(tmp_path, scores, refresh_elo_curve=False)

    result = detect_plateau(tmp_path, B=39)

    expected_contrast = _closed_form((3.0 + 0.5) / 5.0) - _closed_form((2.0 + 0.5) / 5.0)
    assert result.current.contrast == pytest.approx(expected_contrast, abs=1e-4)
    assert result.current.mann_kendall.s == 16  # every one of the 4x4 cross pairs is a "+"


# --- snapshot-only reads: an on-disk partial cell perturbs nothing ----------------


def test_on_disk_partial_cell_perturbs_nothing(tmp_path):
    rng = random.Random(0)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 120) for v in range(1, 11)}
    _build_plateau_fixture(tmp_path, scores)
    baseline = detect_plateau(tmp_path, B=39)

    # A stray, never-completed member-11 cell -- structurally "scheduled" but not
    # yet complete -- must be structurally invisible (task 1 pin 9 / P2.2).
    header = _header(candidate_version=11, rung=7, opponent_id="random", n_pairs=2)
    register_member(tmp_path, 11, [header.cell_id.to_string()])
    _fill(tmp_path, header, [1.0, 1.0])  # opened + appended, never completed

    after = detect_plateau(tmp_path, B=39)

    assert after == baseline  # exact dataclass equality, not just the outcome field


# --- determinism: same store + seed -> identical result --------------------------


def test_same_store_and_seed_give_identical_plateau_results_across_two_runs(tmp_path):
    rng = random.Random(0)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 120) for v in range(1, 11)}

    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    _build_plateau_fixture(root_a, scores)
    _build_plateau_fixture(root_b, scores)

    result_a = detect_plateau(root_a, B=39)
    result_b = detect_plateau(root_b, B=39)

    assert result_a == result_b
    # And calling twice over the one store is equally deterministic.
    assert detect_plateau(root_a, B=39) == result_a


# --- the payload carries the fingerprints and sub-condition values ----------------


def test_plateau_result_carries_the_snapshot_elo_curve_and_protocol_fingerprints(tmp_path):
    rng = random.Random(0)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 120) for v in range(1, 11)}
    snapshot = _build_plateau_fixture(tmp_path, scores)

    result = detect_plateau(tmp_path, B=39)

    assert result.snapshot_fingerprint == snapshot.snapshot_fingerprint
    assert (
        result.elo_curve_fingerprint
        == hashlib.sha256(elo_curve_path(tmp_path).read_bytes()).hexdigest()
    )
    assert result.protocol_fingerprint == eval_protocol.protocol_fingerprint()
    assert result.window_m == eval_protocol.PLATEAU_WINDOW_M


def test_detect_plateau_rejects_a_non_admissible_b(tmp_path):
    rng = random.Random(0)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 120) for v in range(1, 11)}
    _build_plateau_fixture(tmp_path, scores)

    with pytest.raises(ValueError):
        detect_plateau(tmp_path, B=100)


def test_conjunction_regressing_at_the_newest_member_discards_the_stale_credit(tmp_path):
    """The anti-flap direction the clause exists for: satisfied at the previous
    window, regressed at the newest one -> no plateau, and the previously
    satisfying reading earns zero confirmation credit (a future refactor of the
    newest-first confirmation loop to a count-based check would credit it).
    Members 1..9 reuse the exact rng(0) stream the confirmed-plateau fixture
    above draws for its first nine members, so the previous window (2..9) is
    known-satisfied; member 10 gets only 12 pairs, whose noisy rating widens the
    newest window's contrast CI past the pinned 75-Elo threshold.
    """
    rng = random.Random(0)
    scores = {v: _win_rate_pair_scores(rng, 0.5, 120) for v in range(1, 10)}
    scores[10] = _win_rate_pair_scores(rng, 0.5, 12)
    _build_plateau_fixture(tmp_path, scores)

    result = detect_plateau(tmp_path, B=39)

    assert result.previous.newest_version == 9
    assert result.previous.satisfied is True
    assert result.current.newest_version == 10
    assert result.current.satisfied is False
    assert result.confirmation_count == 0
    assert result.confirmed_versions == ()
    assert result.outcome == PLATEAU_OUTCOME_NO_PLATEAU
