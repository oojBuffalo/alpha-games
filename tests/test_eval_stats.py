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

import json
import math

import pytest

from core.agents import RandomAgent
from core.elo import fit_elo
from core.eval_stats import (
    ANCHOR_AGENT,
    bootstrap_replicate,
    bootstrap_replicate_matches,
    bootstrap_replicates,
    bootstrap_seed,
    checkpoint_elo,
    elo_curve,
    elo_curve_path,
    fit_snapshot_elo,
    snapshot_matches,
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
