"""M2.5 falsifiable exit-gate battery (§12 M2.5, task 7).

The gate is the milestone's pass/fail verdict, so what needs testing is its
*mechanics*, not whether the real run passes: predicates computed from persisted
evidence, the pinned comparator directions, loud rejection of evidence that
cannot support the pinned windows or that came from another instance, and the
PASS-iff-conjunction verdict.

Three layers:

1. **Predicates on synthetic run records** — fast, deterministic, no torch: the
   three §12 M2.5 predicates evaluated on hand-built records written to disk and
   read back through ``core.selfplay.load_run_record``, including both boundary
   directions (a statistic exactly at the pinned bound **passes**) and the
   truncated-record rejection.
2. **Completeness of the evidence** — synthetic records at the *pinned* 2,000
   step / 2,000 game scale: the complete one is accepted, and a short one, a
   broken step-id sequence, a short games list or a checkpoint away from the
   configured final step are each rejected as not-evaluable (exit 2), before any
   predicate is scored.
3. **Verdict logic** — all eight pass/fail quadrants of the three-predicate
   conjunction.
4. **End-to-end smoke** — a tiny real run (CPU, few games, few sims) driven
   through the whole gate: identity checks, the paired set, the written verdict
   record, and the process exit code. It asserts the machinery ran, never that
   the gate passed: the thresholds are pre-registered and a tiny run says
   nothing about them.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.runconfig import MICRO_RUN_CONFIG_PATH, RunConfig, load_run_config
from core.runner import GameRecord, PairResult
from core.selfplay import RUN_RECORD_SCHEMA, load_run_record
from games.blokus_duo.config import FULL_CONFIG, MICRO_CONFIG
from games.blokus_duo.pieces import orientation_table_hash

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """Import a module from ``scripts/`` (which is not a package).

    Args:
        name: Module file stem under ``scripts/``.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass resolution needs the sys.modules entry
    spec.loader.exec_module(module)
    return module


GATE = load_script("micro_exit_gate")
DRIVER = load_script("run_micro")


# --- synthetic evidence ---------------------------------------------------------


def write_record(tmp_path: Path, *, policy: list[float], value: list[float], **extra) -> dict:
    """Write a synthetic run record and read it back through the gate's reader.

    The predicates must consume *persisted* evidence, so even the synthetic
    fixtures go through the file and ``load_run_record``.

    Args:
        tmp_path: Directory to write ``run_record.json`` into.
        policy: Per-step policy losses.
        value: Per-step value losses.
        **extra: Extra top-level record fields (e.g. ``game_identity``).

    Returns:
        The parsed record.
    """
    steps = [
        {
            "step": i,
            "policy_loss": p,
            "value_loss": v,
            "aux_loss": 0.1,
            "total_loss": p + v,
            "learning_rate": 0.02,
            "window_size": 32,
            "games_played": i + 1,
        }
        for i, (p, v) in enumerate(zip(policy, value, strict=True))
    ]
    record = {"schema": RUN_RECORD_SCHEMA, "run_name": "synthetic", "steps": steps, **extra}
    path = tmp_path / GATE.RUN_RECORD_NAME
    path.write_text(json.dumps(record))
    return load_run_record(path)


def flat(head_value: float, tail_value: float, head_steps: int = 2, tail_steps: int = 2):
    """Build a two-window loss series with exact window means.

    Args:
        head_value: Value repeated across the head window.
        tail_value: Value repeated across the tail window.
        head_steps: Head window length.
        tail_steps: Tail window length.

    Returns:
        The series, of length ``head_steps + tail_steps`` (the windows are then
        exactly disjoint and their means are exactly the two values).
    """
    return [head_value] * head_steps + [tail_value] * tail_steps


def pairs_scoring(*scores: float) -> list[PairResult]:
    """Build synthetic pair results with the given per-pair scores for agent A.

    Args:
        *scores: Agent A's score in each pair, in ``[0, 2]``.

    Returns:
        One :class:`~core.runner.PairResult` per score. The embedded game
        records are placeholders: the predicate consumes ``score_a`` only.

    Raises:
        ValueError: If a score is outside ``[0, 2]``.
    """
    results = []
    for i, score in enumerate(scores):
        if not 0.0 <= score <= 2.0:
            raise ValueError(f"pair score {score} is outside [0, 2]")
        stub = GameRecord(utilities=(0.0, 0.0), plies=6, opening=0)
        results.append(
            PairResult(pair_index=i, score_a=score, score_b=2.0 - score, games=(stub, stub))
        )
    return results


# --- loss predicates (§12 M2.5 predicates 2–3) ----------------------------------


def test_loss_series_reads_the_persisted_components(tmp_path):
    """The gate's reader mirrors ``RunRecord.loss_series`` over the written file."""
    record = write_record(tmp_path, policy=[1.0, 0.5], value=[0.4, 0.2])
    assert GATE.loss_series(record, "policy_loss") == [1.0, 0.5]
    assert GATE.loss_series(record, "value_loss") == [0.4, 0.2]
    with pytest.raises(KeyError, match="unknown loss component"):
        GATE.loss_series(record, "not_a_loss")


def test_loss_series_rejects_a_hole_in_the_evidence(tmp_path):
    """A missing or non-numeric entry must raise, never average to something plausible."""
    path = tmp_path / GATE.RUN_RECORD_NAME
    path.write_text(
        json.dumps({"schema": RUN_RECORD_SCHEMA, "steps": [{"step": 0, "value_loss": 1.0}]})
    )
    with pytest.raises(ValueError, match="no 'policy_loss' entry"):
        GATE.loss_series(load_run_record(path), "policy_loss")

    path.write_text(
        json.dumps({"schema": RUN_RECORD_SCHEMA, "steps": [{"step": 0, "policy_loss": "nan-ish"}]})
    )
    with pytest.raises(ValueError, match="non-numeric policy_loss"):
        GATE.loss_series(load_run_record(path), "policy_loss")

    path.write_text(json.dumps({"schema": RUN_RECORD_SCHEMA}))
    with pytest.raises(ValueError, match="no 'steps' list"):
        GATE.loss_series(load_run_record(path), "policy_loss")


def test_policy_predicate_passes_when_the_tail_window_falls_far_enough(tmp_path):
    """Ratio 0.5 against the pinned 0.70 bound: PASS, with both window means recorded."""
    record = write_record(tmp_path, policy=flat(1.0, 0.5), value=flat(1.0, 1.0))
    predicate = GATE.loss_predicate(record, "policy_loss", 2, 2, 0.7)

    assert predicate.passed
    assert predicate.name == "policy_loss"
    assert predicate.measured == pytest.approx(0.5)
    assert predicate.threshold == 0.7
    assert predicate.comparator == "<="
    assert predicate.detail["head_mean"] == pytest.approx(1.0)
    assert predicate.detail["tail_mean"] == pytest.approx(0.5)
    assert predicate.detail["recorded_steps"] == 4


def test_policy_predicate_fails_when_the_tail_window_barely_falls(tmp_path):
    """Ratio 0.9 > 0.70: FAIL — and the measured number is still reported."""
    record = write_record(tmp_path, policy=flat(1.0, 0.9), value=flat(1.0, 0.5))
    predicate = GATE.loss_predicate(record, "policy_loss", 2, 2, 0.7)
    assert not predicate.passed
    assert predicate.measured == pytest.approx(0.9)


def test_value_predicate_fails_on_its_own_looser_bound(tmp_path):
    """Predicates 2–3 are separate: 0.75 passes the value bound and fails the policy one."""
    record = write_record(tmp_path, policy=flat(1.0, 0.75), value=flat(1.0, 0.9))
    policy = GATE.loss_predicate(record, "policy_loss", 2, 2, 0.7)
    value = GATE.loss_predicate(record, "value_loss", 2, 2, 0.8)

    assert not policy.passed  # 0.75 > 0.70
    assert not value.passed  # 0.90 > 0.80
    assert GATE.loss_predicate(record, "policy_loss", 2, 2, 0.8).passed  # 0.75 <= 0.80


@pytest.mark.parametrize("max_ratio", [0.7, 0.8])
def test_a_ratio_exactly_at_the_bound_passes(tmp_path, max_ratio):
    """The pinned comparator is ``<=``: exactly at the bound is a PASS, not a FAIL."""
    record = write_record(tmp_path, policy=flat(1.0, max_ratio), value=flat(1.0, max_ratio))
    predicate = GATE.loss_predicate(record, "policy_loss", 2, 2, max_ratio)
    assert predicate.measured == max_ratio  # exact: means of 1.0s and of equal values
    assert predicate.passed


@pytest.mark.parametrize("max_ratio", [0.7, 0.8])
def test_a_ratio_one_ulp_above_the_bound_fails(tmp_path, max_ratio):
    """...and the very next representable ratio above it is a FAIL."""
    above = math.nextafter(max_ratio, 1.0)
    record = write_record(tmp_path, policy=flat(1.0, above), value=flat(1.0, above))
    predicate = GATE.loss_predicate(record, "policy_loss", 2, 2, max_ratio)
    assert predicate.measured > max_ratio
    assert not predicate.passed


def test_windows_exactly_filling_the_record_are_disjoint_and_evaluable(tmp_path):
    """``head + tail`` steps is the shortest evaluable record — the windows just meet."""
    record = write_record(tmp_path, policy=[1.0, 1.0, 0.5, 0.5], value=[1.0] * 4)
    predicate = GATE.loss_predicate(record, "policy_loss", 2, 2, 0.7)
    assert predicate.detail["head_mean"] == pytest.approx(1.0)
    assert predicate.detail["tail_mean"] == pytest.approx(0.5)


def test_a_truncated_record_fails_loudly_instead_of_scoring(tmp_path):
    """Fewer steps than head+tail: raise. Overlapping windows would flatter the ratio."""
    record = write_record(tmp_path, policy=[1.0, 1.0, 0.5], value=[1.0] * 3)
    with pytest.raises(ValueError, match="fewer than the pinned"):
        GATE.loss_predicate(record, "policy_loss", 2, 2, 0.7)

    # The pinned 200/200 windows against a short real-shaped record: same rejection.
    pinned = load_run_config().loss_predicates
    short = write_record(tmp_path, policy=[1.0] * 399, value=[1.0] * 399)
    with pytest.raises(ValueError, match="fewer than the pinned"):
        GATE.loss_predicate(
            short,
            "policy_loss",
            pinned.head_window_steps,
            pinned.tail_window_steps,
            pinned.policy_max_ratio,
        )


def test_a_falling_series_that_only_overlaps_would_have_passed(tmp_path):
    """Why the rejection matters: the overlapping read of the same record 'passes'."""
    series = [1.0, 1.0, 0.5]
    with pytest.raises(ValueError, match="fewer than the pinned"):
        GATE.window_means(series, 2, 2)
    head, tail = GATE.window_means(series, 2, 1)  # a *shorter* tail window is evaluable
    assert (head, tail) == (1.0, 0.5)


def test_a_non_positive_head_window_mean_is_rejected(tmp_path):
    """A zero head mean makes the ratio undefined; the gate must not divide anyway."""
    record = write_record(tmp_path, policy=flat(0.0, 0.0), value=flat(1.0, 0.5))
    with pytest.raises(ValueError, match="not meaningful"):
        GATE.loss_predicate(record, "policy_loss", 2, 2, 0.7)


def test_window_means_reject_a_non_positive_window():
    """Window lengths come from config, but the helper is loud about degenerate input."""
    with pytest.raises(ValueError, match="windows must be positive"):
        GATE.window_means([1.0] * 10, 0, 2)


# --- strength predicate (§12 M2.5 predicate 1) ----------------------------------


def test_score_rate_counts_draws_as_half():
    """``total_score_a / (2 × n_pairs)`` — the runner already scores a draw 0.5."""
    predicate = GATE.strength_predicate(pairs_scoring(2.0, 1.0, 0.0), 0.5)
    assert predicate.measured == pytest.approx(0.5)
    assert predicate.detail == {"total_score": 3.0, "games": 6, "pairs": 3}
    assert predicate.comparator == ">="


def test_strength_passes_at_exactly_the_pinned_floor():
    """The pinned comparator is ``>=``: exactly 0.70 is a PASS."""
    # 10 pairs = 20 games; 14 points is exactly 0.70.
    predicate = GATE.strength_predicate(pairs_scoring(*([2.0] * 7 + [0.0] * 3)), 0.7)
    assert predicate.measured == 0.7
    assert predicate.passed


def test_strength_fails_just_below_the_pinned_floor():
    """One half-point less over the same 20 games is a FAIL."""
    predicate = GATE.strength_predicate(pairs_scoring(*([2.0] * 6 + [1.5] + [0.0] * 3)), 0.7)
    assert predicate.measured == pytest.approx(0.675)
    assert not predicate.passed


def test_strength_rejects_an_empty_match():
    """A zero-pair match must not report a score rate at all."""
    with pytest.raises(ValueError, match="no pairs played"):
        GATE.strength_predicate([], 0.7)


# --- verdict logic --------------------------------------------------------------


def predicate(name: str, passed: bool):
    """Build a stub predicate with a known verdict.

    Args:
        name: Predicate id.
        passed: Whether it holds.

    Returns:
        The stub :class:`Predicate`.
    """
    return GATE.Predicate(
        name=name, passed=passed, measured=1.0, threshold=1.0, comparator=">=", detail={}
    )


@pytest.mark.parametrize("strength", [True, False])
@pytest.mark.parametrize("policy", [True, False])
@pytest.mark.parametrize("value", [True, False])
def test_verdict_is_pass_iff_all_three_predicates_hold(strength, policy, value):
    """All eight quadrants of the §12 M2.5 conjunction."""
    predicates = [
        predicate(GATE.STRENGTH, strength),
        predicate(GATE.POLICY_LOSS, policy),
        predicate(GATE.VALUE_LOSS, value),
    ]
    verdict = GATE.combine(predicates, {"run_name": "synthetic"})
    assert verdict.passed is (strength and policy and value)
    assert verdict.to_dict()["verdict"] == ("PASS" if verdict.passed else "FAIL")
    assert [p["verdict"] for p in verdict.to_dict()["predicates"]] == [
        "PASS" if p else "FAIL" for p in (strength, policy, value)
    ]
    assert verdict.predicate(GATE.STRENGTH).passed is strength
    with pytest.raises(KeyError):
        verdict.predicate("throughput")


def test_an_empty_conjunction_is_not_a_pass():
    """``all([])`` is True; a gate that evaluated nothing must raise instead."""
    with pytest.raises(ValueError, match="empty gate"):
        GATE.combine([], {"run_name": "synthetic"})


# --- identity and protocol guards -----------------------------------------------


def test_expected_identity_is_the_micro_instance_hash():
    """Invariant 4: the hash is re-derived from the config, not read off the artifact."""
    identity = GATE.expected_identity(load_run_config())
    assert identity == {
        "game": "blokus_duo",
        "game_config": "MICRO_CONFIG",
        "orientation_hash": orientation_table_hash(MICRO_CONFIG),
    }
    assert identity["orientation_hash"] != orientation_table_hash(FULL_CONFIG)


@pytest.mark.parametrize(
    "field, wrong",
    [
        ("orientation_hash", orientation_table_hash(FULL_CONFIG)),
        ("game_config", "FULL_CONFIG"),
        ("game", "othello"),
    ],
)
def test_identity_mismatch_is_rejected(field, wrong):
    """A mismatched artifact must fail loudly, never be silently scored."""
    expected = GATE.expected_identity(load_run_config())
    GATE.check_identity(expected, dict(expected), "artifact")  # the matching case is quiet
    with pytest.raises(ValueError, match=f"{field} is"):
        GATE.check_identity(expected, {**expected, field: wrong}, "artifact")
    with pytest.raises(ValueError, match=f"{field} is"):
        GATE.check_identity(expected, {k: v for k, v in expected.items() if k != field}, "artifact")


def test_expected_identity_rejects_a_game_the_gate_cannot_drive():
    """The gate is the micro-Blokus gate, not a generic runner."""
    foreign = SimpleNamespace(game="othello", game_config="DEFAULT")
    with pytest.raises(ValueError, match="blokus_duo only"):
        GATE.expected_identity(foreign)


def test_check_protocol_accepts_the_pinned_evaluation_form():
    """The pinned config is exactly the protocol this gate realizes."""
    GATE.check_protocol(load_run_config())


@pytest.mark.parametrize(
    "override, message",
    [
        ({"agent_form": "rung3_mobility"}, "agent_form"),
        ({"opponent": "rung2_largest_piece"}, "opponent"),
        ({"move_selection": "sample_n"}, "move_selection"),
        ({"root_noise": True}, "root_noise"),
    ],
)
def test_check_protocol_rejects_a_substituted_evaluation_form(override, message):
    """Anything but the pinned form is a doc-first change, not a silent substitution."""
    cfg = load_run_config()
    cfg = dataclasses.replace(cfg, evaluation=dataclasses.replace(cfg.evaluation, **override))
    with pytest.raises(ValueError, match=message):
        GATE.check_protocol(cfg)


def test_checkpoint_selection_takes_the_one_checkpoint_of_the_configured_kind(tmp_path):
    """``training.checkpoint_selection`` picks exactly one artifact — never 'the last one'."""
    (tmp_path / "checkpoint_final.pt").write_text("stub")
    record = {
        "checkpoints": [
            {"step": 2, "kind": "periodic", "path": str(tmp_path / "checkpoint_step2.pt")},
            {"step": 4, "kind": "final", "path": str(tmp_path / "checkpoint_final.pt")},
        ]
    }
    path, entry = GATE.select_checkpoint(record, "final", tmp_path)
    assert path == tmp_path / "checkpoint_final.pt"
    assert entry["step"] == 4

    with pytest.raises(ValueError, match="0 'final' checkpoint"):
        GATE.select_checkpoint({"checkpoints": []}, "final", tmp_path)
    doubled = {"checkpoints": record["checkpoints"] + [record["checkpoints"][1]]}
    with pytest.raises(ValueError, match="2 'final' checkpoint"):
        GATE.select_checkpoint(doubled, "final", tmp_path)


def test_checkpoint_selection_relocates_a_copied_run_but_not_a_missing_one(tmp_path):
    """A run dir copied off the GPU box still resolves; a truly absent file raises."""
    (tmp_path / "checkpoint_final.pt").write_text("stub")
    record = {"checkpoints": [{"step": 4, "kind": "final", "path": "/nowhere/checkpoint_final.pt"}]}
    path, _ = GATE.select_checkpoint(record, "final", tmp_path)
    assert path == tmp_path / "checkpoint_final.pt"

    gone = {"checkpoints": [{"step": 4, "kind": "final", "path": "/nowhere/checkpoint_other.pt"}]}
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GATE.select_checkpoint(gone, "final", tmp_path)


# --- completeness of the persisted evidence -------------------------------------

# The loss levels the committed PASS verdict was computed from
# (docs/bench/m2_5-exit-gate.md): head/tail window means for each component. The
# synthetic full-scale record below interpolates between them, so the
# completeness tests run against realistic evidence rather than toy numbers.
RECORDED_POLICY_LOSS = (1.7778, 1.0050)
RECORDED_VALUE_LOSS = (0.4225, 0.1335)


def full_run_record(
    cfg: RunConfig,
    *,
    steps: int | None = None,
    games: int | None = None,
    checkpoint_step: int | None = None,
) -> dict:
    """Build a run record for ``cfg`` at the pinned scale, complete unless told otherwise.

    Identity, embedded config and evaluation protocol are the real ones, so the
    gate's earlier guards all pass and the completeness check is what is under
    test. The losses fall linearly between the committed head/tail means.

    Args:
        cfg: The run config the record claims to have run under.
        steps: Learner steps to record (default: the configured total).
        games: Self-play games to record (default: the configured total).
        checkpoint_step: Step of the recorded ``final`` checkpoint (default: the
            configured final step).

    Returns:
        The record, as a plain dict ready to be written.
    """
    n_steps = cfg.training.learner_steps if steps is None else steps
    n_games = cfg.training.games if games is None else games
    total = max(cfg.training.learner_steps - 1, 1)
    step_entries = []
    for i in range(n_steps):
        fraction = min(i, total) / total
        policy = RECORDED_POLICY_LOSS[0] + fraction * (
            RECORDED_POLICY_LOSS[1] - RECORDED_POLICY_LOSS[0]
        )
        value = RECORDED_VALUE_LOSS[0] + fraction * (
            RECORDED_VALUE_LOSS[1] - RECORDED_VALUE_LOSS[0]
        )
        step_entries.append(
            {
                "step": i,
                "policy_loss": policy,
                "value_loss": value,
                "aux_loss": 0.1,
                "total_loss": policy + value,
                "learning_rate": 0.02,
                "window_size": cfg.training.replay_window,
                "games_played": min(i + 1, n_games),
            }
        )
    return {
        "schema": RUN_RECORD_SCHEMA,
        "run_name": cfg.name,
        "run_seed": cfg.run_seed,
        "config": cfg.to_dict(),
        "game_identity": GATE.expected_identity(cfg),
        "device": "cpu",
        "steps": step_entries,
        "games": [
            {
                "game_index": i,
                "plies": 6,
                "samples": 6,
                "utilities": [1.0, -1.0],
                "moves": [0, 1, 2, 3, 4, 5],
            }
            for i in range(n_games)
        ],
        "checkpoints": [
            {
                "step": cfg.training.learner_steps if checkpoint_step is None else checkpoint_step,
                "kind": "final",
                "path": "checkpoint_final.pt",
            }
        ],
        "timing": {},
    }


def write_run_dir(tmp_path: Path, record: dict) -> Path:
    """Write a record into a run directory the gate can be pointed at.

    Args:
        tmp_path: Parent directory for the run dir.
        record: The record to write.

    Returns:
        The run directory.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    (run_dir / GATE.RUN_RECORD_NAME).write_text(json.dumps(record))
    return run_dir


def gate_exit_code(tmp_path: Path, record: dict) -> int:
    """Write a record into a run dir and run the gate over it against the pinned config.

    Args:
        tmp_path: Parent directory for the run dir.
        record: The record to write.

    Returns:
        The process exit code ``GATE.main`` would return.
    """
    run_dir = write_run_dir(tmp_path, record)
    code = GATE.main(
        ["--config", str(MICRO_RUN_CONFIG_PATH), "--run-dir", str(run_dir), "--device", "cpu"]
    )
    if code == 2:
        assert not (run_dir / GATE.VERDICT_NAME).exists()
    return code


def test_completeness_accepts_the_committed_2000_by_2000_run():
    """The real protocol must survive the check: 2,000 steps, 2,000 games, step 2000."""
    cfg = load_run_config()
    record = full_run_record(cfg)

    GATE.check_completeness(record, cfg, "record")  # the complete case is quiet
    GATE.check_final_step(cfg.training.learner_steps, cfg, "checkpoint")
    assert len(record["steps"]) == cfg.training.learner_steps == 2000
    assert len(record["games"]) == cfg.training.games == 2000

    # ...and the loss predicates it feeds still PASS at the recorded levels.
    for name, bound in (
        ("policy_loss", cfg.loss_predicates.policy_max_ratio),
        ("value_loss", cfg.loss_predicates.value_max_ratio),
    ):
        predicate = GATE.loss_predicate(
            record,
            name,
            cfg.loss_predicates.head_window_steps,
            cfg.loss_predicates.tail_window_steps,
            bound,
        )
        assert predicate.passed


def test_a_record_truncated_to_400_steps_is_not_evaluable(tmp_path):
    """The reviewer's reproduction: 400 steps clears head+tail (200+200) but is not the run.

    The rejection is matched by *message*, not merely by exit code: before the
    completeness check existed the gate scored this record and only later tripped
    over the missing checkpoint, which also exits 2.
    """
    cfg = load_run_config()
    record = full_run_record(cfg, steps=400, games=400)
    assert len(record["steps"]) >= (
        cfg.loss_predicates.head_window_steps + cfg.loss_predicates.tail_window_steps
    )

    with pytest.raises(ValueError, match="400 recorded learner steps"):
        GATE.check_completeness(record, cfg, "record")
    run_dir = write_run_dir(tmp_path, record)
    with pytest.raises(ValueError, match="400 recorded learner steps"):
        GATE.run_gate(run_dir, config_path=MICRO_RUN_CONFIG_PATH, device="cpu")
    assert gate_exit_code(tmp_path, record) == 2


def test_a_record_of_the_right_length_with_broken_step_ids_is_not_evaluable(tmp_path):
    """Right count, wrong sequence: a gap or a duplicate is not one whole run."""
    cfg = load_run_config()
    gapped = full_run_record(cfg)
    gapped["steps"][1500]["step"] = 1501  # duplicate id, and a gap at 1500
    with pytest.raises(ValueError, match="ids are not the contiguous sequence"):
        GATE.check_completeness(gapped, cfg, "record")
    assert gate_exit_code(tmp_path, gapped) == 2

    dropped = full_run_record(cfg)
    del dropped["steps"][7]  # a hole, refilled by appending a duplicate tail step
    dropped["steps"].append(dict(dropped["steps"][-1]))
    assert len(dropped["steps"]) == cfg.training.learner_steps
    with pytest.raises(ValueError, match="ids are not the contiguous sequence"):
        GATE.check_completeness(dropped, cfg, "record")


def test_a_short_or_missing_games_list_is_not_evaluable(tmp_path):
    """Self-play games are pinned too — the full step count cannot vouch for them."""
    cfg = load_run_config()
    short = full_run_record(cfg, games=1200)
    with pytest.raises(ValueError, match="1200 recorded self-play games"):
        GATE.check_completeness(short, cfg, "record")
    assert gate_exit_code(tmp_path, short) == 2

    absent = full_run_record(cfg)
    del absent["games"]
    with pytest.raises(ValueError, match="no 'games' list"):
        GATE.check_completeness(absent, cfg, "record")


def test_a_final_step_that_undercounts_the_games_is_not_evaluable():
    """The cross-check: a full games list cannot rescue steps paced against a shorter run."""
    cfg = load_run_config()
    record = full_run_record(cfg)
    record["steps"][-1]["games_played"] = cfg.training.games - 1
    with pytest.raises(ValueError, match="games_played"):
        GATE.check_completeness(record, cfg, "record")


def test_a_checkpoint_step_disagreeing_with_the_config_is_not_evaluable(tmp_path):
    """A mid-run checkpoint is not the weights the persisted loss series describes."""
    cfg = load_run_config()
    GATE.check_final_step(cfg.training.learner_steps, cfg, "checkpoint")  # the match is quiet
    for wrong in (1000, cfg.training.learner_steps - 1, None):
        with pytest.raises(ValueError, match="checkpoint step is"):
            GATE.check_final_step(wrong, cfg, "checkpoint")
    assert gate_exit_code(tmp_path, full_run_record(cfg, checkpoint_step=1000)) == 2


def test_gate_and_driver_agree_on_the_artifact_contract():
    """The reader's schema/name constants are pinned to the writer's."""
    assert GATE.CHECKPOINT_SCHEMA == DRIVER.CHECKPOINT_SCHEMA
    assert GATE.RUN_RECORD_NAME == DRIVER.RUN_RECORD_NAME


def test_cli_defaults_to_the_pinned_config():
    """No arguments → the pre-registered config, CPU inference, verdict beside the run."""
    args = GATE.parse_args([])
    assert args.config == MICRO_RUN_CONFIG_PATH
    assert args.run_dir is None and args.out is None
    assert args.device == "cpu"


# --- end-to-end smoke -----------------------------------------------------------


def tiny_config() -> RunConfig:
    """Build a tiny but coherent run config off the pinned micro config.

    Only the budget shrinks: the game instance, the D7 constants and the pinned
    comparator thresholds stay exactly as registered, so the gate under test is
    the pinned gate.

    Returns:
        The validated tiny :class:`~core.runconfig.RunConfig`.
    """
    cfg = load_run_config()
    return dataclasses.replace(
        cfg,
        self_play=dataclasses.replace(cfg.self_play, sims=4),
        training=dataclasses.replace(
            cfg.training,
            games=4,
            learner_steps=4,
            steps_per_game=1,
            batch_size=8,
            replay_window=40,
            warmup_steps=0,
            cosine_total_steps=8,
        ),
        evaluation=dataclasses.replace(cfg.evaluation, sims=4, n_pairs=2),
        loss_predicates=dataclasses.replace(
            cfg.loss_predicates, head_window_steps=2, tail_window_steps=2
        ),
    )


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Run the tiny loop once and share it across the end-to-end tests.

    Args:
        tmp_path_factory: pytest's session-scoped temp-directory factory.

    Returns:
        ``(cfg, config_path, run_dir)`` for the completed run.
    """
    cfg = tiny_config()
    base = tmp_path_factory.mktemp("gate_run")
    config_path = base / "tiny.json"
    config_path.write_text(json.dumps(cfg.to_dict()))
    run_dir = base / "run"
    DRIVER.run_loop(cfg, run_dir=run_dir, device="cpu")
    return cfg, config_path, run_dir


@pytest.mark.slow
def test_gate_runs_end_to_end_and_writes_a_well_formed_verdict(tiny_run, tmp_path):
    """Both halves are computed from the persisted artifacts and every input recorded.

    Mechanics only: a 4-game run says nothing about the pre-registered
    thresholds, so this asserts the shape of the verdict, never its value.
    """
    cfg, config_path, run_dir = tiny_run
    verdict = GATE.run_gate(run_dir, config_path=config_path, device="cpu")

    assert [p.name for p in verdict.predicates] == [
        GATE.STRENGTH,
        GATE.POLICY_LOSS,
        GATE.VALUE_LOSS,
    ]
    assert verdict.passed is all(p.passed for p in verdict.predicates)

    strength = verdict.predicate(GATE.STRENGTH)
    assert strength.threshold == cfg.evaluation.min_score_rate == 0.7
    assert strength.detail["pairs"] == cfg.evaluation.n_pairs
    assert strength.detail["games"] == 2 * cfg.evaluation.n_pairs
    assert 0.0 <= strength.measured <= 1.0

    for name, bound in (
        (GATE.POLICY_LOSS, cfg.loss_predicates.policy_max_ratio),
        (GATE.VALUE_LOSS, cfg.loss_predicates.value_max_ratio),
    ):
        loss = verdict.predicate(name)
        assert loss.threshold == bound
        assert loss.detail["recorded_steps"] == cfg.training.learner_steps
        # Read from the file, not recomputed: the means match the persisted series.
        series = GATE.loss_series(load_run_record(run_dir / GATE.RUN_RECORD_NAME), name)
        assert loss.detail["head_mean"] == pytest.approx(sum(series[:2]) / 2)
        assert loss.detail["tail_mean"] == pytest.approx(sum(series[-2:]) / 2)

    inputs = verdict.inputs
    assert inputs["run_seed"] == cfg.run_seed
    assert inputs["evaluation"]["eval_seed"] == cfg.evaluation.eval_seed != cfg.run_seed
    assert inputs["evaluation"]["root_noise"] is False
    assert inputs["evaluation"]["move_selection"] == "argmax_n"
    assert inputs["config"] == cfg.to_dict()
    assert inputs["game_identity"]["orientation_hash"] == orientation_table_hash(MICRO_CONFIG)
    assert inputs["checkpoint"]["kind"] == "final"
    assert inputs["checkpoint"]["step"] == cfg.training.learner_steps
    assert Path(inputs["checkpoint"]["path"]).exists()
    assert len(inputs["measurements"]["pair_scores"]) == cfg.evaluation.n_pairs
    assert set(inputs["measurements"]["elo"]) == {
        cfg.evaluation.agent_form,
        cfg.evaluation.opponent,
    }
    assert inputs["measurements"]["elo"][cfg.evaluation.opponent] == 0.0  # rung-1 anchor

    written = verdict.write(tmp_path / GATE.VERDICT_NAME)
    persisted = json.loads(written.read_text())
    assert persisted["schema"] == GATE.VERDICT_SCHEMA
    assert persisted["verdict"] in ("PASS", "FAIL")
    assert "VERDICT:" in verdict.render()


@pytest.mark.slow
def test_gate_evaluation_set_is_reproducible_from_the_eval_seed(tiny_run):
    """Same evaluation seed → the same fixed paired set, twice."""
    _, config_path, run_dir = tiny_run
    first = GATE.run_gate(run_dir, config_path=config_path, device="cpu")
    again = GATE.run_gate(run_dir, config_path=config_path, device="cpu")
    assert (
        first.inputs["measurements"]["pair_scores"] == again.inputs["measurements"]["pair_scores"]
    )
    assert first.predicate(GATE.STRENGTH).measured == again.predicate(GATE.STRENGTH).measured


@pytest.mark.slow
def test_main_writes_the_verdict_beside_the_run_and_exits_on_it(tiny_run, capsys):
    """Exit code carries the verdict, so the script is usable as a gate."""
    _, config_path, run_dir = tiny_run
    code = GATE.main(["--config", str(config_path), "--run-dir", str(run_dir), "--device", "cpu"])
    out = capsys.readouterr().out

    persisted = json.loads((run_dir / GATE.VERDICT_NAME).read_text())
    assert code == (0 if persisted["verdict"] == "PASS" else 1)
    assert f"VERDICT: {persisted['verdict']}" in out
    assert str(run_dir / GATE.VERDICT_NAME) in out


def copy_run(tiny_run, tmp_path) -> Path:
    """Copy a completed run so a test can tamper with its artifacts.

    The record's checkpoint paths are repointed at the copy (they are written
    absolute) and any verdict written by an earlier test is dropped, so the
    tampering tests observe only what *this* run dir contains.

    Args:
        tiny_run: The shared tiny-run fixture value.
        tmp_path: Destination parent directory.

    Returns:
        The copied run directory.
    """
    _, _, run_dir = tiny_run
    target = tmp_path / "tampered"
    shutil.copytree(run_dir, target)
    (target / GATE.VERDICT_NAME).unlink(missing_ok=True)
    path = target / GATE.RUN_RECORD_NAME
    record = json.loads(path.read_text())
    for entry in record["checkpoints"]:
        entry["path"] = str(target / Path(entry["path"]).name)
    path.write_text(json.dumps(record))
    return target


@pytest.mark.slow
def test_gate_rejects_a_record_whose_identity_does_not_match(tiny_run, tmp_path):
    """A run record stamped with the full game's orientation table is not evidence."""
    _, config_path, _ = tiny_run
    target = copy_run(tiny_run, tmp_path)
    path = target / GATE.RUN_RECORD_NAME
    record = json.loads(path.read_text())
    record["game_identity"]["orientation_hash"] = orientation_table_hash(FULL_CONFIG)
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="orientation_hash is"):
        GATE.run_gate(target, config_path=config_path, device="cpu")
    code = GATE.main(["--config", str(config_path), "--run-dir", str(target), "--device", "cpu"])
    assert code == 2  # not 0: a gate never passes evidence it could not check
    assert not (target / GATE.VERDICT_NAME).exists()


@pytest.mark.slow
def test_gate_rejects_a_checkpoint_from_another_run(tiny_run, tmp_path):
    """The weights and the loss series must come from the same run."""
    cfg, config_path, _ = tiny_run
    target = copy_run(tiny_run, tmp_path)
    torch = pytest.importorskip("torch")
    checkpoint = target / "checkpoint_final.pt"
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    blob["run_seed"] = cfg.run_seed + 1
    torch.save(blob, checkpoint)

    with pytest.raises(ValueError, match="run_seed"):
        GATE.run_gate(target, config_path=config_path, device="cpu")


@pytest.mark.slow
def test_gate_rejects_a_checkpoint_written_before_the_end_of_the_run(tiny_run, tmp_path):
    """The blob's own step must be the configured final step, not just the record's entry."""
    cfg, config_path, _ = tiny_run
    target = copy_run(tiny_run, tmp_path)
    torch = pytest.importorskip("torch")
    checkpoint = target / "checkpoint_final.pt"
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert blob["step"] == cfg.training.learner_steps
    blob["step"] = cfg.training.learner_steps - 1
    torch.save(blob, checkpoint)

    with pytest.raises(ValueError, match="checkpoint step is"):
        GATE.run_gate(target, config_path=config_path, device="cpu")
    code = GATE.main(["--config", str(config_path), "--run-dir", str(target), "--device", "cpu"])
    assert code == 2
    assert not (target / GATE.VERDICT_NAME).exists()


@pytest.mark.slow
def test_gate_rejects_a_run_registered_under_another_config(tiny_run):
    """Scoring a run against a protocol it did not run under is not the pre-registered gate."""
    _, _, run_dir = tiny_run
    with pytest.raises(ValueError, match="pre-registered"):
        GATE.run_gate(run_dir, config_path=MICRO_RUN_CONFIG_PATH, device="cpu")


@pytest.mark.slow
def test_gate_rejects_a_truncated_run_record(tiny_run, tmp_path):
    """Dropping steps below the pinned windows exits loudly, before any game is played."""
    _, config_path, _ = tiny_run
    target = copy_run(tiny_run, tmp_path)
    path = target / GATE.RUN_RECORD_NAME
    record = json.loads(path.read_text())
    record["steps"] = record["steps"][:3]  # head(2) + tail(2) needs 4
    path.write_text(json.dumps(record))

    code = GATE.main(["--config", str(config_path), "--run-dir", str(target), "--device", "cpu"])
    assert code == 2
    assert not (target / GATE.VERDICT_NAME).exists()


@pytest.mark.slow
def test_gate_reports_a_missing_checkpoint(tiny_run, tmp_path):
    """No weights, no verdict — the strength half cannot be faked."""
    _, config_path, _ = tiny_run
    target = copy_run(tiny_run, tmp_path)
    (target / "checkpoint_final.pt").unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        GATE.run_gate(target, config_path=config_path, device="cpu")
