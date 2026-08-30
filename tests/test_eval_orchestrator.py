"""The M4 eval orchestrator's config/correctness spine (design doc §9; tasks/m4/009.1).

Covers: :class:`~core.eval_run.EvalConfig`'s JSON-parse discipline, the
launch-provenance write + the relaunch guard's "no override, refuse and name
every differing field" behavior (both the config-field and the
protocol-registry-fingerprint drift paths), the membership arithmetic that
keeps ``ckpt-0.pt``/the rolling ``resume.pt`` snapshot/the ``latest`` pointer
structurally unschedulable, the cell-scheduling arithmetic against a real
game's declared eval profile, and a from-scratch stub profile proving the
mechanism never needs a ``core/`` or Blokus-specific edit to support a second
game (S2.1). The watch/resume loop itself and the report/CLI integration are
later subtasks (9.2/9.3); nothing here drives an actual game.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

import core.eval_protocol as eval_protocol
import core.eval_run as eval_run
from core.checkpoint import LATEST_FILENAME, RESUME_FILENAME
from core.eval_profile import EvalProfile
from core.eval_run import (
    RUNG8_CANDIDATE_FORM,
    EvalConfig,
    EvalRelaunchRefusedError,
    agent_identity,
    cell_seed,
    checkpoint_dir,
    load_eval_config,
    required_cell_ids,
    resolve_eval_launch,
    schedulable_versions,
)
from core.eval_store import build_cell_id
from core.seeding import PURPOSE_EVAL, derive_seed
from games.registry import build_eval_profile

M4_EVAL_CONFIG_PATH = "configs/m4_eval.json"


def _make_config(run_dir: str, **overrides) -> EvalConfig:
    """A small, schema-valid :class:`EvalConfig` for tests (cheap, admissible B)."""
    fields = {
        "run_dir": run_dir,
        "pairs_per_cell": 2,
        "eval_sims": 8,
        "rung8_lag_divisor": 4,
        "rung8_earliest_version": 1,
        "forms": (5, 6, 7),
        "eval_seed": 123,
        "device": "cpu",
        "bootstrap_b": 39,  # smallest admissible B (39 mod 40 == 39).
        "schema_version": eval_run.EVAL_CONFIG_SCHEMA_VERSION,
    }
    fields.update(overrides)
    return EvalConfig(**fields)


class _StubAgent:
    """A minimal ``core.runner.AgentFactory``-shaped stand-in: no game needed.

    ``EvalProfile.rung_identity`` only ever calls ``factory(seed).name`` --
    it never plays a game -- so a fixture proving the profile mechanism is
    game-generic needs nothing more than this.
    """

    def __init__(self, name: str):
        self._name = name

    def __call__(self, seed: int):
        del seed
        return self

    @property
    def name(self) -> str:
        return self._name


# --- EvalConfig: JSON-parse discipline ------------------------------------------


def test_config_doc_agreement_with_core_eval_protocol():
    """configs/m4_eval.json's pinned values equal core.eval_protocol's registry."""
    config = load_eval_config(M4_EVAL_CONFIG_PATH)
    assert config.pairs_per_cell == eval_protocol.PAIRS_PER_CELL
    assert config.eval_sims == eval_protocol.EVAL_SIMS
    assert config.rung8_lag_divisor == eval_protocol.RUNG8_LAG_DIVISOR
    assert config.rung8_earliest_version == eval_protocol.RUNG8_EARLIEST_VERSION
    assert config.bootstrap_b == eval_protocol.BOOTSTRAP_B_PRODUCTION
    assert config.forms == (5, 6, 7)
    assert config.device == "cpu"


def test_config_eval_seed_independent_of_the_watched_run_seed():
    """The M4 harness seed must differ from configs/blokus_duo.json's run_seed."""
    config = load_eval_config(M4_EVAL_CONFIG_PATH)
    run_config = json.loads(open("configs/blokus_duo.json").read())
    assert config.eval_seed != run_config["run_seed"]


def test_eval_config_rejects_unknown_key(tmp_path):
    payload = json.loads(open(M4_EVAL_CONFIG_PATH).read())
    payload["bogus_field"] = 1
    with pytest.raises(ValueError, match="unknown config keys"):
        EvalConfig.from_dict(payload)


def test_eval_config_rejects_missing_key():
    payload = json.loads(open(M4_EVAL_CONFIG_PATH).read())
    del payload["eval_sims"]
    with pytest.raises(ValueError, match="missing config keys"):
        EvalConfig.from_dict(payload)


def test_eval_config_rejects_unrecognized_form():
    with pytest.raises(ValueError, match="unrecognized form"):
        _make_config("runs/x", forms=(5, 8))


def test_eval_config_rejects_duplicate_forms():
    with pytest.raises(ValueError, match="duplicates"):
        _make_config("runs/x", forms=(5, 5, 6))


def test_eval_config_normalizes_forms_ascending():
    config = _make_config("runs/x", forms=(7, 5, 6))
    assert config.forms == (5, 6, 7)


def test_eval_config_rejects_inadmissible_bootstrap_b():
    with pytest.raises(ValueError, match="not admissible"):
        _make_config("runs/x", bootstrap_b=100)


def test_eval_config_rejects_blank_run_dir():
    with pytest.raises(ValueError, match="run_dir"):
        _make_config("   ")


# --- launch provenance + the relaunch guard -------------------------------------


def test_resolve_eval_launch_first_time_writes_provenance(tmp_path):
    run_dir = tmp_path / "run"
    config = _make_config(str(run_dir))

    effective = resolve_eval_launch(run_dir, config)
    assert effective == config

    stored = eval_run.read_eval_provenance(run_dir)
    assert stored.config == config
    assert stored.protocol_version == eval_protocol.PROTOCOL_VERSION
    assert stored.protocol_fingerprint == eval_protocol.protocol_fingerprint()
    assert eval_run.eval_config_path(run_dir).exists()


def test_resolve_eval_launch_identical_relaunch_is_accepted(tmp_path):
    run_dir = tmp_path / "run"
    config = _make_config(str(run_dir))
    resolve_eval_launch(run_dir, config)

    # A second launch with a field-for-field identical config must not raise,
    # and must not rewrite the recorded provenance (never mutated on resume).
    before = eval_run.eval_config_path(run_dir).read_text()
    effective = resolve_eval_launch(run_dir, config)
    after = eval_run.eval_config_path(run_dir).read_text()
    assert effective == config
    assert before == after


def test_resolve_eval_launch_refuses_on_changed_sim_budget(tmp_path):
    run_dir = tmp_path / "run"
    original = _make_config(str(run_dir), eval_sims=64)
    resolve_eval_launch(run_dir, original)

    changed = dataclasses.replace(original, eval_sims=32)
    with pytest.raises(EvalRelaunchRefusedError, match="eval_sims") as excinfo:
        resolve_eval_launch(run_dir, changed)
    assert excinfo.value.config_diff == {"eval_sims": (64, 32)}
    assert excinfo.value.protocol_diff == {}


def test_resolve_eval_launch_refuses_on_changed_forms(tmp_path):
    run_dir = tmp_path / "run"
    original = _make_config(str(run_dir), forms=(5, 6, 7))
    resolve_eval_launch(run_dir, original)

    changed = dataclasses.replace(original, forms=(5, 6))
    with pytest.raises(EvalRelaunchRefusedError, match="forms"):
        resolve_eval_launch(run_dir, changed)


def test_resolve_eval_launch_refuses_on_protocol_fingerprint_drift(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    config = _make_config(str(run_dir))
    resolve_eval_launch(run_dir, config)

    # Simulate a code-level protocol change: the registry's fingerprint moved
    # even though this EvalConfig instance did not, and PROTOCOL_VERSION was
    # not bumped either (the whole point of the fingerprint: it catches drift
    # nobody remembered to also version-bump).
    monkeypatch.setattr(eval_run, "protocol_fingerprint", lambda: "deadbeef" * 8)
    with pytest.raises(EvalRelaunchRefusedError, match="protocol_fingerprint") as excinfo:
        resolve_eval_launch(run_dir, config)
    assert excinfo.value.config_diff == {}
    assert set(excinfo.value.protocol_diff) == {"protocol_fingerprint"}


def test_resolve_eval_launch_never_offers_an_override(tmp_path):
    """No parameter of any kind exists to bypass the refusal (design constraint)."""
    import inspect

    sig = inspect.signature(resolve_eval_launch)
    assert set(sig.parameters) == {"run_dir", "config"}


# --- membership arithmetic: exactly versions 1..K, on disk, right now ----------


def test_schedulable_versions_excludes_v0_snapshot_and_latest(tmp_path):
    ckpt_dir = checkpoint_dir(tmp_path)
    ckpt_dir.mkdir(parents=True)
    for name in ("ckpt-0.pt", "ckpt-1.pt", "ckpt-2.pt", RESUME_FILENAME, LATEST_FILENAME):
        (ckpt_dir / name).write_bytes(b"")

    versions = schedulable_versions(tmp_path, k_total=30)

    assert versions == (1, 2)
    assert 0 not in versions
    # The exclusion is structural, not a filter applied after the fact: the
    # planted resume/latest files are real files sitting right there on disk
    # while the returned versions tuple -- being a tuple of int -- could
    # never literally "contain" either filename in the first place.
    on_disk = {p.name for p in ckpt_dir.iterdir()}
    assert {RESUME_FILENAME, LATEST_FILENAME, "ckpt-0.pt"} <= on_disk


def test_schedulable_versions_excludes_beyond_k_total(tmp_path):
    ckpt_dir = checkpoint_dir(tmp_path)
    ckpt_dir.mkdir(parents=True)
    for v in range(1, 6):
        (ckpt_dir / f"ckpt-{v}.pt").write_bytes(b"")

    assert schedulable_versions(tmp_path, k_total=3) == (1, 2, 3)
    assert schedulable_versions(tmp_path, k_total=30) == (1, 2, 3, 4, 5)


def test_schedulable_versions_empty_checkpoint_dir(tmp_path):
    assert schedulable_versions(tmp_path, k_total=30) == ()


def test_schedulable_versions_rejects_non_positive_k_total(tmp_path):
    with pytest.raises(ValueError, match="k_total"):
        schedulable_versions(tmp_path, k_total=0)


# --- cell-scheduling arithmetic --------------------------------------------------


def test_required_cell_ids_matches_hand_built_list_for_a_mid_run_member():
    """12 form cells (3 forms x 4 rungs) + the rung-8 cells, for a mid-run member."""
    profile = build_eval_profile("blokus_duo")
    member_version = 10
    k_total = 30
    available_versions = tuple(range(1, 11))  # 1..10 published so far

    cells = required_cell_ids(member_version, profile, available_versions, k_total, forms=(5, 6, 7))

    opponent_names = ("random", "largest-piece", "mobility", "uct-rollout-v1")
    form_cells = {
        build_cell_id(member_version, form, name) for form in (5, 6, 7) for name in opponent_names
    }
    assert len(form_cells) == 12

    # lag = ceil(30 / 4) = 8; wanted = {9, 2, 1} intersected with [1, 9] and
    # the available 1..10 -- all three survive.
    rung8_cells = {
        build_cell_id(member_version, RUNG8_CANDIDATE_FORM, agent_identity(7, u)) for u in (1, 2, 9)
    }
    assert len(rung8_cells) == 3

    expected = form_cells | rung8_cells
    assert set(cells) == expected
    assert len(cells) == 15
    assert cells == sorted(cells)  # ascending, deduplicated


def test_required_cell_ids_no_rung8_cells_when_form_7_not_requested():
    profile = build_eval_profile("blokus_duo")
    cells = required_cell_ids(10, profile, tuple(range(1, 11)), k_total=30, forms=(5, 6))
    assert len(cells) == 8  # 2 forms x 4 rungs, no rung-8 category at all
    assert all(".7." not in c for c in cells)


def test_required_cell_ids_first_candidate_has_no_rung8_opponents():
    profile = build_eval_profile("blokus_duo")
    cells = required_cell_ids(1, profile, (1,), k_total=30, forms=(5, 6, 7))
    assert len(cells) == 12  # candidate 1 has no earlier version to face


def test_required_cell_ids_rejects_non_positive_member_version():
    profile = build_eval_profile("blokus_duo")
    with pytest.raises(ValueError, match="member_version"):
        required_cell_ids(0, profile, (1,), k_total=30, forms=(5,))


def test_required_cell_ids_rejects_empty_forms():
    profile = build_eval_profile("blokus_duo")
    with pytest.raises(ValueError, match="forms"):
        required_cell_ids(1, profile, (1,), k_total=30, forms=())


# --- second-game fixture (S2.1): zero orchestrator/core edits ------------------


def test_stub_profile_schedules_its_reduced_ladder_with_zero_core_edits():
    """A from-scratch, two-rung profile for a hypothetical second game.

    Neither ``core/eval_run.py``, ``core/eval_profile.py``, nor
    ``games/registry.py`` needed a single edit to support this: the whole
    fixture is built right here, and :func:`required_cell_ids` schedules it
    exactly like Blokus's real, four-rung profile above.
    """
    stub_profile = EvalProfile(
        network_free_rungs={
            1: _StubAgent("stub-random"),
            2: _StubAgent("stub-heuristic"),
        },
        opening_balancer=None,
    )
    assert stub_profile.rungs() == (1, 2)
    assert stub_profile.rung_identity(1) == "stub-random"
    assert stub_profile.rung_identity(2) == "stub-heuristic"

    # An even more reduced ladder (rung 1 only) and a single form (5): one cell.
    reduced_profile = EvalProfile(network_free_rungs={1: _StubAgent("stub-random")})
    cells = required_cell_ids(2, reduced_profile, (1, 2), k_total=2, forms=(5,))
    assert cells == [build_cell_id(2, 5, "stub-random")]

    # Both rungs, both a network-free-only form and the rung-8-eligible one.
    cells_full = required_cell_ids(2, stub_profile, (1, 2), k_total=2, forms=(5, 7))
    # lag = ceil(2 / 4) = 1; wanted = {1, 1, 1} intersected with [1, 1] -> {1}.
    expected = {
        build_cell_id(2, 5, "stub-random"),
        build_cell_id(2, 5, "stub-heuristic"),
        build_cell_id(2, 7, "stub-random"),
        build_cell_id(2, 7, "stub-heuristic"),
        build_cell_id(2, 7, agent_identity(7, 1)),
    }
    assert set(cells_full) == expected


def test_stub_profile_rung_identity_unknown_rung_raises():
    stub_profile = EvalProfile(network_free_rungs={1: _StubAgent("only-one")})
    with pytest.raises(ValueError, match="declares no network-free rung"):
        stub_profile.rung_identity(2)


def test_eval_profile_rejects_empty_rungs():
    with pytest.raises(ValueError, match="non-empty"):
        EvalProfile(network_free_rungs={})


# --- per-cell seed ----------------------------------------------------------------


def test_cell_seed_matches_derive_seed_directly():
    assert cell_seed(123, "5.7.random") == derive_seed(123, PURPOSE_EVAL, "5.7.random")


def test_cell_seed_matches_the_eval_protocol_seed_label():
    """PURPOSE_EVAL's literal value is pinned equal to the protocol registry's
    own recorded label -- the two sides of the seed-derivation shape agree by
    construction, not by two people remembering the same string.
    """
    assert PURPOSE_EVAL == eval_protocol.SEED_LABEL_EVAL


def test_cell_seed_is_a_pure_function_of_eval_seed_and_cell_id():
    a = cell_seed(1, "5.7.random")
    b = cell_seed(1, "5.7.random")
    c = cell_seed(2, "5.7.random")
    d = cell_seed(1, "5.7.mobility")
    assert a == b
    assert a != c
    assert a != d
