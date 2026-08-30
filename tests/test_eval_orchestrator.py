"""The M4 eval orchestrator's config/correctness spine (design doc §9; tasks/m4/009.1).

Covers: :class:`~core.eval_run.EvalConfig`'s JSON-parse discipline, the
launch-provenance write + the relaunch guard's "no override, refuse and name
every differing field" behavior (both the config-field and the
protocol-registry-fingerprint drift paths), the membership arithmetic that
keeps ``ckpt-0.pt``/the rolling ``resume.pt`` snapshot/the ``latest`` pointer
structurally unschedulable, the cell-scheduling arithmetic against a real
game's declared eval profile, and a from-scratch stub profile proving the
mechanism never needs a ``core/`` or Blokus-specific edit to support a second
game (S2.1) -- subtask 9.1.

Subtask 9.2 (below the 9.1 section marker) covers the watch/resume/catch-up
loop itself (:func:`~core.eval_run.run_watch_loop`) against a *real*,
tiny-trunk micro-Blokus checkpoint chain: idempotent scheduling and
completion end-to-end, downtime catch-up (a member published while the loop
was not running, and one published *between* two live polls), the
``max_idle_polls``/``single_pass`` stop conditions, kill-mid-cell ->
relaunch byte-identical-cell-file equivalence, two-independent-runs
determinism, the eval-lag observable against a hand-built manifest, and
protocol asserts (no root noise, pinned sims) on the agents the loop's own
checkpoint-load cache constructs.

Subtask 9.3 (below the 9.2 section marker) covers the CLI face
(``scripts/run_eval.py``) and the milestone's own acceptance criterion: a
tiny, *real* ``scripts/run_selfplay.py`` run (genuine multiprocessing, the m3
task-12 pattern) with the eval harness running concurrently in a background
thread, driven through the exact same ``run_watch_loop``/``bench_candidate``
plumbing the script calls -- every member ending complete in the manifest,
cell files round-tripping, the §1 report artifacts appearing with `delta:
null` before the K-set completes and Delta/CI/gate present (but
`authoritative: false` at a reduced test `B`, `true` only at the pinned
production `B`) after -- plus fast, non-``slow`` unit coverage of the report
hook, the bench measurement, and the ``--report``/``--plateau``/``--bench``
CLI wiring against hand-built (no-real-play) stores, mirroring
``tests/test_eval_stats.py``'s own fixture conventions.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest
import torch

import core.eval_agents as eval_agents_module
import core.eval_protocol as eval_protocol
import core.eval_run as eval_run
from core.agents import MobilityAgent, RandomAgent
from core.checkpoint import (
    LATEST_FILENAME,
    RESUME_FILENAME,
    build_bundle,
    write_published_checkpoint,
)
from core.eval_profile import EvalProfile
from core.eval_run import (
    RUNG8_CANDIDATE_FORM,
    BenchResult,
    EvalConfig,
    EvalRelaunchRefusedError,
    WatchLoopResult,
    _CandidateCache,
    _resolve_opponent_factory,
    agent_identity,
    bench_candidate,
    cell_seed,
    checkpoint_dir,
    eval_lag,
    load_eval_config,
    required_cell_ids,
    resolve_eval_launch,
    run_watch_loop,
    schedulable_versions,
)
from core.eval_stats import (
    PLATEAU_OUTCOME_INSUFFICIENT_DATA,
    PlateauResult,
    build_verdict,
    elo_curve_path,
    verdict_path,
)
from core.eval_store import (
    CellId,
    GameRecordSnapshot,
    PairRecord,
    append_pair_record,
    build_cell_id,
    build_header,
    cell_path,
    cells_dir,
    complete_cell,
    load_snapshot,
    open_cell_for_write,
    read_cell,
    register_member,
)
from core.mcts import MCTS
from core.metrics import EpochMetricsWriter
from core.network import Network, NetworkConfig
from core.observability import CHECKPOINT_PUBLISHED_KIND
from core.run_identity import (
    ENTRY_CONDITION,
    LAUNCH_SCHEMA_VERSION,
    LaunchConfig,
    RunRecord,
    write_provenance,
)
from core.runconfig import MICRO_RUN_CONFIG_PATH
from core.seeding import PURPOSE_EVAL, derive_seed
from core.train import make_optimizer, make_scaler
from games.blokus_duo import BlokusDuo
from games.blokus_duo.baselines import LargestPieceAgent, start_square_balancer
from games.blokus_duo.config import MICRO_CONFIG
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


# ==============================================================================
# 9.2: the watch/resume/catch-up loop (tasks/m4/009.2)
# ==============================================================================
#
# Real, tiny-trunk micro-Blokus checkpoints throughout, never a hand-built
# manifest standing in for actual play -- except the eval-lag golden, which is
# deliberately manifest-only: lag is defined purely in terms of schedulable
# versions and the snapshot's member prefix, neither of which needs a game
# played to test.
#
# The network-free ladder used here is deliberately *not*
# ``games.registry.build_eval_profile("blokus_duo")`` (which includes rung 4,
# ``core.uct.UCTAgent``, at its frozen 1,000-simulation-per-move budget) -- a
# fast, from-scratch 3-rung profile over the same real micro-Blokus adapter
# and baselines keeps every test here genuinely seconds-fast while still
# exercising real play through ``core.runner.play_pairs``, the real task-5
# store, and a real loaded checkpoint per candidate version.

GAME = BlokusDuo(config=MICRO_CONFIG)
FAST_PROFILE = EvalProfile(
    network_free_rungs={1: RandomAgent, 2: LargestPieceAgent, 3: MobilityAgent},
    opening_balancer=start_square_balancer,
)


def _tiny_network_config(game) -> NetworkConfig:
    """A tiny, fast-to-build ``NetworkConfig`` matching ``game``'s surface.

    Mirrors ``tests/test_eval_agents.py``'s own ``_tiny_network_config`` --
    kept local per this file's existing no-cross-test-import convention (see
    this file's helpers above, and ``tests/test_eval_stats.py``'s docstring
    for the same rule stated explicitly).
    """
    return NetworkConfig(
        input_planes=game.input_planes,
        input_shape=tuple(game.input_shape),
        policy_shape=tuple(game.policy_shape),
        trunk_blocks=1,
        trunk_channels=4,
        num_aux=len(game.value_targets.aux_names),
    )


def _write_checkpoint(ckpt_dir, game, *, version: int, seed: int):
    """Build and publish one tiny, seeded, real checkpoint for ``game``."""
    torch.manual_seed(seed)
    net = Network(_tiny_network_config(game))
    optimizer = make_optimizer(net, lr=1e-2)
    scaler = make_scaler("cpu")
    bundle = build_bundle(
        version=version,
        learner_step=0,
        game=game,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    return write_published_checkpoint(ckpt_dir, bundle)


def _write_run_provenance(
    run_dir, *, checkpoint_count: int, eval_seed: int, run_id: str = "watched-run"
):
    """Write a real, valid watched-run ``config.json``/``run_record.json`` pair.

    Starts from the real pinned micro-Blokus run config and overrides only the
    scalars the watch loop actually reads through it (``training.
    checkpoint_count`` via ``watched_k_total``, and the run's identity via
    ``core.run_identity.read_run_record``) -- mirrors
    ``tests/test_eval_stats.py``'s ``_write_run_config`` helper (kept local,
    same convention: never imported across test files here).
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


def _build_watched_run(run_dir, *, k_total: int, versions=None, eval_seed: int = 999):
    """Write real provenance plus one real tiny checkpoint per version.

    Args:
        run_dir: Root directory to build.
        k_total: The watched run's ``training.checkpoint_count``.
        versions: Which member versions to publish now (default: every one of
            ``1..k_total``) -- a caller passes a strict subset to model a
            checkpoint not yet (or no longer) published.
        eval_seed: The watched run's own recorded ``evaluation.eval_seed``
            (unrelated to, and never used as, the harness's own
            ``EvalConfig.eval_seed`` -- see :func:`_eval_config`).
    """
    _write_run_provenance(run_dir, checkpoint_count=k_total, eval_seed=eval_seed)
    ckpt_dir = checkpoint_dir(run_dir)
    for v in versions if versions is not None else range(1, k_total + 1):
        _write_checkpoint(ckpt_dir, GAME, version=v, seed=v)


def _eval_config(run_dir, **overrides) -> EvalConfig:
    """A tiny, real-play-ready :class:`EvalConfig` (2 pairs/cell, 4 sims, forms 5/6/7)."""
    fields = {"pairs_per_cell": 2, "eval_sims": 4, "forms": (5, 6, 7)}
    fields.update(overrides)
    return _make_config(str(run_dir), **fields)


def _cell_bytes(run_dir) -> dict[str, bytes]:
    """Every cell file's raw bytes under ``run_dir``, by filename."""
    return {p.name: p.read_bytes() for p in cells_dir(run_dir).iterdir()}


# --- end-to-end scheduling + play -----------------------------------------------


def test_watch_loop_schedules_and_plays_every_required_cell_in_one_pass(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=2)
    config = _eval_config(run_dir)

    result = run_watch_loop(run_dir, config, FAST_PROFILE, GAME, single_pass=True)

    assert isinstance(result, WatchLoopResult)
    assert result.completed_members == (1, 2)
    assert result.stopped_reason == "single_pass"
    assert result.polls == 1

    # 3 forms x 3 network-free rungs = 9 cells/member, plus member 2's one
    # rung-8 historical opponent (lag = ceil(2/4) = 1; wanted = {1,1,1} -> {1}).
    expected_cells = {
        build_cell_id(v, form, opp)
        for v in (1, 2)
        for form in (5, 6, 7)
        for opp in ("random", "largest-piece", "mobility")
    }
    expected_cells.add(
        build_cell_id(2, RUNG8_CANDIDATE_FORM, agent_identity(RUNG8_CANDIDATE_FORM, 1))
    )

    snapshot = load_snapshot(run_dir)
    assert snapshot.completed_cell_ids == frozenset(expected_cells)
    assert snapshot.member_prefix == 2
    assert result.pairs_played == len(expected_cells) * config.pairs_per_cell


def test_watch_loop_second_pass_is_idempotent_and_plays_nothing_new(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=1)
    config = _eval_config(run_dir)

    result1 = run_watch_loop(run_dir, config, FAST_PROFILE, GAME, single_pass=True)
    assert result1.pairs_played > 0
    before = _cell_bytes(run_dir)

    result2 = run_watch_loop(run_dir, config, FAST_PROFILE, GAME, single_pass=True)

    assert result2.pairs_played == 0
    assert result2.completed_members == (1,)
    assert _cell_bytes(run_dir) == before


# --- downtime catch-up + continuous-polling pickup -------------------------------


def test_watch_loop_downtime_catchup_schedules_a_member_published_while_stopped(tmp_path):
    """A member published while the harness process was not running at all --
    two entirely separate calls, nothing in between -- is scheduled the next
    time it is launched (design doc §9's coupling requirement)."""
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=2, versions=[1])
    config = _eval_config(run_dir)

    result1 = run_watch_loop(run_dir, config, FAST_PROFILE, GAME, single_pass=True)
    assert result1.completed_members == (1,)

    _write_checkpoint(checkpoint_dir(run_dir), GAME, version=2, seed=2)

    result2 = run_watch_loop(run_dir, config, FAST_PROFILE, GAME, single_pass=True)
    assert result2.completed_members == (1, 2)
    assert result2.pairs_played > 0


def test_watch_loop_continuous_polling_picks_up_a_checkpoint_published_between_polls(tmp_path):
    """A member published *while the loop keeps running* -- discovered between
    two of its own polls, no relaunch involved at all."""
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=2, versions=[1])
    config = _eval_config(run_dir)
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) == 2:
            _write_checkpoint(checkpoint_dir(run_dir), GAME, version=2, seed=2)

    result = run_watch_loop(
        run_dir,
        config,
        FAST_PROFILE,
        GAME,
        poll_interval=0.0,
        max_idle_polls=5,
        sleep=fake_sleep,
    )

    # poll 1 plays v1 (not idle); poll 2 finds nothing new (idle, and the
    # injected sleep publishes v2); poll 3 plays v2 and both members are
    # complete.
    assert result.stopped_reason == "k_complete"
    assert result.completed_members == (1, 2)
    assert result.polls == 3
    assert sleep_calls == [0.0, 0.0]


def test_watch_loop_stops_after_max_idle_polls_with_no_further_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=2, versions=[1])  # v2 never appears.
    config = _eval_config(run_dir)
    sleep_calls: list[float] = []

    result = run_watch_loop(
        run_dir,
        config,
        FAST_PROFILE,
        GAME,
        poll_interval=0.0,
        max_idle_polls=2,
        sleep=sleep_calls.append,
    )

    assert result.stopped_reason == "idle"
    assert result.completed_members == (1,)
    # poll 1 plays v1 (idle resets to 0); polls 2 and 3 find nothing new
    # (idle -> 1, then 2 -> stop).
    assert result.polls == 3
    assert len(sleep_calls) == 2


# --- kill mid-cell -> relaunch equivalence, and determinism ----------------------


def test_watch_loop_kill_mid_cell_then_relaunch_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    baseline_dir = tmp_path / "baseline"
    _build_watched_run(baseline_dir, k_total=1)
    run_watch_loop(baseline_dir, _eval_config(baseline_dir), FAST_PROFILE, GAME, single_pass=True)

    killed_dir = tmp_path / "killed"
    _build_watched_run(killed_dir, k_total=1)
    killed_config = _eval_config(killed_dir)

    real_append = eval_run.append_pair_record
    calls = {"n": 0}

    def flaky_append(path, record):
        calls["n"] += 1
        if calls["n"] > 3:  # cell 1 (2 pairs) fully written; cell 2's 2nd pair dies.
            raise RuntimeError("simulated kill mid-cell")
        return real_append(path, record)

    monkeypatch.setattr(eval_run, "append_pair_record", flaky_append)
    with pytest.raises(RuntimeError, match="simulated kill mid-cell"):
        run_watch_loop(killed_dir, killed_config, FAST_PROFILE, GAME, single_pass=True)
    monkeypatch.undo()  # restore the real writer before the relaunch below.

    relaunch_result = run_watch_loop(
        killed_dir, killed_config, FAST_PROFILE, GAME, single_pass=True
    )

    assert relaunch_result.completed_members == (1,)
    # The manifest's own wall-clock fields (scheduled_at/completed_at) are
    # outside this equality claim by construction: only cell files are
    # compared, and core.eval_store never writes wall-clock into a cell file.
    assert _cell_bytes(killed_dir) == _cell_bytes(baseline_dir)


def test_watch_loop_determinism_two_independent_runs_produce_byte_identical_stores(tmp_path):
    run_dir_a = tmp_path / "a"
    run_dir_b = tmp_path / "b"
    _build_watched_run(run_dir_a, k_total=1)
    _build_watched_run(run_dir_b, k_total=1)

    run_watch_loop(run_dir_a, _eval_config(run_dir_a), FAST_PROFILE, GAME, single_pass=True)
    run_watch_loop(run_dir_b, _eval_config(run_dir_b), FAST_PROFILE, GAME, single_pass=True)

    assert _cell_bytes(run_dir_a) == _cell_bytes(run_dir_b)


# --- eval lag ---------------------------------------------------------------------


def _lag_header(*, candidate_version: int, rung: int, opponent_id: str, n_pairs: int):
    return build_header(
        run_id="run",
        cell_id=CellId(candidate_version, rung, opponent_id),
        candidate_identity=agent_identity(rung, candidate_version),
        opponent_identity=opponent_id,
        eval_config={"pairs_per_cell": n_pairs},
        candidate_fingerprint={"orientation_table_hash": "test"},
    )


def _lag_fill_and_complete(run_dir, header, scores):
    path = cell_path(run_dir, header.cell_id.to_string())
    next_index = open_cell_for_write(run_dir, header)
    for i, score in enumerate(scores, start=next_index):
        append_pair_record(
            path,
            PairRecord(
                pair_index=i,
                pair_seed=i,
                score_a=score,
                games=(
                    GameRecordSnapshot(plies=1, opening=0),
                    GameRecordSnapshot(plies=1, opening=0),
                ),
            ),
        )
    complete_cell(run_dir, header.cell_id.to_string())


def test_eval_lag_matches_a_hand_built_manifest_state(tmp_path):
    run_dir = tmp_path / "run"
    ckpt_dir = checkpoint_dir(run_dir)
    ckpt_dir.mkdir(parents=True)
    for v in (1, 2, 3):
        (ckpt_dir / f"ckpt-{v}.pt").write_bytes(b"")

    # Member 1: registered and fully complete.
    header1 = _lag_header(candidate_version=1, rung=5, opponent_id="random", n_pairs=2)
    register_member(run_dir, 1, [header1.cell_id.to_string()])
    _lag_fill_and_complete(run_dir, header1, [1.0, 1.0])

    # Member 2: registered but left incomplete -- breaks the contiguous prefix.
    header2 = _lag_header(candidate_version=2, rung=5, opponent_id="random", n_pairs=2)
    register_member(run_dir, 2, [header2.cell_id.to_string()])

    # Member 3: a published checkpoint file only -- never even registered.

    assert eval_lag(run_dir, k_total=3) == 2  # newest published (3) - member_prefix (1)


def test_eval_lag_is_zero_with_no_checkpoints_published(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir(run_dir).mkdir(parents=True)
    assert eval_lag(run_dir, k_total=5) == 0


def test_eval_lag_is_zero_once_fully_caught_up(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=1)
    run_watch_loop(run_dir, _eval_config(run_dir), FAST_PROFILE, GAME, single_pass=True)
    assert eval_lag(run_dir, k_total=1) == 0


# --- the checkpoint-load cache: protocol asserts + the memory stance -------------


class _RecordingMCTS(MCTS):
    """A real, unmodified ``MCTS`` that records every construction's kwargs.

    Mirrors ``tests/test_eval_agents.py``'s own ``_RecordingMCTS`` shim --
    kept local (this file's existing convention): monkeypatch
    ``core.eval_agents.MCTS`` to this class so a :class:`~core.eval_agents.SearchAgent`
    built through the loop's own cache can be white-box inspected without
    duplicating any of its construction logic.
    """

    kwargs_log: list[dict] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _RecordingMCTS.kwargs_log.append(kwargs)


def test_candidate_cache_builds_search_agents_with_no_root_noise_and_pinned_sims(
    tmp_path, monkeypatch
):
    ckpt_dir = checkpoint_dir(tmp_path)
    _write_checkpoint(ckpt_dir, GAME, version=1, seed=1)

    _RecordingMCTS.kwargs_log = []
    monkeypatch.setattr(eval_agents_module, "MCTS", _RecordingMCTS)

    cache = _CandidateCache(ckpt_dir=ckpt_dir, game=GAME, device="cpu", eval_sims=4)
    state = GAME.initial_state()
    for form in (6, 7):
        agent = cache.candidate_factory(1, form)(seed=0)
        assert agent._sims == 4  # the harness's configured S, not EVAL_SIMS's 512 default.
        first = agent.select_action(GAME, state)
        second = agent.select_action(GAME, state)
        assert first == second  # deterministic: no root noise, no rng consumed.

    assert len(_RecordingMCTS.kwargs_log) == 4  # 2 forms x 2 select_action calls each.
    assert all(kwargs["root_noise"] is None for kwargs in _RecordingMCTS.kwargs_log)
    assert [kwargs["uniform_prior"] for kwargs in _RecordingMCTS.kwargs_log] == [
        True,
        True,
        False,
        False,
    ]


def test_candidate_cache_loads_the_checkpoint_exactly_once_per_version(tmp_path, monkeypatch):
    ckpt_dir = checkpoint_dir(tmp_path)
    _write_checkpoint(ckpt_dir, GAME, version=1, seed=1)

    real_load = eval_run.load_eval_network
    calls = {"n": 0}

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(eval_run, "load_eval_network", counting_load)

    cache = _CandidateCache(ckpt_dir=ckpt_dir, game=GAME, device="cpu", eval_sims=4)
    cache.candidate_factory(1, 5)
    cache.candidate_factory(1, 6)
    cache.candidate_factory(1, 7)

    assert calls["n"] == 1  # loaded once for the version, shared across all 3 forms.


def test_candidate_cache_rejects_an_unrecognized_form(tmp_path):
    ckpt_dir = checkpoint_dir(tmp_path)
    _write_checkpoint(ckpt_dir, GAME, version=1, seed=1)
    cache = _CandidateCache(ckpt_dir=ckpt_dir, game=GAME, device="cpu", eval_sims=4)
    with pytest.raises(ValueError, match="form must be 5, 6, or 7"):
        cache.candidate_factory(1, 8)


def test_historical_cache_loads_the_checkpoint_exactly_once_across_requests(tmp_path, monkeypatch):
    ckpt_dir = checkpoint_dir(tmp_path)
    _write_checkpoint(ckpt_dir, GAME, version=1, seed=1)

    real_factory = eval_run.historical_opponent_factory
    calls = {"n": 0}

    def counting_factory(*args, **kwargs):
        calls["n"] += 1
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(eval_run, "historical_opponent_factory", counting_factory)

    cache = _CandidateCache(ckpt_dir=ckpt_dir, game=GAME, device="cpu", eval_sims=4)
    cache.historical_factory(1)
    cache.historical_factory(1)

    assert calls["n"] == 1


def test_resolve_opponent_factory_resolves_a_declared_rung_identity(tmp_path):
    cache = _CandidateCache(ckpt_dir=checkpoint_dir(tmp_path), game=GAME, device="cpu", eval_sims=4)
    factory = _resolve_opponent_factory("random", {"random": RandomAgent}, cache)
    assert factory is RandomAgent


def test_resolve_opponent_factory_resolves_a_rung8_historical_identity(tmp_path):
    ckpt_dir = checkpoint_dir(tmp_path)
    _write_checkpoint(ckpt_dir, GAME, version=1, seed=1)
    cache = _CandidateCache(ckpt_dir=ckpt_dir, game=GAME, device="cpu", eval_sims=4)

    factory = _resolve_opponent_factory(agent_identity(RUNG8_CANDIDATE_FORM, 1), {}, cache)

    assert factory(seed=0).name == "rung7-v1-1"


def test_resolve_opponent_factory_rejects_an_unrecognized_identity(tmp_path):
    cache = _CandidateCache(ckpt_dir=checkpoint_dir(tmp_path), game=GAME, device="cpu", eval_sims=4)
    with pytest.raises(ValueError, match="neither"):
        _resolve_opponent_factory("not-a-real-identity", {"random": RandomAgent}, cache)


def test_watch_loop_never_touches_config_json_relaunch_guard_between_polls(tmp_path):
    """The resolved config is checked once at launch, not re-checked every poll
    (:func:`resolve_eval_launch` runs before the poll loop starts, never
    inside it) -- a config mutated on disk mid-run has no effect on an
    already-running call."""
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=1)
    config = _eval_config(run_dir)
    run_watch_loop(run_dir, config, FAST_PROFILE, GAME, single_pass=True)

    stored_before = eval_run.read_eval_provenance(run_dir)
    # A second, field-for-field identical call must not touch the recorded
    # provenance file at all (never rewritten on an unchanged relaunch).
    before_bytes = eval_run.eval_config_path(run_dir).read_bytes()
    run_watch_loop(run_dir, config, FAST_PROFILE, GAME, single_pass=True)
    after_bytes = eval_run.eval_config_path(run_dir).read_bytes()

    assert before_bytes == after_bytes
    assert stored_before.config == config


def test_watch_loop_refuses_a_relaunch_with_a_changed_material_field(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=1)
    run_watch_loop(run_dir, _eval_config(run_dir), FAST_PROFILE, GAME, single_pass=True)

    changed_config = _eval_config(run_dir, eval_sims=64)
    with pytest.raises(EvalRelaunchRefusedError, match="eval_sims"):
        run_watch_loop(run_dir, changed_config, FAST_PROFILE, GAME, single_pass=True)


# ==============================================================================
# 9.3: the CLI face (scripts/run_eval.py) and the concurrent acceptance run
# (tasks/m4/009.3)
# ==============================================================================
#
# Two kinds of coverage, deliberately split for speed and determinism:
#
# * The **acceptance run** below (``slow``) is the milestone's own acceptance
#   criterion: a tiny, *real* ``scripts/run_selfplay.py`` run (genuine
#   multiprocessing, the m3 task-12 pattern) with the eval harness running
#   concurrently in a background thread -- membership completeness, the
#   task-5 cell round-trip, and the provisional-then-authoritative report
#   distinction, all against evidence the concurrency itself produced.
# * The **report-hook / bench / plateau / partial-cell-golden** tests further
#   below are fast and fully deterministic: they exercise the exact same
#   ``core.eval_run``/``core.eval_stats`` machinery the acceptance run and
#   ``scripts/run_eval.py`` both stand on, but over hand-built (no-real-play)
#   stores -- mirroring ``tests/test_eval_stats.py``'s own fixture
#   conventions -- rather than by trying to win a race against a second live
#   process. Nothing about the *correctness* of the report/plateau/bench
#   layer depends on a live concurrent run; only the milestone's own
#   end-to-end coupling claim does, and that is exactly what the acceptance
#   run alone is for.

ROOT = Path(__file__).resolve().parent.parent


def _load_script_module(name: str, path: Path):
    """Import a ``scripts/`` module by file path (``scripts/`` is not a package).

    Mirrors ``tests/test_run_entrypoint.py``'s own ``_load_run_selfplay``
    helper -- kept local per this file's existing no-cross-test-import
    convention.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rs = _load_script_module("run_selfplay", ROOT / "scripts" / "run_selfplay.py")
run_eval = _load_script_module("run_eval", ROOT / "scripts" / "run_eval.py")


def _selfplay_raw(tmp_path, *, checkpoint_count: int, publish_interval: int) -> dict:
    """The pinned micro-Blokus run config, tiny-fied for a bounded real run.

    Mirrors ``tests/test_run_entrypoint.py``'s own ``_base_raw`` exactly
    (``replay_warmup_positions`` kept high so the D5 replay-ratio
    ceiling/floor stays dormant, per that helper's own docstring) -- kept
    local rather than imported across test files.
    """
    raw = json.loads(MICRO_RUN_CONFIG_PATH.read_text())
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    raw["self_play"] = {**raw["self_play"], "sims": 128}
    training = dict(raw["training"])
    training.update(
        publish_interval=publish_interval,
        checkpoint_count=checkpoint_count,
        replay_warmup_positions=10_000,
        batch_size=4,
        replay_window=2000,
        learning_rate=1e-3,
        warmup_steps=0,
        cosine_total_steps=100,
    )
    raw["training"] = training
    raw["run_dir"] = str(tmp_path / "selfplay_runs")
    raw["num_actors"] = 1
    raw["device"] = "cpu"
    raw["schema_version"] = LAUNCH_SCHEMA_VERSION
    raw["runtime"] = {
        "refresh_poll_interval": 0.02,
        "pacing_poll_interval": 0.02,
        "ceiling_poll_interval": 0.02,
    }
    return raw


def _only_run_dir(runs_root):
    dirs = list(runs_root.iterdir())
    assert len(dirs) == 1, dirs
    return dirs[0]


@pytest.mark.slow
def test_acceptance_concurrent_run_scores_every_member_end_to_end(tmp_path):
    """The milestone's own acceptance criterion (tasks/m4/009 Description):
    "a tiny end-to-end run whose §1 report computes with zero manual steps."

    A real, tiny M3 self-play run (K=2 publishes) launched with
    ``scripts/run_selfplay.py`` (genuine multiprocessing), with the eval
    harness running concurrently in a background thread over the exact
    ``core.eval_run.run_watch_loop`` plumbing ``scripts/run_eval.py`` itself
    calls. The harness's own ``on_member_complete`` hook captures a §1 report
    at the instant each member transitions to complete -- race-free by
    construction: the hook fires synchronously, before this same poll's
    ``for version in available`` loop ever reaches a later version, so the
    report captured when member 1 completes reflects *exactly* member 1's
    evidence regardless of how far training has meanwhile progressed on disk.
    """
    raw = _selfplay_raw(tmp_path, checkpoint_count=2, publish_interval=2)
    selfplay_cfg_path = tmp_path / "selfplay_cfg.json"
    selfplay_cfg_path.write_text(json.dumps(raw))
    net_config = _tiny_network_config(GAME)

    launched = rs.cmd_launch(
        selfplay_cfg_path,
        max_games_per_actor=12,
        block=False,
        now=1_700_000_000.0,
        network_config=net_config,
    )
    run_dir = _only_run_dir(tmp_path / "selfplay_runs")

    eval_config = _eval_config(
        run_dir, pairs_per_cell=2, eval_sims=4, forms=(5, 6, 7), bootstrap_b=39
    )
    eval_cfg_path = tmp_path / "eval_cfg.json"
    eval_cfg_path.write_text(json.dumps(eval_config.to_dict()))

    captured_reports: dict[int, dict] = {}
    errors: list[BaseException] = []

    def _capture_hook(version: int) -> None:
        captured_reports[version] = build_verdict(str(run_dir), B=eval_config.bootstrap_b)

    def _run_harness() -> None:
        try:
            game, profile = run_eval._resolve_watched_game(str(run_dir))
            run_watch_loop(
                run_dir,
                eval_config,
                profile,
                game,
                poll_interval=0.05,
                max_idle_polls=3000,  # a generous (~150s) safety net, never expected to fire
                on_member_complete=_capture_hook,
            )
        except BaseException as exc:  # noqa: BLE001 -- surfaced on the main thread below.
            errors.append(exc)

    harness_thread = threading.Thread(target=_run_harness, daemon=True)
    harness_thread.start()
    try:
        rs.wait_for_completion(launched)
        harness_thread.join(timeout=200)
    finally:
        if harness_thread.is_alive():
            launched.shutdown()  # best-effort: never leave a live process behind on failure.

    assert not harness_thread.is_alive(), "eval harness did not finish within the test timeout"
    if errors:
        raise errors[0]

    k_total = 2
    assert set(captured_reports) == {1, 2}

    # -- membership: every member ends complete in the manifest ---------------
    snapshot = load_snapshot(run_dir)
    assert snapshot.member_prefix == k_total

    # -- cell files pass the task-5 round-trip ---------------------------------
    assert snapshot.completed_cell_ids  # non-trivial: real cells were really played
    for cell_id in sorted(snapshot.completed_cell_ids):
        header, records = read_cell(cell_path(run_dir, cell_id))
        assert header.cell_id.to_string() == cell_id
        assert len(records) == eval_config.pairs_per_cell

    # -- provisional (delta: null, no gate) before the K-set completes --------
    provisional = captured_reports[1]
    assert provisional["checkpoints_evaluated"] == 1
    assert provisional["k_target"] == k_total
    assert provisional["delta"] is None
    assert provisional["authoritative"] is False
    assert "gate" not in json.dumps(provisional["delta"])  # delta itself is just `null`

    # -- Delta/CI/gate present once the K-set completes; authoritative iff B ==
    #    the pinned production B (task 7 pin 8; the distinction this stage
    #    exists to prove) -----------------------------------------------------
    final_test_b = captured_reports[k_total]
    assert final_test_b["checkpoints_evaluated"] == k_total == final_test_b["k_target"]
    assert final_test_b["bootstrap_b"] == 39
    assert final_test_b["delta"] is not None
    assert set(final_test_b["delta"]) == {"delta_hat", "ci", "gate"}
    assert final_test_b["authoritative"] is False  # B=39, not the pinned production B

    final_production_b = build_verdict(run_dir, B=eval_protocol.BOOTSTRAP_B_PRODUCTION)
    assert final_production_b["checkpoints_evaluated"] == k_total
    assert final_production_b["bootstrap_b"] == eval_protocol.BOOTSTRAP_B_PRODUCTION
    assert final_production_b["delta"] is not None
    assert final_production_b["authoritative"] is True

    assert eval_lag(run_dir, k_total) == 0

    # -- the script's own CLI wiring, smoke-level, over the same real run -----
    report_payload = run_eval.cmd_report(eval_cfg_path)
    assert report_payload["eval_lag"] == 0
    assert report_payload["verdict"]["checkpoints_evaluated"] == k_total
    assert report_payload["verdict"]["authoritative"] is False
    rendered_report = run_eval._render_report(report_payload)
    assert "authoritative: False" in rendered_report

    plateau_result = run_eval.cmd_plateau(eval_cfg_path)
    assert plateau_result.outcome == PLATEAU_OUTCOME_INSUFFICIENT_DATA  # K=2 << window M=8
    rendered_plateau = run_eval._render_plateau(plateau_result)
    assert rendered_plateau.splitlines()[0] == f"outcome: {PLATEAU_OUTCOME_INSUFFICIENT_DATA}"

    bench_result = run_eval.cmd_bench(eval_cfg_path, n_pairs=1)
    assert bench_result.candidate_version == k_total  # newest published member
    assert bench_result.games_played > 0
    assert bench_result.seconds_per_game > 0.0


# --- fast, hand-built-store coverage (mirrors tests/test_eval_stats.py's own fixtures) --


def _write_checkpoint_markers(run_dir, versions):
    """Write one ``checkpoint_published`` marker per member version.

    Mirrors ``tests/test_eval_stats.py``'s own local ``_write_checkpoint_markers``
    helper -- kept local per this file's existing no-cross-test-import
    convention. The minimal metrics fixture ``elo_curve``/``build_verdict``
    need: ``core.observability.reduce_run`` needs no GPU segments or actor
    deltas to compute ``net_evals``/``gpu_hours`` (both default to ``0.0``
    with none on disk) -- only a marker per scored member version.
    """
    learner = EpochMetricsWriter(run_dir, "learner")
    for i, version in enumerate(sorted(versions), start=1):
        learner.append(
            {
                "kind": CHECKPOINT_PUBLISHED_KIND,
                "model_version": version,
                "learner_step": 10 * i,
                "timestamp": float(i),
            }
        )


def _write_scored_member(run_dir, member_version, scores, *, opponent_id="random"):
    """Register and complete one member's single rung-7-vs-anchor cell.

    The minimal fixture ``build_verdict``/``detect_plateau`` need to fit: one
    cell connecting the candidate to ``core.eval_stats.ANCHOR_AGENT``
    (``"random"``).
    """
    header = _lag_header(
        candidate_version=member_version, rung=7, opponent_id=opponent_id, n_pairs=len(scores)
    )
    register_member(run_dir, member_version, [header.cell_id.to_string()])
    _lag_fill_and_complete(run_dir, header, scores)


def _hand_built_store(run_dir, *, checkpoint_count: int, scored_versions, eval_seed: int = 999):
    """A run directory with real provenance + a hand-built, fully-scored eval store.

    No checkpoint files, no real play -- purely for exercising the
    report/plateau layer's own reading and rendering (the acceptance run
    above is what exercises the real play path end to end).
    """
    _write_run_provenance(run_dir, checkpoint_count=checkpoint_count, eval_seed=eval_seed)
    for version in scored_versions:
        _write_scored_member(run_dir, version, [1.0, 1.5])
    _write_checkpoint_markers(run_dir, scored_versions)


def _write_eval_config_file(path, config: EvalConfig):
    path.write_text(json.dumps(config.to_dict()))
    return path


# --- run_watch_loop's report-hook seam -------------------------------------------


def test_on_member_complete_fires_exactly_once_per_newly_completed_member(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=2)
    config = _eval_config(run_dir)
    calls: list[int] = []

    run_watch_loop(
        run_dir, config, FAST_PROFILE, GAME, single_pass=True, on_member_complete=calls.append
    )
    assert calls == [1, 2]

    # A member already complete before a poll examines it never re-fires the
    # hook (its `pending` list is empty -- the loop never reaches the callback
    # for it): a second, idempotent pass calls the hook zero more times.
    calls.clear()
    run_watch_loop(
        run_dir, config, FAST_PROFILE, GAME, single_pass=True, on_member_complete=calls.append
    )
    assert calls == []


def test_run_watch_loop_defaults_to_no_hook_at_all(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=1)
    result = run_watch_loop(run_dir, _eval_config(run_dir), FAST_PROFILE, GAME, single_pass=True)
    assert result.completed_members == (1,)  # no TypeError, no hook required


# --- bench_candidate / --bench -----------------------------------------------------


def test_bench_candidate_measures_the_newest_published_member(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=2)  # real provenance + two real checkpoints
    config = _eval_config(run_dir, pairs_per_cell=3)

    result = bench_candidate(run_dir, config, FAST_PROFILE, GAME, n_pairs=1)

    assert isinstance(result, BenchResult)
    assert result.candidate_version == 2  # newest schedulable, not the earliest
    assert result.eval_sims == config.eval_sims
    assert result.device == config.device
    assert result.pairs_per_cell == config.pairs_per_cell
    assert result.bench_pairs_per_cell == 1
    assert result.cells_benched > 0
    assert result.games_played == result.cells_benched * 1 * 2
    assert result.elapsed_seconds > 0.0
    assert result.seconds_per_game == pytest.approx(result.elapsed_seconds / result.games_played)
    assert result.games_per_hour == pytest.approx(3600.0 / result.seconds_per_game)
    assert result.games_per_checkpoint == result.cells_benched * config.pairs_per_cell * 2
    assert result.projected_hours_per_checkpoint == pytest.approx(
        result.games_per_checkpoint * result.seconds_per_game / 3600.0
    )
    # No metrics markers exist at all in this fixture -- an undefined cadence,
    # never a fabricated 0.0, and therefore an undefined feasibility verdict.
    assert result.publish_cadence_hours is None
    assert result.feasible is None


def test_bench_candidate_rejects_with_no_published_member_yet(tmp_path):
    run_dir = tmp_path / "run"
    _write_run_provenance(run_dir, checkpoint_count=1, eval_seed=1)
    checkpoint_dir(run_dir).mkdir(parents=True)  # no ckpt-*.pt inside
    config = _eval_config(run_dir)
    with pytest.raises(ValueError, match="no schedulable member checkpoint"):
        bench_candidate(run_dir, config, FAST_PROFILE, GAME, n_pairs=1)


def test_bench_candidate_rejects_non_positive_n_pairs(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=1)
    config = _eval_config(run_dir)
    with pytest.raises(ValueError, match="n_pairs"):
        bench_candidate(run_dir, config, FAST_PROFILE, GAME, n_pairs=0)


def test_publish_cadence_hours_averages_consecutive_gpu_hour_gaps(tmp_path, monkeypatch):
    class _FakeReduced:
        checkpoints = {1: (10, 100, 2.0), 2: (20, 200, 5.0), 3: (30, 300, 9.0)}

    monkeypatch.setattr(eval_run, "reduce_run", lambda run_dir: _FakeReduced())
    assert eval_run._publish_cadence_hours(tmp_path, k_total=3) == pytest.approx((3.0 + 4.0) / 2)


def test_publish_cadence_hours_none_with_fewer_than_two_members(tmp_path, monkeypatch):
    class _FakeReduced:
        checkpoints = {1: (10, 100, 2.0)}

    monkeypatch.setattr(eval_run, "reduce_run", lambda run_dir: _FakeReduced())
    assert eval_run._publish_cadence_hours(tmp_path, k_total=3) is None


def test_publish_cadence_hours_ignores_versions_beyond_k_total(tmp_path, monkeypatch):
    class _FakeReduced:
        checkpoints = {1: (10, 100, 2.0), 2: (20, 200, 5.0), 99: (999, 999, 999.0)}

    monkeypatch.setattr(eval_run, "reduce_run", lambda run_dir: _FakeReduced())
    assert eval_run._publish_cadence_hours(tmp_path, k_total=2) == pytest.approx(3.0)


def test_render_bench_includes_every_documented_field(tmp_path):
    run_dir = tmp_path / "run"
    _build_watched_run(run_dir, k_total=1)
    config = _eval_config(run_dir)
    result = bench_candidate(run_dir, config, FAST_PROFILE, GAME, n_pairs=1)

    rendered = run_eval._render_bench(result)

    for token in (
        "candidate_version",
        "device",
        "eval_sims",
        "cells_benched",
        "games_played",
        "seconds_per_game",
        "games_per_hour",
        "projected per checkpoint",
        "publish_cadence_hours",
        "feasible",
    ):
        assert token in rendered


# --- scripts/run_eval.py: --report / --plateau, against a hand-built store --------


def test_cmd_report_shows_null_delta_before_the_k_set_completes(tmp_path):
    run_dir = tmp_path / "run"
    _hand_built_store(run_dir, checkpoint_count=2, scored_versions=[1])
    config = _make_config(str(run_dir))
    cfg_path = _write_eval_config_file(tmp_path / "eval_cfg.json", config)

    payload = run_eval.cmd_report(cfg_path)

    assert payload["verdict"]["checkpoints_evaluated"] == 1
    assert payload["verdict"]["k_target"] == 2
    assert payload["verdict"]["delta"] is None
    assert payload["verdict"]["authoritative"] is False
    assert "delta: null" in run_eval._render_report(payload)


def test_cmd_report_shows_delta_and_the_authoritative_distinction_once_complete(tmp_path):
    run_dir = tmp_path / "run"
    _hand_built_store(run_dir, checkpoint_count=1, scored_versions=[1])
    config = _make_config(str(run_dir), bootstrap_b=39)
    cfg_path = _write_eval_config_file(tmp_path / "eval_cfg.json", config)

    payload = run_eval.cmd_report(cfg_path)

    assert payload["eval_lag"] == 0
    assert payload["verdict"]["checkpoints_evaluated"] == 1 == payload["verdict"]["k_target"]
    assert payload["verdict"]["delta"] is not None
    assert payload["verdict"]["authoritative"] is False  # B=39, not the pinned production B
    rendered = run_eval._render_report(payload)
    assert "eval_lag: 0" in rendered
    assert "authoritative: False" in rendered


def test_cmd_plateau_smoke_reports_insufficient_data_tri_state_rendered_not_coerced(tmp_path):
    run_dir = tmp_path / "run"
    _hand_built_store(run_dir, checkpoint_count=1, scored_versions=[1])
    config = _make_config(str(run_dir))
    cfg_path = _write_eval_config_file(tmp_path / "eval_cfg.json", config)

    result = run_eval.cmd_plateau(cfg_path)

    assert isinstance(result, PlateauResult)
    assert result.outcome == PLATEAU_OUTCOME_INSUFFICIENT_DATA

    rendered = run_eval._render_plateau(result)
    # The tri-state is rendered as its own literal name -- never coerced to a
    # True/False-shaped string anywhere on its own line.
    assert rendered.splitlines()[0] == f"outcome: {PLATEAU_OUTCOME_INSUFFICIENT_DATA}"
    assert "True" not in rendered.splitlines()[0]
    assert "False" not in rendered.splitlines()[0]


# --- partial-cell report golden (P2.2): never opens a mid-write cell --------------


def test_report_with_a_cell_mid_write_never_opens_it_and_matches_last_complete_prefix(tmp_path):
    """A report generated while a later member's cell is mid-write must never
    open that cell's file, and must byte-equal the report generated from the
    last complete prefix.

    Registering member 2's required cell set and partially writing (header +
    one of its two declared pairs, never ``complete_cell``-ed) into exactly
    one of its cells changes nothing ``load_snapshot`` can see: a "scheduled"
    (not "complete") cell contributes to neither ``completed_cell_ids`` nor
    the fit, so the second report is computed from *exactly* the same
    evidence as the first -- and if ``load_snapshot`` had tried to open the
    partial file for its post-hoc pair-count cross-check (the check completed
    cells alone are subject to), it would raise ``ManifestError`` (the file
    has only 1 of its declared 2 pairs); the absence of that error is itself
    proof the partial file was never opened.
    """
    run_dir = tmp_path / "run"
    _hand_built_store(run_dir, checkpoint_count=2, scored_versions=[1])
    b = 39

    report_before = build_verdict(run_dir, B=b)
    verdict_bytes_before = verdict_path(run_dir).read_bytes()
    elo_curve_bytes_before = elo_curve_path(run_dir).read_bytes()

    header2 = _lag_header(candidate_version=2, rung=7, opponent_id="random", n_pairs=2)
    register_member(run_dir, 2, [header2.cell_id.to_string()])
    path2 = cell_path(run_dir, header2.cell_id.to_string())
    next_index = open_cell_for_write(run_dir, header2)
    append_pair_record(
        path2,
        PairRecord(
            pair_index=next_index,
            pair_seed=next_index,
            score_a=1.0,
            games=(
                GameRecordSnapshot(plies=1, opening=0),
                GameRecordSnapshot(plies=1, opening=0),
            ),
        ),
    )
    # Deliberately never complete_cell()'d: this cell stays "scheduled" --
    # a real, on-disk, mid-write cell -- for the rest of this test.

    report_after = build_verdict(run_dir, B=b)

    assert report_after == report_before
    assert verdict_path(run_dir).read_bytes() == verdict_bytes_before
    assert elo_curve_path(run_dir).read_bytes() == elo_curve_bytes_before

    # And via the script's own --report entrypoint too.
    config = _make_config(str(run_dir), bootstrap_b=b)
    cfg_path = _write_eval_config_file(tmp_path / "eval_cfg.json", config)
    payload = run_eval.cmd_report(cfg_path)
    assert payload["verdict"] == report_after == report_before
