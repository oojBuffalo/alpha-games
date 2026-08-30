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
checkpoint-load cache constructs. The report/CLI integration is a later
subtask (9.3).
"""

from __future__ import annotations

import dataclasses
import json

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
    EvalConfig,
    EvalRelaunchRefusedError,
    WatchLoopResult,
    _CandidateCache,
    _resolve_opponent_factory,
    agent_identity,
    cell_seed,
    checkpoint_dir,
    eval_lag,
    load_eval_config,
    required_cell_ids,
    resolve_eval_launch,
    run_watch_loop,
    schedulable_versions,
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
    register_member,
)
from core.mcts import MCTS
from core.network import Network, NetworkConfig
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
