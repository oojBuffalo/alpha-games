"""Minimal micro-Blokus loop battery (§12 M2.5, task 6.3/6.4).

Two layers, both CPU-only and seeded:

1. **Self-play mechanics** (``core/selfplay.py``, no network — the M0 evaluator
   path keeps these tests stdlib-fast): the D10 boundary (sampled ∝ raw N below
   ``k_temp``, argmax N at and after it, with ``MCTS.best_action``'s tie-break),
   verbatim sparse policy targets that survive subtree reuse, the sparse D12
   sample shape (never a dense 225-vector), mover-relative ``z``/aux backfill
   read back through the public ``Game`` surface, seeded reproducibility, and
   replay-window eviction.
2. **Loop assembly** (``scripts/run_micro.py``): a tiny seeded run completes its
   games and steps, the checkpoint reloads carrying the micro orientation hash
   and the run seed, and the persisted run record's per-step losses match the
   in-memory series — the property task 7's exit gate depends on, since it reads
   the predicates from the file and never recomputes them.

The micro instance is the §5.3 config through the same ``games/blokus_duo/``
package; nothing here hardcodes a full-game constant.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.mcts import MCTS
from core.runconfig import RunConfig, load_run_config
from core.seeding import GameRNGs
from core.selfplay import (
    RUN_RECORD_SCHEMA,
    ReplayWindow,
    Sample,
    load_run_record,
    play_game,
    policy_target,
    select_move,
)
from games.blokus_duo import BlokusDuo
from games.blokus_duo.config import MICRO_CONFIG
from games.blokus_duo.pieces import orientation_table_hash

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_micro.py"

# The §5.3 micro instance: a config argument, not a fork.
MICRO = BlokusDuo(config=MICRO_CONFIG)
MICRO_NUM_ACTIONS = 225  # (5, 5, 9) — the dense size no stored target may reach.


def load_driver():
    """Import ``scripts/run_micro.py`` as a module (``scripts/`` is not a package).

    Returns:
        The imported driver module.
    """
    spec = importlib.util.spec_from_file_location("run_micro", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass resolution needs the sys.modules entry
    spec.loader.exec_module(module)
    return module


DRIVER = load_driver()


def self_play_cfg(**overrides):
    """Build a tiny :class:`~core.runconfig.SelfPlayConfig` off the pinned one.

    Args:
        **overrides: Fields to replace (e.g. ``sims``, ``k_temp``).

    Returns:
        The overridden config.
    """
    return dataclasses.replace(load_run_config().self_play, **overrides)


def tiny_run_config(**training_overrides) -> RunConfig:
    """Build a tiny but coherent run config off the pinned micro config.

    Only the budget shrinks: the game instance, the D7 constants, and λ_aux stay
    exactly as pinned, so the loop under test is the pinned loop.

    Args:
        **training_overrides: ``TrainingConfig`` fields to replace.

    Returns:
        A validated :class:`~core.runconfig.RunConfig` with a few-games budget.

    Raises:
        ValueError: Propagated from the config's own pacing/range validation.
    """
    cfg = load_run_config()
    training = dict(
        games=2,
        learner_steps=2,
        steps_per_game=1,
        batch_size=8,
        replay_window=40,
        warmup_steps=0,
        cosine_total_steps=8,
    )
    training.update(training_overrides)
    return dataclasses.replace(
        cfg,
        self_play=dataclasses.replace(cfg.self_play, sims=8),
        training=dataclasses.replace(cfg.training, **training),
    )


# --- D10 move selection ---------------------------------------------------------


def test_argmax_at_and_after_k_temp_is_deterministic():
    """At and after ``k_temp`` the most-visited action is played, RNG untouched."""
    visits = {7: 3, 2: 11, 40: 5}
    for ply in (4, 5, 99):
        rng = random.Random(0)
        assert select_move(visits, ply, 4, rng) == 2
        # The argmax branch must not consume the move-selection stream: an
        # untouched Random still yields its first value.
        assert rng.random() == random.Random(0).random()


def test_argmax_tie_break_is_lowest_action_id_and_matches_best_action():
    """Ties go to the lowest action id — ``MCTS.best_action``'s rule, not dict order."""
    assert select_move({40: 3, 7: 3, 2: 3}, 0, 0, random.Random(0)) == 2

    search = MCTS(MICRO)
    search.run(24, MICRO.initial_state())
    visits = search.action_visit_counts()
    assert select_move(visits, 0, 0, random.Random(0)) == search.best_action()


def test_below_k_temp_samples_proportional_to_raw_visit_counts():
    """τ = 1 with no exponentiation: empirical frequencies track ``N / ΣN``."""
    visits = {5: 1, 9: 3, 12: 16}
    rng = random.Random(20250812)
    draws = [select_move(visits, 0, 4, rng) for _ in range(6000)]
    total = sum(visits.values())
    for action, count in visits.items():
        assert draws.count(action) / len(draws) == pytest.approx(count / total, abs=0.02)
    # A softmax/exponentiated rule would starve the low-count actions.
    assert draws.count(5) > 0


def test_below_k_temp_sampling_is_seeded():
    """Same stream, same sequence — the D10 sample is a pure function of the rng."""
    visits = {5: 4, 9: 6, 12: 7}
    first = [select_move(visits, 0, 4, random.Random(11)) for _ in range(1)]
    again = [select_move(visits, 0, 4, random.Random(11)) for _ in range(1)]
    assert first == again


def test_select_move_rejects_an_empty_root():
    """An empty root violates the pass invariant and must fail loudly."""
    with pytest.raises(ValueError, match="no root visit counts"):
        select_move({}, 0, 4, random.Random(0))


def test_k_temp_boundary_holds_across_a_played_game():
    """Every ply at/after ``k_temp`` plays its stored target's argmax; below, it may not."""
    cfg = self_play_cfg(sims=24, k_temp=3)
    off_argmax_below = 0
    for game_index in range(6):
        result = play_game(MICRO, None, cfg, GameRNGs.for_game(4242, game_index))
        for sample, move in zip(result.samples, result.moves, strict=True):
            argmax = min(sample.sparse_pi, key=lambda pair: (-pair[1], pair[0]))[0]
            if sample.ply >= cfg.k_temp:
                assert move == argmax
            elif move != argmax:
                off_argmax_below += 1
    assert off_argmax_below > 0, "sampling below k_temp never deviated from argmax"


# --- policy targets -------------------------------------------------------------


def test_policy_target_mirrors_the_root_counts_verbatim():
    """The stored pairs are the root's raw ``N``, unnormalized, over the legal set only."""
    search = MCTS(MICRO)
    root_state = MICRO.initial_state()
    search.run(32, root_state)
    visits = search.action_visit_counts()
    pairs = policy_target(visits)

    assert pairs == list(visits.items())
    assert dict(pairs) == visits
    assert [a for a, _ in pairs] == list(MICRO.legal_moves(root_state))
    assert all(isinstance(n, int) for _, n in pairs)


def test_stored_targets_stay_sparse_and_survive_subtree_reuse():
    """Sparse over the legal set (never 225-dense), and ΣN may exceed the sim budget."""
    cfg = self_play_cfg(sims=16, k_temp=99)
    result = play_game(MICRO, None, cfg, GameRNGs.for_game(7, 0))

    inflated = 0
    for sample in result.samples:
        ids = [a for a, _ in sample.sparse_pi]
        assert len(ids) == len(set(ids))
        assert len(ids) < MICRO_NUM_ACTIONS
        total = sum(n for _, n in sample.sparse_pi)
        if sample.ply > 0:
            # Subtree reuse hands the promoted root its share of the previous
            # search's visits, so ΣN is *not* clamped to cfg.sims.
            assert total >= cfg.sims
            inflated += total > cfg.sims
    assert inflated > 0


def test_first_ply_target_totals_the_sim_budget_minus_root_expansion():
    """A fresh root spends one simulation expanding itself; the rest hit edges."""
    cfg = self_play_cfg(sims=16, k_temp=99)
    result = play_game(MICRO, None, cfg, GameRNGs.for_game(7, 0))
    assert sum(n for _, n in result.samples[0].sparse_pi) == cfg.sims - 1


# --- terminal backfill ----------------------------------------------------------


def test_backfilled_targets_are_mover_relative():
    """Each record's ``(z, aux)`` equals the public surface's targets for *its* mover."""
    cfg = self_play_cfg(sims=12, k_temp=2)
    result = play_game(MICRO, None, cfg, GameRNGs.for_game(99, 3))
    assert MICRO.is_terminal(result.terminal_state)
    assert {s.mover for s in result.samples} == {0, 1}

    for sample in result.samples:
        z, aux = MICRO.training_targets(result.terminal_state, sample.mover)
        assert sample.z == z
        assert sample.aux == aux
        assert sample.z == MICRO.terminal_utility(result.terminal_state, sample.mover)
    # Zero-sum: the two movers' targets are opposite (or both 0 on a draw).
    by_mover = {s.mover: s.z for s in result.samples}
    assert by_mover[0] == -by_mover[1]


def test_every_stored_sample_carries_pi_z_and_aux():
    """D12: no stored sample is value-only or target-free."""
    cfg = self_play_cfg(sims=8, k_temp=2)
    result = play_game(MICRO, None, cfg, GameRNGs.for_game(5, 1))
    num_aux = len(MICRO.value_targets.aux_names)
    assert num_aux == 1
    assert len(result.samples) == result.plies > 0

    for sample in result.samples:
        assert sample.sparse_pi
        assert sample.z is not None
        assert sample.aux is not None and len(sample.aux) == num_aux
        assert abs(sample.aux[0]) <= 1.0
        row = sample.training_row(num_aux)
        assert len(row) == 4
        assert row[1] is sample.sparse_pi


def test_training_row_rejects_an_unbackfilled_sample():
    """A sample that never saw the terminal must not silently reach collate."""
    sample = Sample(planes=(), sparse_pi=((0, 1),), ply=0, mover=0)
    with pytest.raises(ValueError, match="never backfilled"):
        sample.training_row(1)


def test_encoded_planes_are_the_micro_surface():
    """Stored planes come from the adapter's declared encoding (12 planes, 5×5)."""
    cfg = self_play_cfg(sims=8, k_temp=2)
    result = play_game(MICRO, None, cfg, GameRNGs.for_game(5, 2))
    planes = result.samples[0].planes
    assert len(planes) == MICRO.input_planes == 12
    assert (len(planes[0]), len(planes[0][0])) == MICRO.input_shape == (5, 5)


# --- determinism ----------------------------------------------------------------


def test_same_seed_reproduces_the_whole_game():
    """Moves, sparse π, and backfilled targets are a pure function of (seed, index)."""
    cfg = self_play_cfg(sims=16, k_temp=4)
    first = play_game(MICRO, None, cfg, GameRNGs.for_game(31337, 2))
    again = play_game(MICRO, None, cfg, GameRNGs.for_game(31337, 2))

    assert first.moves == again.moves
    assert first.utilities == again.utilities
    assert [s.sparse_pi for s in first.samples] == [s.sparse_pi for s in again.samples]
    assert [(s.z, s.aux, s.ply, s.mover) for s in first.samples] == [
        (s.z, s.aux, s.ply, s.mover) for s in again.samples
    ]


def test_root_noise_off_changes_the_game_but_not_its_legality():
    """D7 is an option: disabling it yields a different, still-legal game."""
    noisy = self_play_cfg(sims=16, k_temp=4, root_noise=True)
    clean = dataclasses.replace(noisy, root_noise=False)
    with_noise = play_game(MICRO, None, noisy, GameRNGs.for_game(8, 0))
    without = play_game(MICRO, None, clean, GameRNGs.for_game(8, 0))

    for result in (with_noise, without):
        state = MICRO.initial_state()
        for move in result.moves:
            assert move in MICRO.legal_moves(state)
            state = MICRO.apply(state, move)
        assert MICRO.is_terminal(state)
    # Uniform priors + noise vs. uniform priors alone: the searches differ.
    assert with_noise.moves != without.moves


# --- replay window --------------------------------------------------------------


def make_sample(ply: int) -> Sample:
    """Build a trivially identifiable backfilled sample.

    Args:
        ply: The sample's ply, used as its identity in eviction assertions.

    Returns:
        The sample.
    """
    return Sample(planes=(), sparse_pi=((0, 1),), ply=ply, mover=ply % 2, z=1.0, aux=(0.5,))


def test_replay_window_evicts_the_oldest_samples():
    """Capacity is a hard bound; the survivors are the most recent ones."""
    window = ReplayWindow(5)
    window.extend(make_sample(i) for i in range(3))
    assert len(window) == 3
    window.extend(make_sample(i) for i in range(3, 9))
    assert len(window) == window.capacity == 5
    assert window.total_added == 9
    assert min(s.ply for s in window.sample_batch(200, random.Random(0))) >= 4


def test_replay_window_sampling_is_seeded_and_uniform():
    """Batches are a pure function of the window-sampling stream, drawn with replacement."""
    window = ReplayWindow(10)
    window.extend(make_sample(i) for i in range(4))
    first = [s.ply for s in window.sample_batch(16, random.Random(3))]
    again = [s.ply for s in window.sample_batch(16, random.Random(3))]
    assert first == again
    assert len(first) == 16  # more draws than the window holds: with replacement
    assert set(first) <= {0, 1, 2, 3}


def test_replay_window_rejects_degenerate_use():
    """A zero capacity or an empty-window draw is a pacing bug, not a silent no-op."""
    with pytest.raises(ValueError, match="capacity must be positive"):
        ReplayWindow(0)
    with pytest.raises(ValueError, match="empty replay window"):
        ReplayWindow(4).sample_batch(2, random.Random(0))


# --- loop assembly --------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Run the tiny loop once and share the result across the assembly tests.

    Args:
        tmp_path_factory: pytest's session-scoped temp-directory factory.

    Returns:
        ``(cfg, record, run_dir)`` for the completed run.
    """
    cfg = tiny_run_config()
    run_dir = tmp_path_factory.mktemp("micro_run")
    record = DRIVER.run_loop(cfg, run_dir=run_dir, device="cpu")
    return cfg, record, run_dir


def test_run_completes_its_configured_budget(tiny_run):
    """Games and learner steps both hit the configured counts, with finite losses."""
    cfg, record, _ = tiny_run
    assert len(record.games) == cfg.training.games
    assert len(record.steps) == cfg.training.learner_steps
    assert [s["step"] for s in record.steps] == list(range(cfg.training.learner_steps))

    for step in record.steps:
        for key in ("policy_loss", "value_loss", "aux_loss", "total_loss"):
            assert math.isfinite(step[key]), (key, step)
        assert step["policy_loss"] > 0.0
        assert step["window_size"] > 0
    for game in record.games:
        assert game["plies"] == game["samples"] > 0
        assert game["utilities"][0] == -game["utilities"][1]


def test_run_record_persists_the_loss_series_the_gate_reads(tiny_run):
    """Task 7's predicates read the file: it must match the in-memory series exactly."""
    _, record, run_dir = tiny_run
    persisted = load_run_record(run_dir / DRIVER.RUN_RECORD_NAME)

    assert persisted["schema"] == RUN_RECORD_SCHEMA
    assert persisted["run_seed"] == record.run_seed
    assert persisted["config"] == record.config
    for key in ("policy_loss", "value_loss", "aux_loss", "total_loss"):
        assert [s[key] for s in persisted["steps"]] == record.loss_series(key)
    assert persisted["game_identity"]["orientation_hash"] == orientation_table_hash(MICRO_CONFIG)
    assert persisted["game_identity"]["game_config"] == "MICRO_CONFIG"


def test_run_record_reader_rejects_a_foreign_schema(tmp_path):
    """An unknown schema tag fails loudly rather than being read as evidence."""
    path = tmp_path / "run_record.json"
    path.write_text('{"schema": "something-else", "steps": []}')
    with pytest.raises(ValueError, match="unknown run-record schema"):
        load_run_record(path)


def test_final_checkpoint_reloads_with_its_identity(tiny_run):
    """Weights plus the identity M3 validates on load: orientation hash and run seed."""
    cfg, record, run_dir = tiny_run
    final = [c for c in record.checkpoints if c["kind"] == "final"]
    assert len(final) == 1
    assert final[0]["step"] == cfg.training.learner_steps

    blob = torch.load(final[0]["path"], map_location="cpu", weights_only=True)
    assert blob["schema"] == DRIVER.CHECKPOINT_SCHEMA
    assert blob["orientation_hash"] == orientation_table_hash(MICRO_CONFIG)
    assert blob["orientation_hash"] != orientation_table_hash()  # not the full game's table
    assert blob["run_seed"] == cfg.run_seed
    assert blob["config"] == cfg.to_dict()
    assert tuple(blob["network_config"]["policy_shape"]) == MICRO.policy_shape
    assert blob["network_config"]["trunk_blocks"] == 8  # D5 trunk, pinned for micro
    assert blob["model_state_dict"], "no weights saved"


def test_periodic_checkpoints_are_written_on_request(tmp_path):
    """``--checkpoint-every`` adds periodic checkpoints alongside the final one."""
    cfg = tiny_run_config(games=2, learner_steps=4, steps_per_game=2)
    record = DRIVER.run_loop(cfg, run_dir=tmp_path, device="cpu", checkpoint_every=2)
    kinds = [c["kind"] for c in record.checkpoints]
    assert kinds == ["periodic", "periodic", "final"]
    assert all(Path(c["path"]).exists() for c in record.checkpoints)


def test_two_runs_with_the_same_seed_match_step_for_step(tmp_path):
    """Same run seed → identical game trace and identical loss trace, end to end."""
    cfg = tiny_run_config()
    first = DRIVER.run_loop(cfg, run_dir=tmp_path / "a", device="cpu")
    again = DRIVER.run_loop(cfg, run_dir=tmp_path / "b", device="cpu")

    assert [g["moves"] for g in first.games] == [g["moves"] for g in again.games]
    assert [g["utilities"] for g in first.games] == [g["utilities"] for g in again.games]
    for key in ("policy_loss", "value_loss", "aux_loss", "total_loss", "learning_rate"):
        assert [s[key] for s in first.steps] == [s[key] for s in again.steps]

    # ... and a different run seed does not reproduce it.
    other = DRIVER.run_loop(
        dataclasses.replace(cfg, run_seed=cfg.run_seed + 1), run_dir=tmp_path / "c", device="cpu"
    )
    assert other.loss_series("total_loss") != first.loss_series("total_loss")


def test_loss_falls_over_a_fixed_window(tmp_path):
    """One game, then many steps on that frozen window: the losses go down.

    The micro rehearsal of the §12 M2.5 tail-vs-head predicates task 7 evaluates
    for real — here only that the learner is wired end to end (real targets, real
    gradients), not that the run passes the gate.
    """
    cfg = tiny_run_config(
        games=1, learner_steps=30, steps_per_game=30, batch_size=8, cosine_total_steps=30
    )
    record = DRIVER.run_loop(cfg, run_dir=tmp_path, device="cpu")

    assert {s["window_size"] for s in record.steps} == {record.games[0]["samples"]}
    for key in ("policy_loss", "value_loss", "total_loss"):
        series = record.loss_series(key)
        head, tail = series[:5], series[-5:]
        assert sum(tail) / 5 < sum(head) / 5, (key, series)


def test_driver_rejects_a_game_it_cannot_construct():
    """The driver is the micro loop's entrypoint, not a generic runner."""
    foreign = SimpleNamespace(game="othello", game_config="DEFAULT")
    with pytest.raises(ValueError, match="blokus_duo only"):
        DRIVER.build_game(foreign)
    with pytest.raises(ValueError, match="blokus_duo only"):
        DRIVER.game_identity(foreign)


def test_main_runs_a_config_file_end_to_end(tmp_path, capsys):
    """The entrypoint reads a config file and lands both artifacts in the run dir."""
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(tiny_run_config().to_dict()))
    run_dir = tmp_path / "out"

    code = DRIVER.main(
        ["--config", str(config_path), "--run-dir", str(run_dir), "--device", "cpu", "--quiet"]
    )
    assert code == 0

    record = load_run_record(run_dir / DRIVER.RUN_RECORD_NAME)
    assert len(record["steps"]) == 2
    assert (run_dir / "checkpoint_final.pt").exists()
    assert str(run_dir / DRIVER.RUN_RECORD_NAME) in capsys.readouterr().out


def test_cli_parses_the_pinned_config_by_default():
    """No arguments → the pinned ``configs/blokus_micro.json``, CPU-safe device default."""
    args = DRIVER.parse_args([])
    assert args.config.name == "blokus_micro.json"
    assert args.device == "auto"
    assert args.checkpoint_every == 0
    assert DRIVER.resolve_device("cpu").type == "cpu"
