"""M3 Othello full-stack zero-``core/``-diff re-check (design doc §12 M3/M1.5, issue #64).

The second zero-``core/``-diff acceptance: M1.5 proved Othello satisfied the
``Game`` ABC with zero ``core/`` edits; this battery drives it through every
M3 surface that did not exist yet at that check -- real replay shards
(``core.replay_shard``), the replay window (``core.replay_window``),
augmentation (``core.augment``), the collate/loss/train step
(``core.train``/``core.losses``), checkpoints (``core.checkpoint``), and the
actor/learner drivers wired over shared directories (``core.actor``/
``core.learner``/``core.ipc``) -- without adding or editing a single line
under ``core/``. The acceptance claim is exactly that diff: empty.

Coverage is deterministic by construction, not by hoping a small run happens
to contain the interesting cases:

* A **scripted evaluator** pins a deterministic, purely state-local move
  policy (never a fixed path -- see :func:`_lowest_legal_policy`'s docstring
  for why a path-keyed script is the wrong tool against a game that
  transposes as freely as Othello does) with overwhelming softmax mass, so
  at tiny sims (16) argmax-N reproduces the exact same trajectory through
  the *real* ``core.mcts.MCTS`` search / ``core.selfplay.select_move`` /
  ``core.selfplay.backfill_targets`` path -- no bypass. The trajectory is
  played from the real ``Othello.initial_state()`` and contains a genuine
  forced pass, the opponent's reply, and the passer's regained placement
  two plies later (§12 M1.5's non-monotone pass-regain property, which
  Blokus's monotone blocking structurally cannot exercise).
* Every one of Othello's 8 declared D4 symmetry elements is applied
  directly to real stored samples through the real
  ``core.augment.augment_sample`` -> ``core.train.collate`` ->
  ``core.train.train_step`` path, iterated in full (never a random draw),
  asserting the pass action (id 64) is a fixed point of every element.
* Real shard and checkpoint round trips assert the no-aux convention (no
  zero-filled array), the ``orientation_hash: None`` fingerprint, and a
  loud cross-game mismatch.
* One small end-to-end run drives ``core.actor.ActorDriver`` +
  ``core.learner.LearnerDriver`` over shared directories (the
  single-process integration pattern ``tests/test_ipc.py`` already proved
  for TTT/Blokus), showing encoding -> self-play -> storage -> window ->
  augmentation -> training -> checkpoint -> resume all compose for Othello.

Run ``git diff --stat origin/feature/m3-run-entrypoint -- core/`` alongside
this file: it must be empty, or the acceptance has failed (see the PR body's
finding section if it is not).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.actor import ActorDriver
from core.artifact_fingerprint import FingerprintMismatchError, build_fingerprint
from core.augment import augment_sample
from core.checkpoint import (
    build_bundle,
    list_published_versions,
    load_checkpoint,
    select_resume_bundle,
    write_published_checkpoint,
    write_resume_snapshot,
)
from core.ipc import build_actor_pacing, build_actor_refresh
from core.learner import LearnerDriver
from core.network import Network, NetworkConfig
from core.replay_shard import PendingSample, ShardWriter, read_shard
from core.runconfig import SelfPlayConfig, TrainingConfig
from core.seeding import GameRNGs
from core.selfplay import play_game
from core.train import collate, make_optimizer, make_scaler, train_step
from games.othello import PASS, Othello
from games.registry import build_game
from games.tictactoe import TicTacToe

OTHELLO = Othello()

# A tiny net over Othello's real declared encoding surface (2 planes, 8x8,
# flat 65-action head, no aux) -- speed, not throughput; only the trunk width
# and depth are hand-picked, everything else comes straight off the adapter.
NET_CONFIG = NetworkConfig(
    input_planes=OTHELLO.input_planes,
    input_shape=OTHELLO.input_shape,
    policy_shape=OTHELLO.policy_shape,
    trunk_blocks=1,
    trunk_channels=4,
    num_aux=len(OTHELLO.value_targets.aux_names),
)


# ==============================================================================
# 1. The curated pass-regain trajectory (deterministic, not a random draw)
# ==============================================================================


def _lowest_legal_policy(game, state):
    """Return the lowest legal action id at ``state``.

    A deterministic, purely **state-local** policy -- never path-dependent:
    the recommended action for a given state is always the same regardless
    of how that state was reached. This sidesteps a real pathology a
    path-keyed script (a precomputed ``state -> action`` dict built along
    one specific move order) would hit: Othello transposes freely, so an
    off-script branch explored by MCTS can coincidentally land on a
    *different* on-script state further down the dict and get spuriously
    rewarded, corrupting the very signal meant to keep search on-script. A
    state-local policy has no such failure mode -- "the correct
    continuation" is well-defined for every reachable state, on-script or
    not, so a stray branch is always scored consistently.

    Args:
        game: The adapter (only ``legal_moves`` is used).
        state: The position to pick an action for.

    Returns:
        The smallest legal action id at ``state``.
    """
    return min(game.legal_moves(state))


def _offline_playout(game, policy):
    """Play ``policy`` from ``game.initial_state()`` to terminal, off any search.

    Pure ``Game`` calls only (no MCTS) -- the ground truth the scripted
    evaluator's real-search reproduction is checked against, and the basis
    the forced-pass / pass-regain properties are verified on directly,
    before any search machinery is trusted.

    Args:
        game: The adapter to play.
        policy: ``(game, state) -> action``, a deterministic, state-local
            move rule.

    Returns:
        ``(moves, states)``: the actions played, and the states from the
        initial one (index 0) through the terminal one (index
        ``len(moves)``), inclusive.
    """
    state = game.initial_state()
    states = [state]
    moves = []
    while not game.is_terminal(state):
        action = policy(game, state)
        moves.append(action)
        state = game.apply(state, action)
        states.append(state)
    return tuple(moves), tuple(states)


_TRAJECTORY_MOVES, _TRAJECTORY_STATES = _offline_playout(OTHELLO, _lowest_legal_policy)
_FORCED_PASS_PLIES = tuple(i for i, a in enumerate(_TRAJECTORY_MOVES) if a == PASS)


def _regains_two_plies_later(ply):
    """Whether the ply-``ply`` passer has a genuine placement two plies later.

    Args:
        ply: A forced-pass ply index into :data:`_TRAJECTORY_MOVES`.

    Returns:
        ``True`` if the same mover who passed at ``ply`` has a real
        placement (not another forced pass, not terminal) at ``ply + 2`` --
        §12 M1.5's non-monotone "blocked, then mobile again" property.
    """
    mover = OTHELLO.current_player(_TRAJECTORY_STATES[ply])
    later = _TRAJECTORY_STATES[ply + 2]
    if OTHELLO.is_terminal(later) or OTHELLO.current_player(later) != mover:
        return False
    legal = list(OTHELLO.legal_moves(later))
    return legal != [PASS] and legal != []


# The curated window this battery drives through the real stack: ply
# _PASS_PLY is a genuine forced pass (mover's only legal action is PASS);
# _OPPONENT_PLY is the opponent's reply; _REGAIN_PLY is the passer's
# regained placement -- the non-monotone property Blokus's monotone
# blocking structurally cannot exercise (§12 M1.5).
_REGAIN_CANDIDATES = tuple(p for p in _FORCED_PASS_PLIES if _regains_two_plies_later(p))
_PASS_PLY = _REGAIN_CANDIDATES[0]
_OPPONENT_PLY = _PASS_PLY + 1
_REGAIN_PLY = _PASS_PLY + 2


def _scripted_evaluator(*, logit_gap=10.0, value=0.0):
    """Build a scripted evaluator pinning :func:`_lowest_legal_policy`.

    Overwhelming (>= 99.9% against up to ~30 legal siblings) softmax mass on
    the policy's recommended action at every state -- ``core.mcts.MCTS``'s
    own ``_priors`` softmaxes evaluator output as logits (never raw
    probabilities), so a fixed additive gap on the recommended action's
    logit is what "overwhelming prior" means here. This drives the *real*
    search/selection/backup path: at tiny sims, argmax-N follows the
    recommended action because it dominates PUCT's score at every node, not
    because anything is bypassed.

    Args:
        logit_gap: Logit advantage the recommended action carries over
            every other legal action.
        value: The fixed leaf value returned for every state (mover's
            perspective). Irrelevant to which action wins argmax-N here --
            the prior gap alone decides it -- but must be finite.

    Returns:
        A ``core.mcts.Evaluator`` usable with ``core.selfplay.play_game``.
    """

    def evaluate(game, state):
        legal = list(game.legal_moves(state))
        best = _lowest_legal_policy(game, state)
        return value, {a: (logit_gap if a == best else 0.0) for a in legal}

    return evaluate


def _scripted_self_play_config(sims=16):
    """A tiny, noise-free ``SelfPlayConfig`` for the scripted evaluator.

    ``k_temp=0`` makes every ply argmax-N (D10); ``root_noise=False`` keeps
    D7's exploration noise from perturbing the scripted evaluator's prior
    dominance -- both real config knobs a self-play caller can set, never a
    bypass of any search code.

    Args:
        sims: Simulations per move (default 16, the issue's "8-16" range).

    Returns:
        The config.
    """
    return SelfPlayConfig(
        sims=sims, k_temp=0, dirichlet_eps=0.25, dirichlet_alpha_numerator=10.8, root_noise=False
    )


# Played once, at import time, exactly like the offline trajectory above --
# fully deterministic (root_noise=False and k_temp=0 mean neither RNG stream
# ``play_game`` takes is ever consumed), so every test below sees the same
# result regardless of run order.
_SCRIPTED_RESULT = play_game(
    OTHELLO,
    _scripted_evaluator(),
    _scripted_self_play_config(),
    GameRNGs.for_game(run_seed=2024, game_index=0),
    model_version=1,
)


def _to_pending_samples(result, model_version):
    """Convert a played game's backfilled samples into storage-ready rows.

    Restates ``core.actor.ActorDriver``'s own ``Sample`` -> ``PendingSample``
    field mapping (that helper is private to ``core.actor``) so this battery
    still exercises the exact same real storage shape without reaching into
    another module's underscore-prefixed internals.

    Args:
        result: A finished ``core.selfplay.GameResult``.
        model_version: The pinned weight version stamped on every sample.

    Returns:
        One ``core.replay_shard.PendingSample`` per sample, in play order.

    Raises:
        AssertionError: If a sample was never backfilled with ``(z, aux)``.
    """
    pending = []
    for sample in result.samples:
        assert sample.z is not None and sample.aux is not None
        pending.append(
            PendingSample(
                planes=sample.planes,
                sparse_pi=sample.sparse_pi,
                z=sample.z,
                aux=sample.aux,
                mover=sample.mover,
                model_version=model_version,
                ply=sample.ply,
            )
        )
    return pending


def _write_and_read_shard(shard_dir, result, *, model_version):
    """Publish one game's samples as a real shard and read it back, validated.

    Args:
        shard_dir: Directory the shard (and writer-state file) publish into.
        result: The finished game whose samples to store.
        model_version: The pinned weight version stamped on every sample.

    Returns:
        ``(path, shard_data)``: the published shard's path, and the
        fingerprint-and-invariant-checked read-back
        (``core.replay_shard.read_shard``).
    """
    writer = ShardWriter(shard_dir, OTHELLO, run_id="othello-fullstack", actor_id="0")
    path = writer.write_shard([_to_pending_samples(result, model_version)])
    return path, read_shard(path, OTHELLO)


# ==============================================================================
# 2. The curated trajectory: legality, forced pass, and genuine regain
# ==============================================================================


def test_curated_trajectory_is_legal_and_contains_forced_pass_regain():
    state = OTHELLO.initial_state()
    for i, action in enumerate(_TRAJECTORY_MOVES):
        assert action in list(OTHELLO.legal_moves(state)), i
        state = OTHELLO.apply(state, action)
    assert state == _TRAJECTORY_STATES[-1]
    assert OTHELLO.is_terminal(state)

    assert len(_FORCED_PASS_PLIES) >= 1
    assert _REGAIN_CANDIDATES  # at least one forced pass with a genuine regain

    passer = OTHELLO.current_player(_TRAJECTORY_STATES[_PASS_PLY])
    assert list(OTHELLO.legal_moves(_TRAJECTORY_STATES[_PASS_PLY])) == [PASS]
    assert OTHELLO.current_player(_TRAJECTORY_STATES[_OPPONENT_PLY]) == 1 - passer
    assert OTHELLO.current_player(_TRAJECTORY_STATES[_REGAIN_PLY]) == passer
    regain_legal = list(OTHELLO.legal_moves(_TRAJECTORY_STATES[_REGAIN_PLY]))
    assert regain_legal and regain_legal != [PASS]  # a real placement, mobility regained

    utilities = tuple(OTHELLO.terminal_utility(state, p) for p in range(2))
    assert 0.0 not in utilities  # decisive: the mover-relative sign checks below are non-vacuous


# ==============================================================================
# 3. The scripted evaluator drives the real search/selection/backup/storage path
# ==============================================================================


def test_scripted_evaluator_drives_play_game_through_the_pass_regain_trajectory():
    result = _SCRIPTED_RESULT

    # Real argmax-N, at 16 sims, reproduced the curated trajectory exactly --
    # the proof that nothing here bypasses core.mcts.MCTS / core.selfplay.
    assert result.moves == _TRAJECTORY_MOVES
    assert result.terminal_state == _TRAJECTORY_STATES[-1]
    assert len(result.samples) == len(_TRAJECTORY_MOVES)
    assert [s.ply for s in result.samples] == list(range(len(_TRAJECTORY_MOVES)))
    assert 0.0 not in result.utilities

    pass_sample = result.samples[_PASS_PLY]
    opponent_sample = result.samples[_OPPONENT_PLY]
    regain_sample = result.samples[_REGAIN_PLY]

    # Explicit pass action stored on the forced-pass ply: a forced pass has
    # exactly one legal action, so pi = [(64, sum(N))].
    assert pass_sample.sparse_pi[0][0] == PASS
    assert len(pass_sample.sparse_pi) == 1
    assert pass_sample.sparse_pi[0][1] > 0

    # mover_id on the pass ply is the passer, not the opponent.
    assert pass_sample.mover == OTHELLO.current_player(_TRAJECTORY_STATES[_PASS_PLY])
    assert pass_sample.mover == regain_sample.mover
    assert pass_sample.mover != opponent_sample.mover

    # z/aux stay mover-relative through the non-monotone consecutive-mover
    # stretch: hand-derived from the decisive terminal outcome, exactly.
    expected_pass_z = OTHELLO.terminal_utility(result.terminal_state, pass_sample.mover)
    expected_opponent_z = OTHELLO.terminal_utility(result.terminal_state, opponent_sample.mover)
    assert pass_sample.z == expected_pass_z
    assert regain_sample.z == expected_pass_z  # same mover, same sign
    assert opponent_sample.z == expected_opponent_z
    assert pass_sample.z != opponent_sample.z  # opposite mover, opposite sign (decisive game)
    assert pass_sample.aux == () and opponent_sample.aux == () and regain_sample.aux == ()

    # Every forced-pass ply in the trajectory carries the explicit pass
    # action and the correct passer, not only the curated one above.
    for ply in _FORCED_PASS_PLIES:
        sample = result.samples[ply]
        assert sample.sparse_pi == ((PASS, sample.sparse_pi[0][1]),)
        assert sample.mover == OTHELLO.current_player(_TRAJECTORY_STATES[ply])


# ==============================================================================
# 4. The real storage path: a real shard, spot-checked after read-back
# ==============================================================================


def test_scripted_game_round_trips_through_a_real_shard(tmp_path):
    path, data = _write_and_read_shard(tmp_path, _SCRIPTED_RESULT, model_version=7)

    # No-aux samples end-to-end: never a zero-filled array.
    with np.load(path, allow_pickle=False) as npz:
        assert "aux" not in npz.files
    assert all(r.aux == () for r in data.records)

    # Flat (65,) head, no spatial factorization: every stored action id is in
    # range, and the pass id (64) survives the round trip untouched.
    assert OTHELLO.policy_shape == (65,)
    action_ids = [a for r in data.records for a, _ in r.sparse_pi]
    assert action_ids and min(action_ids) >= 0 and max(action_ids) < 65

    pass_record = data.records[_PASS_PLY]
    opponent_record = data.records[_OPPONENT_PLY]
    regain_record = data.records[_REGAIN_PLY]
    assert pass_record.sparse_pi[0][0] == PASS
    assert pass_record.mover == OTHELLO.current_player(_TRAJECTORY_STATES[_PASS_PLY])
    assert pass_record.z == regain_record.z
    assert pass_record.z != opponent_record.z

    # Shard header fingerprint carries orientation hash None, cleanly.
    assert data.fingerprint["orientation_hash"] is None
    assert build_fingerprint(OTHELLO)["orientation_hash"] is None


# ==============================================================================
# 5. Full 8-element D4 augmentation through the real collate + loss path
# ==============================================================================


def test_full_d4_group_augments_real_stored_samples_through_collate_and_loss(tmp_path):
    _, data = _write_and_read_shard(tmp_path, _SCRIPTED_RESULT, model_version=5)
    chosen_plies = (0, _PASS_PLY, _REGAIN_PLY, len(data.records) - 1)
    records = [data.records[p] for p in chosen_plies]
    assert any(a == PASS for r in records for a, _ in r.sparse_pi)  # the fixed-point case is live

    group = OTHELLO.symmetry_group
    assert len(group) == 8

    net = Network(NET_CONFIG)
    optimizer = make_optimizer(net)
    scaler = make_scaler("cpu")

    # Every declared element, iterated in full -- never a random draw.
    for g_index in range(len(group)):
        _, perm = group[g_index]
        assert len(perm) == 65
        assert perm[PASS] == PASS  # pass id is a fixed point of every element

        rows = []
        for record in records:
            aug_planes, aug_pairs = augment_sample(
                OTHELLO, record.planes, record.sparse_pi, g_index
            )
            expected_pairs = [(perm[a], n) for a, n in record.sparse_pi]
            assert list(aug_pairs) == expected_pairs  # permuted targets match the group action
            original = dict(record.sparse_pi)
            if PASS in original:
                assert dict(aug_pairs)[PASS] == original[PASS]  # fixed point, on real data
            rows.append((aug_planes, tuple(aug_pairs), record.z))  # no-aux row shape (§7)

        batch = collate(OTHELLO, rows)
        assert batch.aux is None

        net.train()
        parts = train_step(net, optimizer, scaler, batch)
        assert torch.isfinite(parts.total)
        assert torch.isfinite(parts.policy)
        assert torch.isfinite(parts.value)
        assert parts.aux is None


# ==============================================================================
# 6. Checkpoint: orientation hash None, cross-game mismatch, resume selection
# ==============================================================================


def test_checkpoint_round_trips_with_orientation_hash_none_and_rejects_cross_game(tmp_path):
    net = Network(NET_CONFIG)
    optimizer = make_optimizer(net)
    scaler = make_scaler("cpu")
    bundle = build_bundle(
        version=0,
        learner_step=0,
        game=OTHELLO,
        run_config={"name": "othello-fullstack-test"},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    assert bundle.fingerprint["orientation_hash"] is None

    path = write_published_checkpoint(tmp_path, bundle)
    loaded = load_checkpoint(path, OTHELLO)
    assert loaded.fingerprint == bundle.fingerprint
    assert loaded.version == 0
    assert loaded.learner_step == 0

    # Cross-game mismatch still fails loudly.
    with pytest.raises(FingerprintMismatchError):
        load_checkpoint(path, TicTacToe())

    # Resume snapshot: loads cleanly through the second reader too.
    write_resume_snapshot(tmp_path, dataclasses.replace(bundle, learner_step=5))
    resumed = select_resume_bundle(tmp_path, OTHELLO)
    assert resumed is not None
    assert resumed.learner_step == 5
    assert resumed.fingerprint["orientation_hash"] is None


# ==============================================================================
# 7. The flat 65-action head: shape, and encode/decode over the full range
# ==============================================================================


def test_policy_head_is_flat_65_with_no_spatial_factorization():
    assert OTHELLO.policy_shape == (65,)
    assert NET_CONFIG.spatial_policy is False
    assert NET_CONFIG.num_actions == 65

    net = Network(NET_CONFIG)
    assert net.policy_fc is not None  # the flat branch: 1x1 conv reduction -> FC, no permute

    x = torch.zeros(2, OTHELLO.input_planes, *OTHELLO.input_shape)
    logits, value, aux = net(x)
    assert tuple(logits.shape) == (2, 65)
    assert tuple(value.shape) == (2,)
    assert aux is None

    for action in range(65):
        move = OTHELLO.decode_action(action)
        assert OTHELLO.encode_action(move) == action
    assert OTHELLO.decode_action(PASS) == "pass"
    assert OTHELLO.encode_action("pass") == PASS


# ==============================================================================
# 8. The registry entry (games/registry.py, issue #63's task-12 surface)
# ==============================================================================


def test_registry_builds_othello_through_the_task12_registry_entry():
    game = build_game("othello", None)
    assert isinstance(game, Othello)
    assert game.policy_shape == (65,)


# ==============================================================================
# 9. Full-stack smoke: actor + learner over shared directories, then resume
# ==============================================================================


def _actor_self_play_config():
    """A real (non-scripted), D6-validate-tier ``SelfPlayConfig`` for the smoke test."""
    return SelfPlayConfig(
        sims=128, k_temp=4, dirichlet_eps=0.25, dirichlet_alpha_numerator=10.8, root_noise=True
    )


def _smoke_training_config(games):
    """A ``TrainingConfig`` sized for a fast Othello full-stack smoke run.

    ``replay_warmup_positions`` is kept deliberately high (never reached by
    this test's tiny window) -- the same judgment call ``tests/test_ipc.py``
    documents: this smoke test's job is proving encoding -> self-play ->
    storage -> window -> augmentation -> training -> checkpoint -> resume
    all *compose*, not exercising D5 replay-ratio pacing dynamics. With a
    single actor alternating one game / one learner step, real floor/ceiling
    enforcement would starve on so little data and deadlock the pacing
    wiring (the actor blocks on "hold" until more positions land, which
    only happens if the actor itself plays another game).

    Args:
        games: Self-play games (and, one-to-one, learner steps) the smoke
            test drives.

    Returns:
        The config.
    """
    return TrainingConfig(
        games=games,
        learner_steps=games,
        steps_per_game=1,
        batch_size=4,
        replay_window=5000,
        learning_rate=1e-2,
        warmup_steps=0,
        cosine_total_steps=50,
        aux_loss_weight=0.0,
        checkpoint_selection="final",
        publish_interval=1,
        checkpoint_count=games + 1,
        replay_warmup_positions=10_000,
    )


def _stub_run_config(training, run_seed):
    """A minimal duck-typed run-config stand-in (mirrors ``tests/test_ipc.py``).

    ``core.learner.LearnerDriver`` only ever reads ``.training``, ``.run_seed``
    and ``.to_dict()`` off its ``run_config`` argument -- Othello, like TTT,
    declares no named ``GAME_CONFIGS`` (only ``blokus_duo`` does), so a real
    ``core.runconfig.RunConfig`` cannot name it; this stand-in is the exact
    pattern the M3 stack already uses for every other such game.

    Args:
        training: The training sub-config.
        run_seed: The run's recorded root seed.

    Returns:
        A duck-typed object satisfying ``LearnerDriver``'s ``run_config``
        contract.
    """
    return SimpleNamespace(
        training=training,
        run_seed=run_seed,
        to_dict=lambda: {"training": dataclasses.asdict(training), "run_seed": run_seed},
    )


@pytest.mark.slow
def test_full_stack_smoke_othello_actor_learner_checkpoint_resume(tmp_path):
    """encoding -> self-play -> storage -> window -> augmentation -> training
    -> checkpoint -> resume, all composing for Othello -- the single-process
    integration pattern ``tests/test_ipc.py`` already proved for TTT/Blokus,
    reused here with the real (non-scripted) actor/learner drivers."""
    shard_dir, ckpt_dir, run_dir = tmp_path / "shards", tmp_path / "ckpt", tmp_path / "run"
    run_seed = 4242
    training = _smoke_training_config(games=2)

    learner = LearnerDriver(
        game=OTHELLO,
        run_config=_stub_run_config(training, run_seed),
        shard_dir=shard_dir,
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
        network_config=NET_CONFIG,
    )  # publishes v0 as a side effect of __init__

    refresh = build_actor_refresh(game=OTHELLO, ckpt_dir=ckpt_dir, network_config=NET_CONFIG)
    pacing = build_actor_pacing(run_dir)
    actor = ActorDriver(
        game=OTHELLO,
        self_play=_actor_self_play_config(),
        run_id="othello-fullstack",
        actor_id=0,
        out_dir=shard_dir,
        run_seed=run_seed,
        refresh=refresh,
        pacing=pacing,
        max_games=1,
    )

    shard_paths = []
    for _ in range(training.games):
        (path,) = actor.run()  # encoding -> real MCTS self-play -> storage
        shard_paths.append(path)
        learner._run_step()  # window rescan -> augmentation -> train_step -> maybe-publish

    assert len(shard_paths) == training.games
    for path in shard_paths:
        data = read_shard(path, OTHELLO)
        assert data.records
        assert data.fingerprint["orientation_hash"] is None

    learner.window.rescan()
    assert learner.window.positions_stored == sum(
        len(read_shard(p, OTHELLO).records) for p in shard_paths
    )
    assert len(list_published_versions(ckpt_dir)) >= 2  # v0 + at least one real publish
    assert learner.step == training.games

    # checkpoint -> resume: a fresh LearnerDriver over the same directories
    # picks up exactly where the first one left off.
    resumed_learner = LearnerDriver(
        game=OTHELLO,
        run_config=_stub_run_config(training, run_seed),
        shard_dir=shard_dir,
        ckpt_dir=ckpt_dir,
        run_dir=run_dir,
        network_config=NET_CONFIG,
    )
    assert resumed_learner.step == learner.step
