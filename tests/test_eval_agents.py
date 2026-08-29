"""Checkpoint-backed evaluator load path + rung-5 network-policy agent (§9, M4).

CPU-only, seeded. Covers the spec's Test Strategy: the distinct-weights
golden (the P1 killer -- two saved checkpoints with known-different weights
must load into evaluators that produce different logits *and* different
chosen actions, each labeled its own ``model_version``); rung-5 argmax
against a hand-computed masked-softmax golden, including the lowest-id
tie-break; determinism with no RNG consumed; one tampered-fingerprint
integration case delegating to the m3 checkpoint battery's own tampering
pattern (mismatched game, not a re-test of the whole battery); an end-to-end
mirrored micro-Blokus pair through ``play_pairs`` +
``games.blokus_duo.baselines.start_square_balancer`` with a rung-5 agent; and
the reflective every-``Game``-ABC-member delegation audit for
``core.runner._OpeningRestricted``.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

import core.eval_agents as eval_agents_module
from core import RandomAgent
from core.artifact_fingerprint import FingerprintMismatchError
from core.checkpoint import build_bundle, write_published_checkpoint
from core.eval_agents import (
    NetworkPolicyAgent,
    load_eval_network,
    rung5_agent_factory,
)
from core.game import Game
from core.network import Network, NetworkConfig
from core.runner import _OpeningRestricted, play_pairs
from core.train import make_optimizer, make_scaler
from games.blokus_duo import BlokusDuo
from games.blokus_duo.baselines import start_square_balancer
from games.blokus_duo.config import MICRO_CONFIG
from games.othello import Othello
from games.tictactoe import TicTacToe

MICRO = BlokusDuo(config=MICRO_CONFIG)
OTHELLO = Othello()


def _tiny_network_config(game):
    """A ``NetworkConfig`` matching ``game``'s declared surface, tiny trunk.

    Mirrors ``tests/test_checkpoint.py``'s ``_tiny_ttt_net`` pattern (a small,
    fast-to-build net for CPU tests) -- and, since ``core.eval_agents``
    reconstructs the trunk shape from the saved weights rather than
    ``NetworkConfig.from_game``, this deliberately non-default trunk
    (1 block x 4 channels, vs. D5's 8x128) is exactly what proves that.
    """
    return NetworkConfig(
        input_planes=game.input_planes,
        input_shape=tuple(game.input_shape),
        policy_shape=tuple(game.policy_shape),
        trunk_blocks=1,
        trunk_channels=4,
        num_aux=len(game.value_targets.aux_names),
    )


def _write_checkpoint(tmp_path, game, *, version, seed, sub_dir="ckpt"):
    """Build and publish one tiny, seeded real checkpoint for ``game``.

    Args:
        tmp_path: Pytest tmp dir.
        game: The adapter to train/validate against.
        version: The published model-version ordinal.
        seed: ``torch.manual_seed`` before construction -- the weights are a
            deterministic function of this seed.
        sub_dir: Sub-directory name (distinct checkpoints need distinct
            checkpoint directories, since ``ckpt-<version>.pt`` is immutable
            per directory but two calls may share a version number).

    Returns:
        The published checkpoint's path.
    """
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
    return write_published_checkpoint(tmp_path / sub_dir, bundle)


# --- distinct-weights golden (the P1 killer) -----------------------------------------


def test_distinct_checkpoints_load_distinct_evaluators_and_actions(tmp_path):
    """Two checkpoints with known-different weights -> different logits and
    different chosen actions on the same probe state, each its own
    model_version -- the load path actually restored the weights, not just a
    freshly initialized net wearing a borrowed version label (review P1).
    """
    path1 = _write_checkpoint(tmp_path, MICRO, version=1, seed=1, sub_dir="ckpt1")
    path2 = _write_checkpoint(tmp_path, MICRO, version=2, seed=2, sub_dir="ckpt2")

    ev1, mv1 = load_eval_network(path1, MICRO)
    ev2, mv2 = load_eval_network(path2, MICRO)
    assert (mv1, mv2) == (1, 2)

    probe = MICRO.initial_state()
    _, priors1 = ev1(MICRO, probe)
    _, priors2 = ev2(MICRO, probe)
    assert set(priors1) == set(priors2)  # same legal ids: same probe state
    assert priors1 != priors2  # different weights -> different raw logits

    agent1 = NetworkPolicyAgent(ev1, mv1)
    agent2 = NetworkPolicyAgent(ev2, mv2)
    assert agent1.name == "rung5-v1-1"
    assert agent2.name == "rung5-v1-2"
    # Pinned via an independent seed search: seed=1 opens with action 6,
    # seed=2 with action 115, on this exact tiny architecture/probe state.
    assert agent1.select_action(MICRO, probe) != agent2.select_action(MICRO, probe)


# --- rung-5 argmax golden, incl. lowest-id tie-break ---------------------------------


def test_select_action_matches_hand_computed_masked_softmax_argmax_with_tie_break():
    logits = {5: 1.0, 2: 3.0, 9: 3.0, 7: -1.0, 0: 2.9999}

    def stub_evaluator(game, state):
        del game, state
        return 0.0, dict(logits)

    agent = NetworkPolicyAgent(stub_evaluator, model_version=42)
    assert agent.name == "rung5-v1-42"

    # Hand-computed masked softmax over exactly these ids -- argmax of a
    # softmax equals argmax of its logits (monotonic), verified here by
    # actually computing the softmax rather than assuming the equivalence.
    peak = max(logits.values())
    unnormalized = {a: math.exp(v - peak) for a, v in logits.items()}
    total = sum(unnormalized.values())
    softmax = {a: v / total for a, v in unnormalized.items()}
    expected = max(sorted(softmax), key=softmax.get)  # lowest id among ties
    assert expected == 2  # 2 and 9 tie at the max; 2 < 9

    assert agent.select_action(object(), object()) == expected


# --- determinism, no RNG ---------------------------------------------------------------


def test_network_policy_agent_is_deterministic_and_consumes_no_rng(tmp_path, monkeypatch):
    path = _write_checkpoint(tmp_path, MICRO, version=1, seed=11)
    ev1, mv1 = load_eval_network(path, MICRO)
    ev2, mv2 = load_eval_network(path, MICRO)  # a second, independent load
    assert mv1 == mv2

    def _boom(*args, **kwargs):
        raise AssertionError("NetworkPolicyAgent must consume no RNG")

    monkeypatch.setattr(random, "random", _boom)
    monkeypatch.setattr(random.Random, "__init__", _boom)

    agent1 = NetworkPolicyAgent(ev1, mv1)
    agent2 = NetworkPolicyAgent(ev2, mv2)

    def _play_out(agent):
        state = MICRO.initial_state()
        actions = []
        while not MICRO.is_terminal(state):
            a = agent.select_action(MICRO, state)
            actions.append(a)
            state = MICRO.apply(state, a)
        return actions

    seq1 = _play_out(agent1)
    seq2 = _play_out(agent2)
    assert seq1 == seq2
    assert len(seq1) > 0


# --- tampered fingerprint: one integration case, delegating to the m3 battery --------


def test_load_eval_network_rejects_a_tampered_fingerprint_checkpoint(tmp_path):
    """A checkpoint loaded against the wrong game must fail loudly through
    the eval load path too -- the same tampering pattern
    ``tests/test_checkpoint.py::test_load_checkpoint_fingerprint_mismatch_names_fields_and_applies_nothing``
    uses (a mismatched game, not hand-corrupted bytes), reused here as one
    integration case rather than re-running that whole negative battery.
    """
    ttt = TicTacToe()
    torch.manual_seed(5)
    net = Network(_tiny_network_config(ttt))
    optimizer = make_optimizer(net, lr=1e-2)
    scaler = make_scaler("cpu")
    bundle = build_bundle(
        version=0,
        learner_step=0,
        game=ttt,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    path = write_published_checkpoint(tmp_path, bundle)

    with pytest.raises(FingerprintMismatchError):
        load_eval_network(path, OTHELLO)


# --- end-to-end: mirrored pair through the real runner -------------------------------


def test_rung5_agent_survives_a_mirrored_micro_blokus_pair(tmp_path):
    """A rung-5 agent, built via the intended factory shape, plays a full
    mirrored pair (``play_pairs`` + the Blokus start-square balancer) without
    tripping the evaluator's cross-wiring guard on the second game's
    ``_OpeningRestricted``-wrapped view."""
    path = _write_checkpoint(tmp_path, MICRO, version=3, seed=1)
    factory_rung5 = rung5_agent_factory(path, MICRO)

    results = play_pairs(
        MICRO,
        factory_rung5,
        lambda seed: RandomAgent(seed),
        n_pairs=2,
        seed=3,
        opening_balancer=start_square_balancer,
    )
    assert len(results) == 2
    for pair in results:
        assert pair.score_a + pair.score_b == 2.0
        for rec in pair.games:
            assert sum(rec.utilities) == 0.0
            assert rec.plies >= 1


def test_rung5_agent_factory_loads_once_and_shares_across_calls(tmp_path, monkeypatch):
    """The intended ``AgentFactory`` shape (documented on
    :func:`~core.eval_agents.rung5_agent_factory`): the checkpoint load
    happens once, outside the returned closure; building an agent per game
    must not reload it. Verified black-box by counting calls to
    ``load_eval_network`` itself, not by inspecting agent internals."""
    path = _write_checkpoint(tmp_path, MICRO, version=7, seed=1)
    calls = []
    real_load = eval_agents_module.load_eval_network

    def _counting_load(*args, **kwargs):
        calls.append((args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(eval_agents_module, "load_eval_network", _counting_load)

    factory = eval_agents_module.rung5_agent_factory(path, MICRO)
    assert len(calls) == 1  # loaded once, at factory-build time

    agent_a = factory(seed=0)
    agent_b = factory(seed=999)  # seed is accepted (AgentFactory shape) and unused
    assert len(calls) == 1  # rebuilding the lightweight agent must not reload
    assert agent_a.name == agent_b.name == "rung5-v1-7"
    assert agent_a is not agent_b  # distinct lightweight wrappers
    assert agent_a.select_action(MICRO, MICRO.initial_state()) == agent_b.select_action(
        MICRO, MICRO.initial_state()
    )  # sharing the one loaded evaluator: identical behavior, not just identical name


# --- reflective delegation audit: every Game ABC member -------------------------------


def test_opening_restricted_delegates_every_game_abc_member():
    """Every abstract *and* concrete ``Game`` member must delegate to the
    wrapped game unchanged (outside the deliberate initial-state opening
    filter) -- so a future ABC addition that this test isn't updated for
    fails loudly here (the ``declared == set(checks)`` guard below) instead
    of silently shipping an undelegated member (as ``orientation_table_hash``/
    ``encoding_conventions`` were before this task, since both are concrete
    on the ABC and Python happily inherits a default for an unoverridden
    concrete method -- no ``TypeError`` the way a missed abstract member
    would raise)."""
    inner = MICRO
    wrapper = _OpeningRestricted(inner, lambda a: True)  # accept-all: no filtering effect

    state0 = inner.initial_state()
    a0 = min(inner.legal_moves(state0))
    state1 = inner.apply(state0, a0)  # non-initial, nonterminal: bypasses the filter path
    a1 = min(inner.legal_moves(state1))
    move1 = inner.decode_action(a1)

    terminal = state0
    while not inner.is_terminal(terminal):
        terminal = inner.apply(terminal, min(inner.legal_moves(terminal)))

    def _symmetry_groups_match(wrapped_group, inner_group):
        # (transform, permutation) pairs: the transform is a freshly built
        # closure on every property access (not cached), so two calls never
        # produce `==`-equal callables even when they behave identically --
        # compare permutations directly and transforms by their output on a
        # real encoded state instead of by object identity.
        sample_planes = inner.encode_state(state1)
        if len(wrapped_group) != len(inner_group):
            return False
        for (t_w, perm_w), (t_i, perm_i) in zip(wrapped_group, inner_group, strict=True):
            if tuple(perm_w) != tuple(perm_i):
                return False
            if t_w(sample_planes) != t_i(sample_planes):
                return False
        return True

    checks = {
        # declared capabilities
        "num_players": (lambda g: g.num_players, None),
        "is_stochastic": (lambda g: g.is_stochastic, None),
        "is_perfect_information": (lambda g: g.is_perfect_information, None),
        "symmetry_group": (lambda g: g.symmetry_group, _symmetry_groups_match),
        "value_targets": (lambda g: g.value_targets, None),
        # fingerprint surface
        "orientation_table_hash": (lambda g: g.orientation_table_hash, None),
        "encoding_conventions": (lambda g: g.encoding_conventions, None),
        # core contract
        "initial_state": (lambda g: g.initial_state(), None),
        "current_player": (lambda g: g.current_player(state1), None),
        "legal_moves": (lambda g: list(g.legal_moves(state1)), None),
        "apply": (lambda g: g.apply(state1, a1), None),
        "is_terminal": (lambda g: g.is_terminal(state1), None),
        "terminal_utility": (lambda g: g.terminal_utility(terminal, 0), None),
        "training_targets": (lambda g: g.training_targets(terminal, 0), None),
        # encoding surface
        "encode_state": (lambda g: g.encode_state(state1), None),
        "encode_action": (lambda g: g.encode_action(move1), None),
        "decode_action": (lambda g: g.decode_action(a1), None),
        "policy_shape": (lambda g: g.policy_shape, None),
        "input_planes": (lambda g: g.input_planes, None),
        "input_shape": (lambda g: g.input_shape, None),
    }

    declared = {name for name in vars(Game) if not name.startswith("_")}
    assert declared == set(checks), (
        f"Game ABC members missing from this delegation audit: {declared - set(checks)}; "
        f"stale entries no longer on the ABC: {set(checks) - declared}"
    )

    for name, (call, compare) in checks.items():
        got, want = call(wrapper), call(inner)
        if compare is None:
            assert got == want, name
        else:
            assert compare(got, want), name
