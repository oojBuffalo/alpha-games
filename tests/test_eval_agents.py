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
the runner delegation audit lives in ``tests/test_runner.py``.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

import core.eval_agents as eval_agents_module
from core import RandomAgent
from core.artifact_fingerprint import FingerprintMismatchError
from core.checkpoint import CheckpointFormatError, build_bundle, write_published_checkpoint
from core.eval_agents import (
    NetworkPolicyAgent,
    load_eval_network,
    rung5_agent_factory,
)
from core.network import Network, NetworkConfig, make_network_evaluator
from core.runner import play_pairs
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
    restores the recorded config rather than
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


def _write_checkpoint(tmp_path, game, *, version, seed, sub_dir="ckpt", return_source=False):
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
        The published checkpoint's path, plus a source evaluator when requested.
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
    path = write_published_checkpoint(tmp_path / sub_dir, bundle)
    return (path, make_network_evaluator(net, game)) if return_source else path


# --- distinct-weights golden (the P1 killer) -----------------------------------------


def test_distinct_checkpoints_load_distinct_evaluators_and_actions(tmp_path):
    """Two checkpoints with known-different weights -> different logits and
    different chosen actions on the same probe state, each its own
    model_version -- the load path actually restored the weights, not just a
    freshly initialized net wearing a borrowed version label (review P1).
    """
    path1, source1 = _write_checkpoint(
        tmp_path, MICRO, version=1, seed=1, sub_dir="ckpt1", return_source=True
    )
    path2, source2 = _write_checkpoint(
        tmp_path, MICRO, version=2, seed=2, sub_dir="ckpt2", return_source=True
    )

    ev1, mv1 = load_eval_network(path1, MICRO)
    ev2, mv2 = load_eval_network(path2, MICRO)
    assert (mv1, mv2) == (1, 2)

    probe = MICRO.initial_state()
    value1, priors1 = ev1(MICRO, probe)
    value2, priors2 = ev2(MICRO, probe)
    expected1, reference1 = source1(MICRO, probe)
    expected2, reference2 = source2(MICRO, probe)
    assert value1 == pytest.approx(expected1)
    assert value2 == pytest.approx(expected2)
    assert priors1 == pytest.approx(reference1)
    assert priors2 == pytest.approx(reference2)
    assert set(priors1) == set(priors2)  # same legal ids: same probe state
    assert priors1 != priors2  # different weights -> different raw logits

    agent1 = NetworkPolicyAgent(ev1, mv1)
    agent2 = NetworkPolicyAgent(ev2, mv2)
    assert agent1.name == "rung5-v1-1"
    assert agent2.name == "rung5-v1-2"
    # Pinned via an independent seed search: seed=1 opens with action 6,
    # seed=2 with action 115, on this exact tiny architecture/probe state.
    assert agent1.select_action(MICRO, probe) == 6
    assert agent2.select_action(MICRO, probe) == 115


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
    path = _write_checkpoint(tmp_path, ttt, version=0, seed=5)

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


@pytest.mark.parametrize("prefix", ["stem.0.", "blocks.0.", "aux_"])
def test_load_eval_network_rejects_missing_components(tmp_path, prefix):
    path = _write_checkpoint(tmp_path, MICRO, version=1, seed=1)
    payload = torch.load(path, weights_only=True)
    original = payload["model_state_dict"]
    payload["model_state_dict"] = {k: v for k, v in original.items() if not k.startswith(prefix)}
    assert len(payload["model_state_dict"]) < len(original)
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_eval_network(path, MICRO)


@pytest.mark.parametrize("filename", ["resume.pt", "ckpt-1.pt"])
def test_load_eval_network_rejects_resume_provenance_even_when_renamed(tmp_path, filename):
    path = _write_checkpoint(tmp_path, MICRO, version=1, seed=1)
    from core.checkpoint import load_checkpoint, write_resume_snapshot

    snapshot = write_resume_snapshot(tmp_path / "snapshot", load_checkpoint(path, MICRO))
    renamed = snapshot.with_name(filename)
    snapshot.rename(renamed)
    with pytest.raises(CheckpointFormatError, match="published checkpoint"):
        load_eval_network(renamed, MICRO)


@pytest.mark.parametrize("config", [{}, {"input_planes": "4"}])
def test_load_eval_network_rejects_malformed_architecture(tmp_path, config):
    path = _write_checkpoint(tmp_path, MICRO, version=1, seed=1)
    payload = torch.load(path, weights_only=True)
    payload["network_config"] = config
    torch.save(payload, path)
    with pytest.raises(CheckpointFormatError, match="network_config"):
        load_eval_network(path, MICRO)


def test_checkpoint_records_architecture_independently_of_weights(tmp_path):
    from core.checkpoint import CHECKPOINT_SCHEMA_VERSION, load_checkpoint

    path = _write_checkpoint(tmp_path, MICRO, version=1, seed=1)
    bundle = load_checkpoint(path, MICRO)
    assert bundle.schema_version == CHECKPOINT_SCHEMA_VERSION == 2
    assert bundle.network_config == _tiny_network_config(MICRO)
    assert bundle.artifact_kind == "published"
    payload = torch.load(path, weights_only=True)
    assert isinstance(payload["network_config"], dict)
