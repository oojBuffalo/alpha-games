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

import inspect
import math
import random

import pytest
import torch

import core.eval_agents as eval_agents_module
from core import RandomAgent
from core.artifact_fingerprint import FingerprintMismatchError
from core.checkpoint import CheckpointFormatError, build_bundle, write_published_checkpoint
from core.eval_agents import (
    EVAL_SIMS,
    NetworkPolicyAgent,
    SearchAgent,
    SearchForm,
    load_eval_network,
    rung5_agent_factory,
    rung_search_agent_factory,
)
from core.mcts import MCTS
from core.network import Network, NetworkConfig, make_network_evaluator
from core.runner import _OpeningRestricted, play_pairs
from core.train import make_optimizer, make_scaler
from games.blokus_duo import BlokusDuo
from games.blokus_duo.baselines import start_square_balancer
from games.blokus_duo.config import MICRO_CONFIG
from games.othello import Othello
from games.tictactoe import TicTacToe
from tests.reference.minimax import optimal_values, reachable_states

MICRO = BlokusDuo(config=MICRO_CONFIG)
OTHELLO = Othello()
TTT = TicTacToe()


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


class _RecordingMCTS(MCTS):
    """A real ``MCTS``, unmodified, that records every constructed instance.

    Lets a test white-box-inspect the search a :class:`SearchAgent` built
    internally (``select_action`` never returns its search object) without
    duplicating any of ``SearchAgent``'s own construction logic: monkeypatch
    ``core.eval_agents.MCTS`` to this class, call the agent normally, then
    read ``_RecordingMCTS.instances[-1]``.
    """

    instances: list[_RecordingMCTS] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _RecordingMCTS.instances.append(self)


def _zero_evaluator(game, state):
    del game, state
    return 0.0, None


@pytest.fixture
def recording_mcts(monkeypatch):
    _RecordingMCTS.instances.clear()
    monkeypatch.setattr(eval_agents_module, "MCTS", _RecordingMCTS)
    yield _RecordingMCTS.instances
    _RecordingMCTS.instances.clear()


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


# --- SearchAgent (rungs 6/7): identity + construction ----------------------------------


def test_search_agent_identity_strings_and_form_validation():
    rung6 = SearchAgent(_zero_evaluator, model_version=3, form=6)
    rung7 = SearchAgent(_zero_evaluator, model_version=3, form=7)
    assert rung6.name == "rung6-v1-3"
    assert rung7.name == "rung7-v1-3"

    for bad_form in (0, 5, 8, "6", 6.0, 7.0, True, None):
        with pytest.raises(ValueError):
            SearchAgent(_zero_evaluator, model_version=1, form=bad_form)


def test_search_agent_defaults_to_the_pinned_eval_sims_budget():
    # EVAL_SIMS is the frozen v1 budget -- the constructor must default to it
    # rather than silently require callers to pass it.
    agent = SearchAgent(_zero_evaluator, model_version=1, form=7)
    assert agent._sims == EVAL_SIMS == 512


@pytest.mark.parametrize("form", [6, 7, SearchForm.UNIFORM_VALUE, SearchForm.POLICY_VALUE])
@pytest.mark.parametrize("sims", [2, 4, 37, 64, 512])
def test_search_budget_is_part_of_identity(form, sims):
    agent = SearchAgent(_zero_evaluator, model_version=9, form=form, sims=sims)
    rung = SearchForm.parse(form).value
    budget = "" if sims == 512 else f"-s{sims}"
    assert agent.name == f"rung{rung}-v1{budget}-9"


@pytest.mark.parametrize("sims", [-1, 0, 1, 2.0, True, "512", None])
def test_search_rejects_invalid_budgets_before_loading(sims, monkeypatch):
    with pytest.raises(ValueError, match="sims"):
        SearchAgent(_zero_evaluator, 1, form=7, sims=sims)

    def unexpected_load(*args):
        pytest.fail("invalid search configuration must fail before checkpoint IO")

    monkeypatch.setattr(eval_agents_module, "load_eval_network", unexpected_load)
    with pytest.raises(ValueError, match="sims"):
        rung_search_agent_factory("unused.pt", TTT, 7, sims=sims)


@pytest.mark.parametrize("form", [6.0, 7.0, "6", True, 8])
def test_search_factory_rejects_invalid_forms_before_loading(form):
    with pytest.raises(ValueError, match="form"):
        rung_search_agent_factory("unused.pt", TTT, form)


# --- prior-source golden: rung 6 uniform, rung 7 softmax, both consume value ------------


def test_prior_source_golden_rung6_uniform_rung7_softmax_both_consume_value(recording_mcts):
    state = TTT.initial_state()
    legal = list(TTT.legal_moves(state))
    logits = {a: float(i) for i, a in enumerate(legal)}  # strictly increasing: distinguishable
    calls = []

    def stub_evaluator(g, s):
        calls.append(s)
        return 0.6, dict(logits)

    agent6 = SearchAgent(stub_evaluator, model_version=1, form=6, sims=5)
    agent7 = SearchAgent(stub_evaluator, model_version=1, form=7, sims=5)
    a6 = agent6.select_action(TTT, state)
    a7 = agent7.select_action(TTT, state)
    assert a6 in legal
    assert a7 in legal

    assert len(_RecordingMCTS.instances) == 2
    m6, m7 = _RecordingMCTS.instances
    assert m6.uniform_prior is True
    assert m7.uniform_prior is False

    n = len(legal)
    assert m6.root.P == [1.0 / n] * n

    peak = max(logits.values())
    exps = {a: math.exp(v - peak) for a, v in logits.items()}
    total = sum(exps.values())
    expected_softmax = [exps[a] / total for a in m7.root.actions]
    for got, want in zip(m7.root.P, expected_softmax, strict=True):
        assert got == pytest.approx(want, rel=1e-9)
    assert m6.root.P != m7.root.P  # the two forms actually consult different prior sources

    # Both consumed the evaluator's *value*: with a non-terminal, non-constant
    # position this shallow, every one of the 5 simulations per tree performs
    # exactly one fresh expansion (call), and any edge on the traversed path
    # backs up the (nonzero) evaluator value -- so some root Q must be nonzero.
    assert len(calls) == 10
    assert any(q != 0.0 for q in m6.root.Q)
    assert any(q != 0.0 for q in m7.root.Q)


# --- budget accounting (the P2 regression) ----------------------------------------------


@pytest.mark.parametrize("sims", [2, 37, EVAL_SIMS])
def test_root_edge_visits_sum_to_sims_minus_one(recording_mcts, sims):
    """After one move, the root's edge visits sum to exactly sims - 1: the
    first simulation only expands the root itself (M0 accounting -- no edge
    on its path), so the remaining sims-1 each add exactly one visit to some
    root edge. Verified independently against the same invariant
    tests/test_subtree_reuse.py pins directly on MCTS (``sum(root.N) ==
    n_sims - 1``), so this proves SearchAgent actually ran the pinned budget
    end to end, not merely that the invariant holds on MCTS in isolation."""

    agent = SearchAgent(_zero_evaluator, model_version=1, form=7, sims=sims)
    agent.select_action(TTT, TTT.initial_state())

    assert len(_RecordingMCTS.instances) == 1
    root = _RecordingMCTS.instances[0].root
    assert sum(root.N) == sims - 1


# --- noiseless determinism --------------------------------------------------------------


def test_noiseless_determinism_identical_sequences_and_visit_counts(tmp_path, recording_mcts):
    path = _write_checkpoint(tmp_path, MICRO, version=1, seed=21)

    def _play_out(agent):
        _RecordingMCTS.instances.clear()
        state = MICRO.initial_state()
        actions = []
        visit_snapshots = []
        while not MICRO.is_terminal(state):
            a = agent.select_action(MICRO, state)
            actions.append(a)
            visit_snapshots.append(_RecordingMCTS.instances[-1].action_visit_counts())
            state = MICRO.apply(state, a)
        return actions, visit_snapshots

    ev1, mv1 = load_eval_network(path, MICRO)
    seq1, visits1 = _play_out(SearchAgent(ev1, mv1, form=7, sims=16))

    ev2, mv2 = load_eval_network(path, MICRO)  # a second, independent load of the same weights
    seq2, visits2 = _play_out(SearchAgent(ev2, mv2, form=7, sims=16))

    assert seq1 == seq2
    assert visits1 == visits2
    assert len(seq1) > 0


# --- statelessness / binding -------------------------------------------------------------


def test_select_action_binds_to_the_passed_game_not_a_captured_one():
    """A wrapper game restricting the opening, passed straight to
    select_action, must have its restriction actually bite -- proving the
    search is constructed fresh from the ``game`` argument received on that
    call, never a game captured at construction (SearchAgent is built with no
    game at all)."""

    state0 = MICRO.initial_state()
    restricted_ids = set(list(MICRO.legal_moves(state0))[:3])
    wrapped = _OpeningRestricted(MICRO, restricted_ids.__contains__)

    agent = SearchAgent(_zero_evaluator, model_version=1, form=7, sims=8)
    action = agent.select_action(wrapped, state0)
    assert action in restricted_ids


def test_interleaved_calls_on_unrelated_states_do_not_cross_contaminate():
    """One SearchAgent instance, called on alternating unrelated
    games/states, must return the same move for the same (game, state) every
    time -- no leftover tree, evaluator cache, or other cross-call state."""

    ttt_s0 = TTT.initial_state()
    ttt_s1 = TTT.apply(ttt_s0, min(TTT.legal_moves(ttt_s0)))
    micro_s0 = MICRO.initial_state()

    agent = SearchAgent(_zero_evaluator, model_version=1, form=7, sims=8)
    results = {"ttt_s0": set(), "ttt_s1": set(), "micro_s0": set()}
    for _ in range(3):
        results["ttt_s0"].add(agent.select_action(TTT, ttt_s0))
        results["ttt_s1"].add(agent.select_action(TTT, ttt_s1))
        results["micro_s0"].add(agent.select_action(MICRO, micro_s0))

    for key, actions in results.items():
        assert len(actions) == 1, f"{key} was not stable across interleaved calls: {actions}"


# --- protocol assert: no construction path yields root noise ---------------------------


def test_no_construction_path_yields_root_noise_enabled(recording_mcts):
    def stub_evaluator(g, s):
        del s
        return 0.0, {a: float(a) for a in g.legal_moves(TTT.initial_state())}

    for form in (6, 7):
        SearchAgent(stub_evaluator, model_version=1, form=form, sims=4).select_action(
            TTT, TTT.initial_state()
        )

    assert len(_RecordingMCTS.instances) == 2
    assert all(m.root_noise is None for m in _RecordingMCTS.instances)

    # The constructor exposes no parameter through which a caller could ever
    # request root noise in the first place.
    params = inspect.signature(SearchAgent.__init__).parameters
    assert "root_noise" not in params


# --- factory helper, parallel to rung5_agent_factory ------------------------------------


def test_rung_search_agent_factory_loads_once_and_builds_the_requested_form(tmp_path, monkeypatch):
    path = _write_checkpoint(tmp_path, MICRO, version=4, seed=1)
    calls = []
    real_load = eval_agents_module.load_eval_network

    def _counting_load(*args, **kwargs):
        calls.append((args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(eval_agents_module, "load_eval_network", _counting_load)

    factory6 = rung_search_agent_factory(path, MICRO, form=6, sims=4)
    assert len(calls) == 1  # loaded once, at factory-build time

    agent_a = factory6(seed=0)
    agent_b = factory6(seed=999)  # seed accepted (AgentFactory shape), unused
    assert len(calls) == 1  # building an agent per game must not reload
    assert agent_a.name == agent_b.name == "rung6-v1-s4-4"
    assert agent_a is not agent_b

    factory7 = rung_search_agent_factory(path, MICRO, form=7, sims=4)
    agent7 = factory7(seed=0)
    assert agent7.name == "rung7-v1-s4-4"
    assert rung_search_agent_factory(path, MICRO, form=7)(0).name == "rung7-v1-4"


# --- slow-marker sanity: recovers minimax moves through the agent seam -----------------


@pytest.mark.slow
def test_rung7_agent_recovers_minimax_moves_with_a_value_perfect_evaluator():
    """The M0 oracle pattern (tests/test_mcts_minimax.py), driven through the
    agent seam instead of MCTS directly: a stub evaluator returning the exact
    solved value at every leaf (uniform priors -- raw is always None) turns
    SearchAgent(form=7)'s search into a value-guided lookup. Every TTT
    position with <= 3 plies remaining must yield a move preserving the
    mover's game-theoretic value, proving MCTS.best_action() is wired
    end-to-end through select_action -- not only when MCTS is driven
    directly, as the existing oracle battery does."""
    value_cache: dict = {}

    def stub_evaluator(g, s):
        return optimal_values(g, s, value_cache)[g.current_player(s)], None

    agent = SearchAgent(stub_evaluator, model_version=1, form=7, sims=40)

    tested = 0
    for state in reachable_states(TTT):
        if TTT.is_terminal(state) or state[0].count(-1) > 3:
            continue
        mover = TTT.current_player(state)
        target = optimal_values(TTT, state, value_cache)[mover]
        action = agent.select_action(TTT, state)
        achieved = optimal_values(TTT, TTT.apply(state, action), value_cache)[mover]
        assert achieved >= target - 1e-9, (
            f"SearchAgent blundered on {state}: chose {action} "
            f"(value {achieved}) < optimal {target}"
        )
        tested += 1
    assert tested > 100  # sanity: many distinct endgames actually exercised
