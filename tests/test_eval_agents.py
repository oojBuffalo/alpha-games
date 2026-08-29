"""Checkpoint-backed evaluator load path + rungs 5/6/7 eval agents (§9, M4).

CPU-only, seeded. Covers the spec's Test Strategy for rung 5 (task 2): the
distinct-weights golden (the P1 killer -- two saved checkpoints with
known-different weights must load into evaluators that produce different
logits *and* different chosen actions, each labeled its own
``model_version``); rung-5 argmax against a hand-computed masked-softmax
golden, including the lowest-id tie-break; determinism with no RNG consumed;
one tampered-fingerprint integration case delegating to the m3 checkpoint
battery's own tampering pattern (mismatched game, not a re-test of the whole
battery); an end-to-end mirrored micro-Blokus pair through ``play_pairs`` +
``games.blokus_duo.baselines.start_square_balancer`` with a rung-5 agent; and
the reflective every-``Game``-ABC-member delegation audit for
``core.runner._OpeningRestricted``.

And for rungs 6/7's shared ``SearchAgent`` (task 3): the prior-source
golden (rung 6 uniform vs. rung 7 softmaxed, both consuming the evaluator's
value) and the budget-accounting proof (root edge visits sum to exactly
``sims - 1``) via a white-box ``MCTS``-recording shim -- black-box on
``SearchAgent`` itself, since it exposes no search-object accessor; noiseless
determinism of both move sequences and visit counts; statelessness/binding
(a wrapper game's opening restriction actually bites, and interleaved calls
on unrelated states/games do not cross-contaminate); a protocol assert that
no construction path yields root noise; and a slow-marker sanity match
recovering minimax moves on TTT through the agent seam with a value-perfect
stub evaluator (the M0 oracle pattern, ``tests/test_mcts_minimax.py``).

And for rung 8 (task 4, historical checkpoints as frozen opponents): the
pinned selection rule reproduced on synthetic version lists including every
listed edge (first member, fewer available versions than wanted, never the
candidate itself, the domain rejection of v0/non-members, determinism over a
growing prefix, and that ``k_total`` -- never ``len(versions)``/``max`` -- is
what the lag is computed from); the pre-load fingerprint assert failing the
cell with the checkpoint path and both the stored and live orientation hash
named, before any agent is even constructed; snapshot/``latest`` files never
reaching the selector at all; and the identity-sharing/connected-Elo-graph
golden tying it back to task 3's rung-7 factory and ``core.elo.fit_elo``.
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
from core.checkpoint import (
    build_bundle,
    list_published_versions,
    published_checkpoint_path,
    write_latest_pointer,
    write_published_checkpoint,
    write_resume_snapshot,
)
from core.elo import fit_elo
from core.eval_agents import (
    EVAL_SIMS,
    NetworkPolicyAgent,
    SearchAgent,
    assert_historical_checkpoint_matches_live_game,
    historical_opponent_factory,
    historical_opponents,
    load_eval_network,
    rung5_agent_factory,
    rung_search_agent_factory,
)
from core.game import Game
from core.mcts import MCTS
from core.network import Network, NetworkConfig
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


# --- SearchAgent (rungs 6/7): identity + construction ----------------------------------


def test_search_agent_identity_strings_and_form_validation():
    def stub_evaluator(g, s):
        del g, s
        return 0.0, None

    rung6 = SearchAgent(stub_evaluator, model_version=3, form=6, sims=1)
    rung7 = SearchAgent(stub_evaluator, model_version=3, form=7, sims=1)
    assert rung6.name == "rung6-v1-3"
    assert rung7.name == "rung7-v1-3"

    for bad_form in (0, 5, 8, "6"):
        with pytest.raises(ValueError):
            SearchAgent(stub_evaluator, model_version=1, form=bad_form)


def test_search_agent_defaults_to_the_pinned_eval_sims_budget():
    # EVAL_SIMS is the frozen v1 budget -- the constructor must default to it
    # rather than silently require callers to pass it.
    def stub_evaluator(g, s):
        del g, s
        return 0.0, None

    agent = SearchAgent(stub_evaluator, model_version=1, form=7)
    assert agent._sims == EVAL_SIMS == 512


# --- prior-source golden: rung 6 uniform, rung 7 softmax, both consume value ------------


def test_prior_source_golden_rung6_uniform_rung7_softmax_both_consume_value(monkeypatch):
    state = TTT.initial_state()
    legal = list(TTT.legal_moves(state))
    logits = {a: float(i) for i, a in enumerate(legal)}  # strictly increasing: distinguishable
    calls = []

    def stub_evaluator(g, s):
        calls.append(s)
        return 0.6, dict(logits)

    monkeypatch.setattr(eval_agents_module, "MCTS", _RecordingMCTS)
    _RecordingMCTS.instances.clear()

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


def test_root_edge_visits_sum_to_sims_minus_one(monkeypatch):
    """After one move, the root's edge visits sum to exactly sims - 1: the
    first simulation only expands the root itself (M0 accounting -- no edge
    on its path), so the remaining sims-1 each add exactly one visit to some
    root edge. Verified independently against the same invariant
    tests/test_subtree_reuse.py pins directly on MCTS (``sum(root.N) ==
    n_sims - 1``), so this proves SearchAgent actually ran the pinned budget
    end to end, not merely that the invariant holds on MCTS in isolation."""

    def stub_evaluator(g, s):
        del g, s
        return 0.0, None

    monkeypatch.setattr(eval_agents_module, "MCTS", _RecordingMCTS)
    _RecordingMCTS.instances.clear()

    sims = 37
    agent = SearchAgent(stub_evaluator, model_version=1, form=7, sims=sims)
    agent.select_action(TTT, TTT.initial_state())

    assert len(_RecordingMCTS.instances) == 1
    root = _RecordingMCTS.instances[0].root
    assert sum(root.N) == sims - 1


# --- noiseless determinism --------------------------------------------------------------


def test_noiseless_determinism_identical_sequences_and_visit_counts(tmp_path, monkeypatch):
    path = _write_checkpoint(tmp_path, MICRO, version=1, seed=21)
    monkeypatch.setattr(eval_agents_module, "MCTS", _RecordingMCTS)

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

    def stub_evaluator(g, s):
        del g, s
        return 0.0, None

    state0 = MICRO.initial_state()
    restricted_ids = set(list(MICRO.legal_moves(state0))[:3])
    wrapped = _OpeningRestricted(MICRO, restricted_ids.__contains__)

    agent = SearchAgent(stub_evaluator, model_version=1, form=7, sims=8)
    action = agent.select_action(wrapped, state0)
    assert action in restricted_ids


def test_interleaved_calls_on_unrelated_states_do_not_cross_contaminate():
    """One SearchAgent instance, called on alternating unrelated
    games/states, must return the same move for the same (game, state) every
    time -- no leftover tree, evaluator cache, or other cross-call state."""

    def stub_evaluator(g, s):
        del g, s
        return 0.0, None

    ttt_s0 = TTT.initial_state()
    ttt_s1 = TTT.apply(ttt_s0, min(TTT.legal_moves(ttt_s0)))
    micro_s0 = MICRO.initial_state()

    agent = SearchAgent(stub_evaluator, model_version=1, form=7, sims=8)
    results = {"ttt_s0": set(), "ttt_s1": set(), "micro_s0": set()}
    for _ in range(3):
        results["ttt_s0"].add(agent.select_action(TTT, ttt_s0))
        results["ttt_s1"].add(agent.select_action(TTT, ttt_s1))
        results["micro_s0"].add(agent.select_action(MICRO, micro_s0))

    for key, actions in results.items():
        assert len(actions) == 1, f"{key} was not stable across interleaved calls: {actions}"


# --- protocol assert: no construction path yields root noise ---------------------------


def test_no_construction_path_yields_root_noise_enabled(monkeypatch):
    def stub_evaluator(g, s):
        del s
        return 0.0, {a: float(a) for a in g.legal_moves(TTT.initial_state())}

    monkeypatch.setattr(eval_agents_module, "MCTS", _RecordingMCTS)
    _RecordingMCTS.instances.clear()

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
    assert agent_a.name == agent_b.name == "rung6-v1-4"
    assert agent_a is not agent_b

    factory7 = rung_search_agent_factory(path, MICRO, form=7, sims=4)
    agent7 = factory7(seed=0)
    assert agent7.name == "rung7-v1-4"


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


# =========================================================================================
# Rung 8: historical checkpoints as frozen opponents (task 4)
# =========================================================================================


# --- historical_opponents: the pinned rule + edges -------------------------------------


def test_historical_opponents_reproduces_the_pinned_rule_and_edges():
    # k_total=8 -> lag = ceil(8 / 4) = 2, so the pinned set is {v-1, v-2, 1}.
    versions = tuple(range(1, 11))  # a full 1..10 member list

    assert historical_opponents(versions, candidate=1, k_total=8) == []  # first member: empty
    assert historical_opponents(versions, candidate=2, k_total=8) == [1]  # {1, 0, 1} -> {1}
    assert historical_opponents(versions, candidate=3, k_total=8) == [1, 2]  # {2, 1, 1}
    assert historical_opponents(versions, candidate=5, k_total=8) == [1, 3, 4]  # {4, 3, 1}
    assert historical_opponents(versions, candidate=10, k_total=8) == [1, 8, 9]  # {9, 8, 1}

    for candidate in versions:  # never selects the candidate itself, for every candidate
        assert candidate not in historical_opponents(versions, candidate=candidate, k_total=8)


def test_historical_opponents_reduced_when_fewer_versions_are_available():
    # candidate=5 at k_total=8 wants {4, 3, 1}, but only 1 and 5 are on disk so far.
    assert historical_opponents([1, 5], candidate=5, k_total=8) == [1]
    assert historical_opponents([1, 4, 5], candidate=5, k_total=8) == [1, 4]


def test_historical_opponents_uses_the_explicit_k_total_never_len_or_max_of_versions():
    versions = tuple(range(1, 10))  # len(versions) == 9, max(versions) == 9

    # k_total=40 -> lag = ceil(40 / 4) = 10 -> candidate - lag = 9 - 10 = -1, out of range.
    result = historical_opponents(versions, candidate=9, k_total=40)
    assert result == [1, 8]
    # Had the lag been (wrongly) derived from len(versions)==9, ceil(9/4)=3 would
    # have put 9-3=6 in the set too -- it must never appear.
    assert 6 not in result


def test_historical_opponents_deterministic_over_a_growing_prefix():
    short = tuple(range(1, 10))
    longer = tuple(range(1, 26))  # more checkpoints published later in the run
    result_short = historical_opponents(short, candidate=9, k_total=40)
    result_longer = historical_opponents(longer, candidate=9, k_total=40)
    assert result_short == result_longer == [1, 8]


def test_historical_opponents_rejects_v0_even_when_present_in_versions():
    with pytest.raises(ValueError, match="non-member"):
        historical_opponents([0, 1, 2, 3], candidate=3, k_total=8)


def test_historical_opponents_rejects_a_candidate_absent_from_versions():
    with pytest.raises(ValueError, match="not a member"):
        historical_opponents([1, 2, 3], candidate=7, k_total=8)


def test_historical_opponents_rejects_a_nonpositive_k_total():
    with pytest.raises(ValueError, match="k_total"):
        historical_opponents([1, 2], candidate=2, k_total=0)


def test_snapshot_and_latest_files_are_never_offered_as_historical_opponents(tmp_path):
    """A run dir carrying v0, a rolling ``resume.pt``, and a ``latest``
    pointer alongside members 1-3: ``list_published_versions`` already
    structurally excludes ``resume.pt``/``latest`` (neither matches the
    ``ckpt-<digits>.pt`` glob), and this function's own domain check rejects
    the v0 it *does* return -- so neither ever reaches the returned opponent
    set."""
    ckpt_dir = tmp_path / "run"
    for v in (0, 1, 2, 3):
        _write_checkpoint(tmp_path, MICRO, version=v, seed=v + 1, sub_dir="run")

    torch.manual_seed(99)
    net = Network(_tiny_network_config(MICRO))
    optimizer = make_optimizer(net, lr=1e-2)
    scaler = make_scaler("cpu")
    snapshot_bundle = build_bundle(
        version=3,
        learner_step=0,
        game=MICRO,
        run_config={},
        net=net,
        optimizer=optimizer,
        scaler=scaler,
        metrics={},
    )
    write_resume_snapshot(ckpt_dir, snapshot_bundle)
    write_latest_pointer(ckpt_dir, 3)
    assert (ckpt_dir / "resume.pt").exists()
    assert (ckpt_dir / "latest").exists()

    on_disk = list_published_versions(ckpt_dir)
    assert on_disk == (0, 1, 2, 3)  # v0 is a recorded artifact, present on disk

    with pytest.raises(ValueError, match="non-member"):
        historical_opponents(on_disk, candidate=3, k_total=8)

    # task 9's real job -- excluding v0 before this function ever sees the
    # list -- still leaves resume.pt/latest with no way in: they never
    # produced an integer version to begin with.
    members_only = tuple(v for v in on_disk if v != 0)
    assert historical_opponents(members_only, candidate=3, k_total=8) == [1, 2]


# --- pre-load fingerprint assert: fails the cell before any agent is built -------------


def _tamper_orientation_hash(path):
    """Rewrite a published checkpoint's stored ``orientation_hash`` in place.

    Mirrors ``tests/test_checkpoint.py``'s ``schema_version`` tampering
    pattern (load the raw payload, mutate one field, re-save) -- simulating a
    real historical checkpoint written back when the orientation table hashed
    differently, without touching any other fingerprint field.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fingerprint = dict(payload["fingerprint"])
    fingerprint["orientation_hash"] = "tampered-" + str(fingerprint["orientation_hash"])
    payload["fingerprint"] = fingerprint
    torch.save(payload, path)


def test_assert_historical_checkpoint_matches_live_game_names_path_and_both_hashes(tmp_path):
    path = _write_checkpoint(tmp_path, MICRO, version=1, seed=1)
    live_hash = MICRO.orientation_table_hash
    _tamper_orientation_hash(path)

    with pytest.raises(FingerprintMismatchError) as excinfo:
        assert_historical_checkpoint_matches_live_game(path, MICRO)

    message = str(excinfo.value)
    assert str(path) in message  # the checkpoint path is named
    assert "orientation_hash" in message  # the field that disagreed is named
    assert f"tampered-{live_hash}" in message  # the stored hash
    assert live_hash in message  # the live hash


def test_historical_opponent_factory_fails_the_cell_before_any_agent_is_built(
    tmp_path, monkeypatch
):
    ckpt_dir = tmp_path / "run"
    path = _write_checkpoint(tmp_path, MICRO, version=1, seed=1, sub_dir="run")
    _tamper_orientation_hash(path)

    def _boom(*args, **kwargs):
        raise AssertionError(
            "rung_search_agent_factory must never run once the pre-load assert has failed"
        )

    monkeypatch.setattr(eval_agents_module, "rung_search_agent_factory", _boom)

    with pytest.raises(FingerprintMismatchError):
        eval_agents_module.historical_opponent_factory(ckpt_dir, MICRO, old_version=1)


def test_historical_opponent_factory_succeeds_when_the_fingerprint_matches(tmp_path):
    ckpt_dir = tmp_path / "run"
    _write_checkpoint(tmp_path, MICRO, version=1, seed=1, sub_dir="run")
    factory = historical_opponent_factory(ckpt_dir, MICRO, old_version=1, sims=1)
    agent = factory(seed=0)
    assert agent.name == "rung7-v1-1"


# --- identity-sharing golden + connected Elo graph -------------------------------------


def test_identity_sharing_and_connected_elo_graph_with_random_anchor(tmp_path):
    """A version appearing both as task 3's own candidate and as a rung-8
    historical opponent shares one agent name -- and a small two-checkpoint
    fixture, fed as synthetic match records, yields a connected Elo graph
    through the real ``fit_elo`` anchored on rung 1 (``"random"``, the M1.6
    anchor -- ``core.agents.RandomAgent.name``)."""
    ckpt_dir = tmp_path / "run"
    _write_checkpoint(tmp_path, MICRO, version=1, seed=1, sub_dir="run")
    _write_checkpoint(tmp_path, MICRO, version=2, seed=2, sub_dir="run")

    # Identity-sharing golden: version 1 built via task 3's own rung-7
    # factory vs. built as a rung-8 historical opponent -- same name.
    candidate_factory_v1 = rung_search_agent_factory(
        published_checkpoint_path(ckpt_dir, 1), MICRO, form=7, sims=1
    )
    historical_factory_v1 = historical_opponent_factory(ckpt_dir, MICRO, old_version=1, sims=1)
    name_v1_as_candidate = candidate_factory_v1(seed=0).name
    name_v1_as_opponent = historical_factory_v1(seed=0).name
    assert name_v1_as_candidate == name_v1_as_opponent == "rung7-v1-1"
    assert RandomAgent(seed=0).name == "random"  # the M1.6 anchor id, unchanged

    candidate_factory_v2 = rung_search_agent_factory(
        published_checkpoint_path(ckpt_dir, 2), MICRO, form=7, sims=1
    )
    name_v2 = candidate_factory_v2(seed=0).name
    assert name_v2 == "rung7-v1-2"

    # A small two-checkpoint fixture: candidate v2's rung-8 opponent set is
    # exactly {1} (k_total=2 -> lag=ceil(2/4)=1 -> {1, 1, 1}).
    assert historical_opponents([1, 2], candidate=2, k_total=2) == [1]

    # Synthetic match records (§ Test Strategy: "feed synthetic match
    # records") connecting v2 to the anchor and to its rung-8 opponent v1 --
    # the whole three-node graph must fit without a connectivity error.
    matches = [
        ("random", name_v2, 1.5, 3),
        (name_v2, name_v1_as_opponent, 2.0, 3),
    ]
    ratings = fit_elo(matches, anchor="random")
    assert set(ratings) == {"random", name_v2, "rung7-v1-1"}
    assert ratings["random"] == 0.0
