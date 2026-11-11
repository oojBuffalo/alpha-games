"""Network→MCTS evaluator bridge battery (§12 M0/M2).

The bridge's three silent-corruption hazards, asserted end-to-end: the
mover-relative value convention reaching root ``Q`` through the player-aware
backup, the action-id↔logit-index correspondence (the legal ids *are* the
logit indices — re-asserted here against ``MCTS._priors``' output, not just
task 7's network-side flatten golden), and single-softmax ownership (the
bridge returns raw logits; the seam performs the one legal-subset softmax —
the priors golden falsifies exactly the double-softmax bug, since
``softmax(softmax(x)) != softmax(x)``). Plus the two integration proofs:
a seeded random-init ``from_game`` net drives Blokus from the opening, and
the same bridge drives Othello with zero ``core/`` changes. CPU-only, seeded,
small sim counts.
"""

from __future__ import annotations

import math

import pytest
import torch

from core import MCTS
from core.network import (
    Network,
    NetworkConfig,
    make_batched_network_evaluator,
    make_network_evaluator,
)
from core.runner import _OpeningRestricted
from games.blokus_duo import BlokusDuo
from games.othello import Othello

torch.manual_seed(0)

BLOKUS = BlokusDuo()
BLOKUS_NET = Network(NetworkConfig.from_game(BLOKUS)).eval()
OTHELLO = Othello()
OTHELLO_NET = Network(NetworkConfig.from_game(OTHELLO)).eval()


def softmax(logits):
    """Max-subtracted softmax over a list of floats (the seam's own recipe)."""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


# --- integration: the bridge drives search --------------------------------------------


def test_bridge_drives_blokus_mcts_from_the_opening():
    m = MCTS(BLOKUS, evaluate=make_network_evaluator(BLOKUS_NET, BLOKUS))
    s0 = BLOKUS.initial_state()
    m.run(30, root_state=s0)
    assert m.best_action() in list(BLOKUS.legal_moves(s0))


def test_bridge_drives_othello_mcts_across_moves():
    # Same bridge, different adapter (flat 65-head, no aux) — nothing in the
    # bridge or in core/ changes; §12 M1.5's zero-core-diff guarantee extended
    # to the network path.
    m = MCTS(OTHELLO, evaluate=make_network_evaluator(OTHELLO_NET, OTHELLO))
    s = OTHELLO.initial_state()
    m.run(24, root_state=s)
    for _ in range(3):
        a = m.best_action()
        assert a in list(OTHELLO.legal_moves(s))
        s = OTHELLO.apply(s, a)
        m.advance(a)
        m.run(24)


# --- priors golden: single softmax, id↔index correspondence ---------------------------


def test_root_priors_equal_single_legal_subset_softmax():
    # Root P (as produced by MCTS._priors from the bridge's returned logits)
    # must equal a directly-computed legal-subset softmax of the net's flat
    # logits, indexed by action id. Random-init logits are non-constant, so
    # softmax(softmax(x)) != softmax(x) here (guard below): matching the
    # single softmax falsifies the double-softmax bug outright.
    ev = make_network_evaluator(BLOKUS_NET, BLOKUS)
    s0 = BLOKUS.initial_state()
    root = MCTS(BLOKUS, evaluate=ev).run(1, root_state=s0)

    x = torch.tensor(BLOKUS.encode_state(s0), dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        logits, _, _ = BLOKUS_NET(x)
    legal = list(BLOKUS.legal_moves(s0))
    expected = dict(zip(legal, softmax([float(logits[0, a]) for a in legal]), strict=True))

    got = dict(zip(root.actions, root.P, strict=True))
    assert set(got) == set(expected) == set(legal)  # every legal id, exactly once
    assert sum(got.values()) == pytest.approx(1.0, abs=1e-9)
    for a in legal:
        assert got[a] == pytest.approx(expected[a], rel=1e-9), a

    # Guard: the golden discriminates — a double softmax lands measurably away
    # from the single softmax on these logits.
    double = softmax(list(expected.values()))
    assert max(abs(d - e) for d, e in zip(double, expected.values(), strict=True)) > 1e-6


def test_bridge_returns_raw_logits_for_every_legal_id():
    # Seam contract, asserted on the bridge's own output: {action_id: raw
    # logit} covering every legal id (never a distribution — _priors owns the
    # softmax and defaults missing ids to logit 0.0).
    ev = make_network_evaluator(OTHELLO_NET, OTHELLO)
    s0 = OTHELLO.initial_state()
    value, priors = ev(OTHELLO, s0)
    x = torch.tensor(OTHELLO.encode_state(s0), dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        logits, direct_value, _ = OTHELLO_NET(x)
    assert set(priors) == set(OTHELLO.legal_moves(s0))
    for a, p in priors.items():
        assert p == pytest.approx(float(logits[0, a]), rel=1e-9)
    assert -1.0 <= value <= 1.0
    assert value == pytest.approx(float(direct_value[0]), rel=1e-9)
    # Raw logits are unnormalized: on a random-init net the legal subset does
    # not sum to 1 (a pre-softmaxed bridge would make this ≈ 1).
    assert sum(priors.values()) != pytest.approx(1.0, abs=1e-6)


# --- value flows through the player-aware backup --------------------------------------


def test_network_value_reaches_root_q():
    # M0's no-evaluator default backs up 0.0 from every nonterminal leaf, so
    # early-game root Q is identically zero; the bridge's tanh value must make
    # it nonzero (nothing terminal is reachable in 40 sims from the start).
    s0 = OTHELLO.initial_state()
    m0 = MCTS(OTHELLO)
    m0.run(40, root_state=s0)
    assert all(q == 0.0 for q in m0.action_values().values())

    m = MCTS(OTHELLO, evaluate=make_network_evaluator(OTHELLO_NET, OTHELLO))
    m.run(40, root_state=s0)
    assert any(q != 0.0 for q in m.action_values().values())


def test_uniform_prior_flag_keeps_network_value():
    # Ladder rung 6: uniform_prior=True discards the bridge's priors (uniform
    # over legal) but keeps its value — flag semantics pinned at M0.
    m = MCTS(OTHELLO, evaluate=make_network_evaluator(OTHELLO_NET, OTHELLO), uniform_prior=True)
    root = m.run(40, root_state=OTHELLO.initial_state())
    n = len(root.actions)
    assert root.P == [1.0 / n] * n
    assert any(q != 0.0 for q in m.action_values().values())


# --- loud pairing validation ----------------------------------------------------------


def test_bridge_rejects_mismatched_net_game_pairing():
    # A net built for one game must not silently index another game's ids.
    with pytest.raises(ValueError):
        make_network_evaluator(OTHELLO_NET, BLOKUS)
    with pytest.raises(ValueError):
        make_network_evaluator(BLOKUS_NET, OTHELLO)


def test_evaluator_rejects_cross_wired_callback_game():
    # The factory validates its own (net, game) pairing, but the evaluator is
    # an opaque callable — MCTS(game_b, evaluate=bridge(net_a, game_a))
    # constructs without complaint. The callback-time guard makes the first
    # search call the loud failure, not a shape blowup (or, for equal-shaped
    # adapters, silent semantic mixing) somewhere inside the net.
    ev = make_network_evaluator(OTHELLO_NET, OTHELLO)
    with pytest.raises(ValueError, match="cross-wired"):
        ev(BLOKUS, BLOKUS.initial_state())
    with pytest.raises(ValueError, match="cross-wired"):
        MCTS(BLOKUS, evaluate=ev).run(1, root_state=BLOKUS.initial_state())


class _SurfacePreservingWrapper:
    """Delegating stand-in for the runner's opening-restriction wrapper.

    Same declared encoding surface and legal-move filtering as the wrapped
    adapter, but a different identity — and a booby-trapped ``encode_state``
    proving the bridge encodes with the factory-validated adapter, never the
    callback-time game.
    """

    def __init__(self, inner, keep):
        self._inner = inner
        self._keep = keep

    @property
    def input_planes(self):
        return self._inner.input_planes

    @property
    def input_shape(self):
        return self._inner.input_shape

    @property
    def policy_shape(self):
        return self._inner.policy_shape

    def legal_moves(self, state):
        return [a for a in self._inner.legal_moves(state) if self._keep(a)]

    def encode_state(self, state):
        raise AssertionError("bridge must encode with the validated adapter")


def test_evaluator_accepts_surface_preserving_wrapper():
    # Runner-style delegating wrappers (same surface, different identity) pass
    # the cross-wiring guard; legal ids come from the wrapper, encoding from
    # the validated adapter (the pairing the net was checked against).
    s0 = OTHELLO.initial_state()
    full = list(OTHELLO.legal_moves(s0))
    restricted = full[: len(full) // 2]
    wrapper = _SurfacePreservingWrapper(OTHELLO, keep=set(restricted).__contains__)

    ev = make_network_evaluator(OTHELLO_NET, OTHELLO)
    value, priors = ev(wrapper, s0)
    assert set(priors) == set(restricted)
    _, unrestricted = ev(OTHELLO, s0)
    for a in restricted:
        assert priors[a] == pytest.approx(unrestricted[a], rel=1e-9)
    assert -1.0 <= value <= 1.0


def test_evaluator_accepts_the_runner_opening_restriction_wrapper():
    # The named integration path (§9 pairing), pinned against the runner's
    # *real* wrapper rather than a test stand-in: a network-backed agent
    # handed game 2's opening-restricted view must search it, not crash on
    # the cross-wiring guard.
    s0 = OTHELLO.initial_state()
    keep = set(list(OTHELLO.legal_moves(s0))[:2])
    wrapped = _OpeningRestricted(OTHELLO, keep.__contains__)

    ev = make_network_evaluator(OTHELLO_NET, OTHELLO)
    _, priors = ev(wrapped, s0)
    assert set(priors) == keep

    m = MCTS(wrapped, evaluate=ev)
    m.run(8, root_state=s0)
    assert m.best_action() in keep


# --- the batched bridge (§12 M5): one forward per leaf batch --------------------------


def _playout_states(game, plies):
    """Return the states along a fixed first-legal-move playout, opening included."""
    state = game.initial_state()
    states = [state]
    for _ in range(plies):
        state = game.apply(state, next(iter(game.legal_moves(state))))
        states.append(state)
    return states


@pytest.mark.parametrize(
    ("game", "net"), [(OTHELLO, OTHELLO_NET), (BLOKUS, BLOKUS_NET)], ids=["othello", "blokus"]
)
def test_batched_forward_matches_per_leaf_forwards(game, net):
    # CPU, seeded weights, no autocast on either side: one stacked forward must
    # reproduce the per-leaf forwards, row for row, in input order.
    states = _playout_states(game, 4)
    states.append(states[1])  # equal states in one batch: results are positional
    single = make_network_evaluator(net, game)
    batched = make_batched_network_evaluator(net, game)

    results = batched(game, states)

    assert len(results) == len(states)
    for state, (value, priors) in zip(states, results, strict=True):
        ref_value, ref_priors = single(game, state)
        assert isinstance(value, float)
        assert list(priors) == list(game.legal_moves(state))  # every legal id, nothing else
        assert value == pytest.approx(ref_value, abs=1e-5)
        for action, logit in priors.items():
            assert logit == pytest.approx(ref_priors[action], abs=1e-5)
    # The repeated state is answered twice, once per position.
    assert results[-1][0] == pytest.approx(results[1][0], abs=1e-6)
    assert results[-1][1] == pytest.approx(results[1][1], abs=1e-6)


def test_batched_evaluator_puts_the_net_in_eval_mode():
    # In train mode BatchNorm would normalize a leaf by its batch-mates, so a
    # row's answer would depend on what else was collected with it.
    net = Network(NetworkConfig.from_game(OTHELLO)).train()
    batched = make_batched_network_evaluator(net, OTHELLO)
    assert net.training is False
    states = _playout_states(OTHELLO, 2)
    alone = batched(OTHELLO, states[:1])[0]
    together = batched(OTHELLO, states)[0]
    assert together[0] == pytest.approx(alone[0], abs=1e-5)


def test_batched_evaluator_returns_nothing_for_an_empty_batch_without_a_forward():
    class _Unreachable(Network):
        def forward(self, x):
            raise AssertionError("an empty batch must not reach the net")

    batched = make_batched_network_evaluator(
        _Unreachable(NetworkConfig.from_game(OTHELLO)), OTHELLO
    )
    assert batched(OTHELLO, []) == []


def test_batched_bridge_rejects_mismatched_pairing_and_cross_wiring():
    with pytest.raises(ValueError, match="does not match"):
        make_batched_network_evaluator(OTHELLO_NET, BLOKUS)
    batched = make_batched_network_evaluator(OTHELLO_NET, OTHELLO)
    with pytest.raises(ValueError, match="cross-wired"):
        batched(BLOKUS, [BLOKUS.initial_state()])


def test_batched_evaluator_accepts_surface_preserving_wrapper():
    # Legal ids from the callback-time game, encoding from the validated adapter.
    s0 = OTHELLO.initial_state()
    restricted = list(OTHELLO.legal_moves(s0))[:2]
    wrapper = _SurfacePreservingWrapper(OTHELLO, keep=set(restricted).__contains__)
    ((_, priors),) = make_batched_network_evaluator(OTHELLO_NET, OTHELLO)(wrapper, [s0])
    assert list(priors) == restricted


def test_batched_bridge_drives_batched_mcts():
    calls = []
    batched = make_batched_network_evaluator(OTHELLO_NET, OTHELLO)

    def spy(game, states):
        calls.append(len(states))
        return batched(game, states)

    def scalar_must_not_run(game, state):
        raise AssertionError("evaluate_batch supplied: the scalar evaluator is not used")

    m = MCTS(OTHELLO, evaluate=scalar_must_not_run, evaluate_batch=spy)
    root = m.run(64, root_state=OTHELLO.initial_state(), batch_size=8)

    assert sum(root.N) == 63  # the first simulation expands the root
    assert max(calls) > 1  # several leaves really shared one forward
    assert sum(calls) <= 64 and all(1 <= n <= 8 for n in calls)
    assert all(v == 0 for v in root.vloss)
