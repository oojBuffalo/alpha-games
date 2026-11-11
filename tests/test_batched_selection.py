"""Virtual-loss / in-flight-leaf correctness battery (design doc §6.2, §12 M5, §13).

Batched selection holds virtual loss on every in-flight path and backs it out again; a
residue or a miscounted visit distorts ``N``, the D10 training target. The properties
here are asserted over seeded random positions, on alternating games and on scripted
games with consecutive movers, at several batch sizes, through the real ``run`` entry
point:

  * Visit accounting is *conditional* conservation: a simulation that expands a node
    traverses none of its edges, and one that ends at a terminal node has no edge to
    traverse. The equations are therefore written per node state, as deltas from a
    snapshot taken before the run — there is no single ``ΣN == budget`` identity.
  * Virtual loss is exact while a batch is in flight (each edge holds one unit per
    in-flight leaf beneath it) and zero once ``run`` returns.
  * A batch's leaves are distinct *nodes*, and no node is evaluated twice. Equal states
    may share a batch: there is no transposition table.
  * The first collision closes the batch. So a batch need not fill even while unevaluated
    leaves remain, and "``K`` distinct leaves whenever ``K`` exist" is deliberately not
    asserted — it is false under that rule.
  * Backup is player-aware on every edge, whatever order a batch's leaves land in.
  * ``batch_size = 1`` is the serial search: bit-identical to an independent serial PUCT.
  * The batched search still plays the solved games optimally (the minimax oracle).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import pairwise
from types import SimpleNamespace

import pytest

from core import MCTS
from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from tests.fixtures.pass_game import (
    PassGame,
    Scenario,
    consecutive_trap_game,
    consecutive_win_game,
)
from tests.reference.minimax import optimal_values, reachable_states, subtree_size
from tests.test_batched_mcts_api import _all_nodes

BATCH_SIZES = [1, 2, 3, 5, 8]
BUDGETS = [1, 2, 17, 200]
PREFIX_PLIES = [0, 2, 5]  # random moves played before the search; 0 is the start state


def _random_pass_game(seed: int, depth: int = 5) -> PassGame:
    """Return a scripted game tree whose mover at each state is drawn at random.

    Movers repeat and alternate with no pattern, and one-move states force collisions,
    so the consecutive-mover backup is exercised deep in a batched search — the two
    hand-written pass fixtures are only two plies deep.

    Args:
        seed: Seeds the tree shape, the movers and the payoffs.
        depth: Maximum number of plies from the start state to a terminal state.

    Returns:
        The game, with zero-sum payoffs in ``{-1, 0, +1}``.
    """
    rng = random.Random(seed)
    to_play, edges, terminal = {}, {}, {}
    frontier, next_id = [(0, 0)], 1
    while frontier:
        state, ply = frontier.pop()
        if ply == depth or (ply > 1 and rng.random() < 0.2):
            utility = rng.choice([-1.0, 0.0, 1.0])
            terminal[state] = (utility, -utility)
            continue
        to_play[state] = rng.randrange(2)
        edges[state] = []
        for action in range(rng.randint(1, 3)):
            edges[state].append((action, next_id))
            frontier.append((next_id, ply + 1))
            next_id += 1
    return PassGame(Scenario(start=0, to_play=to_play, edges=edges, terminal=terminal))


GAMES = {
    "ttt": TicTacToe,
    "c4": lambda: Connect4(4, 4, 3),
    "pass-win": consecutive_win_game,
    "pass-trap": consecutive_trap_game,
    "pass-random": lambda: _random_pass_game(7),
}


def _zero(game, state):
    """Evaluate like the M0 default: value ``0.0`` and uniform priors."""
    return 0.0, None


def _dyadic(game, state):
    """Evaluate to a value in eighths and random logits, as a pure function of the state.

    Sums of eighths are exact in floating point, so the value equations below hold with
    ``==`` whatever order a batch's leaves are backed up in.
    """
    rng = random.Random(repr(state))
    value = rng.randrange(-8, 9) / 8
    return value, {a: rng.uniform(-2.0, 2.0) for a in game.legal_moves(state)}


EVALUATORS = [_zero, _dyadic]


def _random_position(game, rng: random.Random, plies: int):
    """Return a nonterminal state reached by ``plies`` random legal moves, or as many fit."""
    state = game.initial_state()
    for _ in range(plies):
        after = game.apply(state, rng.choice(list(game.legal_moves(state))))
        if game.is_terminal(after):
            break
        state = after
    return state


def _in_flight_under(node, unit: int = 1) -> int:
    """Count the in-flight leaves under ``node``, checking the virtual loss held for them.

    Every edge must hold exactly ``unit`` of virtual loss per in-flight leaf beneath it:
    one per accepted descent that has not been backed up, none for a rejected descent or
    a finished simulation. A return of ``0`` at the root is therefore the full back-out
    — no virtual loss and no in-flight mark anywhere in the tree.

    Args:
        node: The subtree to check; ``None`` (an unmaterialized child) counts zero.
        unit: The engine's ``virtual_loss`` magnitude.

    Returns:
        The number of in-flight leaves at or below ``node``.
    """
    if node is None:
        return 0
    held = [_in_flight_under(child, unit) for child in node.children]
    assert node.vloss == [unit * count for count in held]
    assert not (node.in_flight and node.is_expanded)  # only an unevaluated leaf is in flight
    return int(node.in_flight) + sum(held)


class _Probe:
    """A batched evaluator that checks the tree at the only moment a batch is observable.

    Inside each ``evaluate_batch`` call the batch is collected and nothing is filled, so
    this asserts the in-flight contract there: the leaves are distinct nodes, none has
    been evaluated before, and the virtual loss in the tree is exactly theirs.

    Args:
        game: The game to search.
        evaluate: The single-state evaluator whose results are returned.
        batch_size: The ``batch_size`` every run on :attr:`mcts` will use.
        **mcts_kwargs: Forwarded to the :class:`MCTS` constructor.

    Attributes:
        mcts: The engine under test, with this probe as its only evaluator.
        calls: The states handed over, one list per call.
        collided: The ``collided`` flag of every descent, in order.
        rejected: The path length of every rejected descent — the number of edges whose
            virtual loss had to be rolled back.
    """

    def __init__(self, game, evaluate=_zero, *, batch_size: int, **mcts_kwargs):
        self.evaluate = evaluate
        self.batch_size = batch_size
        self.mcts = MCTS(game, evaluate_batch=self, **mcts_kwargs)
        self.calls: list[list] = []
        self.collided: list[bool] = []
        self.rejected: list[int] = []
        self._evaluated: set = set()
        descend = self.mcts._descend

        def spy(**kwargs):
            result = descend(**kwargs)
            self.collided.append(result[2])
            if result[2]:
                self.rejected.append(len(result[0]))
            return result

        self.mcts._descend = spy

    def __call__(self, game, states):
        m = self.mcts
        self.calls.append(list(states))
        leaves = [node for node in _all_nodes(m.root) if node.in_flight]
        pending = [leaf for leaf in leaves if not leaf.is_terminal]
        # Every state handed over belongs to its own in-flight node (the tree walk yields
        # each node once), though two of those nodes may hold equal states.
        assert Counter(leaf.state for leaf in pending) == Counter(states)
        assert not any(leaf.is_expanded for leaf in pending)
        assert self._evaluated.isdisjoint(pending)  # no leaf is evaluated twice
        self._evaluated.update(pending)
        assert 0 < len(leaves) <= self.batch_size
        assert _in_flight_under(m.root, m.virtual_loss) == len(leaves)
        return [self.evaluate(game, state) for state in states]

    def run(self, num_simulations: int) -> dict:
        """Run a search and return the pre-run snapshot :func:`_assert_visits_conserved` takes.

        Args:
            num_simulations: The budget to spend at this probe's batch size.

        Returns:
            ``{node: edge visit counts}`` for every node that was expanded before the run.
        """
        before = {node: list(node.N) for node in _all_nodes(self.mcts.root) if node.is_expanded}
        self.mcts.run(num_simulations, batch_size=self.batch_size)
        return before


def _searches(game_name: str, batch_size: int):
    """Drive the seeded sweep over random positions, yielding after every run.

    Each engine searches a fresh root, searches it again once expanded, then reuses the
    subtree under a random root action and searches that — a root that may be expanded,
    never materialized, or terminal.

    Args:
        game_name: A key of ``GAMES``.
        batch_size: The batch size of every run.

    Yields:
        ``(probe, before, budget, done)``: the probe owning the engine, the pre-run
        snapshot, the budget just spent, and ``completed_simulations`` before the run.
    """
    game = GAMES[game_name]()
    for evaluate in EVALUATORS:
        for plies in PREFIX_PLIES:
            for budget in BUDGETS:
                rng = random.Random(f"{game_name}:{plies}:{budget}")
                probe = _Probe(game, evaluate, batch_size=batch_size)
                m = probe.mcts
                m.set_root(_random_position(game, rng, plies))
                for reuse in (False, False, True):
                    if reuse:
                        m.advance(rng.choice(m.root.actions))
                    done = m.completed_simulations
                    yield probe, probe.run(budget), budget, done


sweep = pytest.mark.parametrize(
    ("game_name", "batch_size"), [(g, k) for g in GAMES for k in BATCH_SIZES]
)


# --- visit accounting: conditional conservation against a pre-run snapshot -------------


def _assert_visits_conserved(root, before: dict, budget: int) -> Counter:
    """Check one run's visit accounting node by node, as deltas from ``before``.

    A node's *visits* are the simulations of this run that reached it: the whole budget
    at the root, the gain on its parent edge everywhere else. What they add to the
    node's own edges depends on the state the run found it in:

      * terminal — every visit ends there; the node never has an edge.
      * reused (expanded before the run) — every visit traversed one of its edges, so
        its edge total gains exactly its visits.
      * fresh (unexpanded, or not yet materialized) — its first visit expanded it and
        traversed no edge, so its edge total is its visits minus one.

    Below the root the same offset holds in absolute terms, since a node's parent edge
    carries every visit it ever had: an expanded child's edge total is its parent-edge
    ``N`` minus one, and a child is materialized exactly when that ``N`` is positive.

    Args:
        root: The root after the run.
        before: ``{node: edge visit counts}`` for the nodes expanded before the run.
        budget: The simulations the run completed.

    Returns:
        How many nodes of each state were checked, so a caller can assert coverage.
    """
    states = Counter()
    stack = [(root, budget)]
    while stack:
        node, visits = stack.pop()
        old = before.get(node)
        if node.is_terminal:
            states["terminal"] += 1
            assert not node.is_expanded and node.actions == [] and node.N == []
            continue
        if old is not None:
            states["reused"] += 1
            assert sum(node.N) - sum(old) == visits
        else:
            states["fresh"] += 1
            assert node.is_expanded and sum(node.N) == visits - 1
        for i, child in enumerate(node.children):
            assert (child is None) == (node.N[i] == 0)
            if child is None:
                continue
            if not child.is_terminal:
                assert child.is_expanded and sum(child.N) == node.N[i] - 1
            stack.append((child, node.N[i] - (old[i] if old else 0)))
    return states


@sweep
def test_visits_are_conserved_per_node_state_against_the_completed_counter(game_name, batch_size):
    states, roots = Counter(), Counter()
    for probe, before, budget, done in _searches(game_name, batch_size):
        m = probe.mcts
        root = m.root
        assert m.completed_simulations - done == budget
        # The root, spelled out: its visits are the budget itself.
        if root.is_terminal:
            roots["terminal"] += 1
            assert root.N == []
        elif root in before:
            roots["reused"] += 1
            assert sum(root.N) == sum(before[root]) + budget
        else:
            roots["fresh"] += 1
            assert sum(root.N) == budget - 1
        states += _assert_visits_conserved(root, before, budget)
    # Every node state was really exercised, at the root and below it.
    assert roots["fresh"] and roots["reused"]
    assert states["fresh"] and states["reused"] and states["terminal"]


def test_the_sweep_reuses_fresh_expanded_and_terminal_roots():
    # The root after advance() is whatever the random action led to; over the sweep that
    # is every state a reused root can be in, so none of the equations above is vacuous.
    roots = Counter()
    for game_name in GAMES:
        for probe, before, _budget, _done in _searches(game_name, 3):
            root = probe.mcts.root
            roots["terminal" if root.is_terminal else "reused" if root in before else "fresh"] += 1
    assert roots["terminal"] and roots["reused"] and roots["fresh"]


# --- full back-out, and exact virtual loss while a batch is in flight ------------------


@sweep
def test_virtual_loss_is_fully_backed_out_after_every_run(game_name, batch_size):
    rolled_back = 0
    for probe, _before, _budget, _done in _searches(game_name, batch_size):
        assert _in_flight_under(probe.mcts.root) == 0
        rolled_back += sum(probe.rejected)
    if batch_size > 1:
        assert rolled_back  # rejected descents that held virtual loss were among them


@pytest.mark.parametrize("batch_size", [2, 5])
def test_back_out_is_exact_at_a_virtual_loss_magnitude_other_than_one(batch_size):
    game = Connect4(4, 4, 3)
    probe = _Probe(game, _dyadic, batch_size=batch_size, virtual_loss=3)

    probe.mcts.set_root(game.initial_state())
    before = probe.run(200)  # the probe checks 3 units per in-flight leaf mid-batch

    assert any(len(call) > 1 for call in probe.calls)
    assert _in_flight_under(probe.mcts.root, unit=3) == 0
    _assert_visits_conserved(probe.mcts.root, before, 200)


def test_an_all_terminal_batch_backs_out_and_keeps_counting_the_parent_edge():
    # Three forced-win moves from the root: once the root is expanded every leaf is
    # terminal, so no batch reaches the evaluator again and each visit lands on an edge.
    game = PassGame(
        Scenario(
            start=0,
            to_play={0: 0},
            edges={0: [(0, 1), (1, 2), (2, 3)]},
            terminal={1: (1.0, -1.0), 2: (1.0, -1.0), 3: (1.0, -1.0)},
        )
    )
    probe = _Probe(game, batch_size=3)
    m = probe.mcts

    m.set_root(0)
    before = probe.run(10)

    assert probe.calls == [[0]]
    assert sum(m.root.N) == 9 and m.root.W == [float(n) for n in m.root.N]
    assert all(child.is_terminal and child.N == [] for child in m.root.children)
    assert _in_flight_under(m.root) == 0
    assert _assert_visits_conserved(m.root, before, 10) == {"fresh": 1, "terminal": 3}


# --- divergence and collisions: the pinned contract, not a maximal-batch claim ---------


@sweep
def test_a_rejected_descent_is_never_a_simulation_and_never_retried(game_name, batch_size):
    collisions = 0
    for probe, _before, _budget, _done in _searches(game_name, batch_size):
        m = probe.mcts
        # Accepted descents are exactly the simulations, over the engine's lifetime.
        assert probe.collided.count(False) == m.completed_simulations
        # A collision closes its batch and the next batch's first descent cannot collide.
        assert not probe.collided[0]
        assert not any(a and b for a, b in pairwise(probe.collided))
        assert sum(len(call) for call in probe.calls) <= m.completed_simulations
        assert all(0 < len(call) <= batch_size for call in probe.calls)
        collisions += sum(probe.collided)
    assert bool(collisions) == (batch_size > 1)  # and the serial search never collides


def _star(width: int) -> PassGame:
    """Return a game whose root has ``width`` moves, each to its own one-move state."""
    children = range(1, width + 1)
    return PassGame(
        Scenario(
            start=0,
            to_play={0: 0, **{c: c % 2 for c in children}},
            edges={0: [(c - 1, c) for c in children], **{c: [(0, c + width)] for c in children}},
            terminal={c + width: (0.0, 0.0) for c in children},
        )
    )


@pytest.mark.parametrize("width", [2, 3, 5, 8])
def test_virtual_loss_spreads_one_batch_over_every_equal_prior_leaf(width):
    game = _star(width)
    probe = _Probe(game, batch_size=width)
    m = probe.mcts
    m.set_root(0)
    probe.run(1)  # expand the root: `width` unexpanded children of equal prior

    probe.run(width)

    # One collection of `width` descents took all `width` leaves, none twice.
    assert probe.calls == [[0], list(range(1, width + 1))]
    assert m.root.N == [1] * width
    assert not any(probe.collided)


@pytest.mark.parametrize("width", [2, 5])
def test_a_batch_wider_than_the_tree_closes_short_at_the_first_collision(width):
    game = _star(width)
    probe = _Probe(game, batch_size=width + 3)
    m = probe.mcts
    m.set_root(0)
    probe.run(1)

    probe.run(width + 3)

    # The descent after the last free leaf re-enters one in flight: the batch closes at
    # `width` and the rest of the budget is spent by later batches.
    assert probe.calls[1] == list(range(1, width + 1))
    assert probe.collided[: width + 2] == [False] * (width + 1) + [True]
    assert probe.rejected[0] == 1  # one edge deep: the root edge it re-entered
    assert sum(m.root.N) == width + 3
    assert _in_flight_under(m.root) == 0


def test_a_collision_deep_in_a_forced_line_rolls_back_its_whole_path():
    # 0 (P0) --0--> 1 (P0) --0--> 2 (P0) --0--> 3 (P0) --0--> 9  (+1, -1)
    #        \-1--> 4 (P1) --0--> 5 (P0) --0--> 7  (0, 0)
    #                      \-1--> 6 (P0) --0--> 8  (0, 0)
    game = PassGame(
        Scenario(
            start=0,
            to_play={0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0},
            edges={
                0: [(0, 1), (1, 4)],
                1: [(0, 2)],
                2: [(0, 3)],
                3: [(0, 9)],
                4: [(0, 5), (1, 6)],
                5: [(0, 7)],
                6: [(0, 8)],
            },
            terminal={7: (0.0, 0.0), 8: (0.0, 0.0), 9: (1.0, -1.0)},
        )
    )
    # Every state looks won for its mover: the forced line is good for the root mover,
    # the side branch (the opponent's move at state 4) bad.
    held = {}  # state under evaluation -> virtual loss on the root and on state 1

    def evaluate(game, state):
        root = probe.mcts.root
        line = root.children[0] if root.is_expanded else None
        held[state] = (list(root.vloss), list(line.vloss) if line else None)
        return 1.0, None

    probe = _Probe(game, evaluate, batch_size=2)
    m = probe.mcts
    m.set_root(0)

    before = probe.run(5)

    # Batch 2 diverges onto both root moves. In batch 3 the first descent defers state 2
    # behind the forced move at state 1; virtual loss cannot outweigh that line's value,
    # so the second descent follows it into state 2 again and collides two edges deep.
    # The batch closes at one leaf, not two, though neither move at state 4 has ever
    # been tried: a batch need not fill while unevaluated leaves remain.
    assert probe.calls[:3] == [[0], [1, 4], [2]]
    assert probe.collided == [False, True, False, False, False, True, False]
    assert probe.rejected == [0, 2]
    # While state 2 was evaluated, both edges of the line held the accepted descent's
    # virtual loss only: the rejected descent's was rolled back from each.
    assert held[2] == ([1, 0], [1])
    # Seven descents, two rejected: exactly the five simulations asked for were spent.
    assert m.completed_simulations == 5
    assert _in_flight_under(m.root) == 0
    _assert_visits_conserved(m.root, before, 5)


# --- perspective: player-aware backup on every edge, under any batch order -------------


def _assert_backed_up_values_are_player_aware(game, root, evaluate) -> None:
    """Check every edge's ``W`` against what was backed up through it (invariant 2).

    A terminal child backs up ``terminal_utility`` for the edge's mover on every visit.
    An expanded child backs up its own evaluation once, then everything its edges
    accumulated — both in the child mover's perspective, so the parent edge takes the
    sum as is when the same player moves at both nodes and negated otherwise.

    Args:
        game: The game searched.
        root: The root of a finished search.
        evaluate: The evaluator the search used; its values must sum exactly.
    """
    for node in _all_nodes(root):
        for i, child in enumerate(node.children):
            if child is None:
                continue
            if child.is_terminal:
                expected = node.N[i] * game.terminal_utility(child.state, node.to_play)
            else:
                below = evaluate(game, child.state)[0] + sum(child.W)
                expected = below if child.to_play == node.to_play else -below
            assert node.W[i] == expected
            assert node.Q[i] == node.W[i] / node.N[i]


@sweep
def test_backup_is_player_aware_on_every_edge(game_name, batch_size):
    same_mover = 0
    for probe, _before, _budget, _done in _searches(game_name, batch_size):
        m = probe.mcts
        _assert_backed_up_values_are_player_aware(m.game, m.root, probe.evaluate)
        same_mover += any(
            child is not None and child.to_play == node.to_play
            for node in _all_nodes(m.root)
            for child in node.children
        )
    if game_name.startswith("pass"):
        assert same_mover  # consecutive-mover edges were among those checked


# --- batch_size = 1 is the serial search -----------------------------------------------


def _serial_node(game, state):
    """Return an unexpanded node of the serial reference search."""
    terminal = game.is_terminal(state)
    to_play = None if terminal else game.current_player(state)
    return SimpleNamespace(
        state=state, to_play=to_play, actions=[], P=[], N=[], W=[], Q=[], children=[]
    )


def _serial_search(game, root, num_simulations: int, evaluate) -> None:
    """Run a textbook serial PUCT search in place — the oracle for ``batch_size = 1``.

    Written independently of ``core.mcts`` from design doc §6.2: D11 selection with
    ``Q = 0`` for unvisited edges and ties to the lowest index, sparse expansion with a
    softmax over the legal logits, player-aware backup. It has no virtual loss, no
    in-flight marker and no batch — one simulation at a time, start to finish.

    Args:
        game: The game to search.
        root: A node from :func:`_serial_node`; searched further if already expanded.
        num_simulations: Simulations to run.
        evaluate: ``(game, state) -> (value, logits | None)`` for non-terminal leaves.
    """
    for _ in range(num_simulations):
        node, path = root, []
        while node.actions:
            total = sum(node.N)
            c = 1.25 + math.log((total + 19_652.0 + 1.0) / 19_652.0)
            scores = [
                node.Q[k] + c * node.P[k] * math.sqrt(total) / (1.0 + node.N[k])
                for k in range(len(node.actions))
            ]
            i = scores.index(max(scores))
            path.append((node, i))
            if node.children[i] is None:
                node.children[i] = _serial_node(game, game.apply(node.state, node.actions[i]))
            node = node.children[i]
        if node.to_play is None:  # terminal: valued for the player who just moved
            if not path:
                continue
            value_player = path[-1][0].to_play
            value = game.terminal_utility(node.state, value_player)
        else:
            value, logits = evaluate(game, node.state)
            value_player = node.to_play
            node.actions = list(game.legal_moves(node.state))
            n = len(node.actions)
            if logits is None:
                node.P = [1.0 / n] * n
            else:
                top = max(logits[a] for a in node.actions)
                exps = [math.exp(logits[a] - top) for a in node.actions]
                node.P = [e / sum(exps) for e in exps]
            node.N, node.W, node.Q, node.children = [0] * n, [0.0] * n, [0.0] * n, [None] * n
        for parent, i in path:
            parent.N[i] += 1
            parent.W[i] += value if parent.to_play == value_player else -value
            parent.Q[i] = parent.W[i] / parent.N[i]


def _stats(root) -> list:
    """Return every node's ``{N, W, Q, P}`` in preorder, floats rendered bit for bit."""
    return [
        (
            node.state,
            node.actions,
            node.N,
            *([x.hex() for x in xs] for xs in (node.W, node.Q, node.P)),
        )
        for node in _all_nodes(root)
    ]


@pytest.mark.parametrize("path", ["default", "batch-size-1", "batched-evaluator"])
@pytest.mark.parametrize("game_name", sorted(GAMES))
def test_batch_size_one_is_bit_identical_to_an_independent_serial_search(game_name, path):
    game = GAMES[game_name]()
    run_kwargs = {} if path == "default" else {"batch_size": 1}
    for evaluate in EVALUATORS:
        for plies in PREFIX_PLIES:
            state = _random_position(game, random.Random(f"{game_name}:{plies}"), plies)
            if path == "batched-evaluator":
                m = MCTS(
                    game, evaluate_batch=lambda g, states, f=evaluate: [f(g, s) for s in states]
                )
            else:  # the M0 default (no evaluator at all) stands in for the zero evaluator
                m = MCTS(game, evaluate=None if evaluate is _zero else evaluate)
            reference = _serial_node(game, state)

            m.run(200, root_state=state, **run_kwargs)
            _serial_search(game, reference, 200, evaluate)
            assert _stats(m.root) == _stats(reference)

            # Most visits, ties to the lowest action id — and on across subtree reuse.
            best = min(
                range(len(reference.N)), key=lambda k: (-reference.N[k], reference.actions[k])
            )
            assert m.best_action() == reference.actions[best]
            m.advance(m.best_action())
            reference = reference.children[best]
            m.run(100, **run_kwargs)
            _serial_search(game, reference, 100, evaluate)
            assert _stats(m.root) == _stats(reference)


# --- the MCTS-vs-minimax oracle, re-run through batched selection ----------------------

SOLVED = {
    "ttt": (TicTacToe, 100),
    "connect3": (lambda: Connect4(3, 3, 3), 30),
}


def _assert_plays_optimally_on_endgames(game, plies: int, sims_for, batch_size: int) -> int:
    """Assert batched MCTS keeps the mover's optimal value at every late position.

    Args:
        game: A solved game whose state is ``(board, to_play)`` with ``-1`` for empty.
        plies: Check every reachable nonterminal position with at most this many empty cells.
        sims_for: Maps a position's subtree size to its simulation budget.
        batch_size: The batch size of every search.

    Returns:
        The number of positions checked.
    """
    values: dict = {}
    sizes: dict = {}
    tested = 0
    for state in reachable_states(game):
        if game.is_terminal(state) or state[0].count(-1) > plies:
            continue
        m = MCTS(game)
        m.run(sims_for(subtree_size(game, state, sizes)), root_state=state, batch_size=batch_size)
        mover = game.current_player(state)
        target = optimal_values(game, state, values)[mover]
        achieved = optimal_values(game, game.apply(state, m.best_action()), values)[mover]
        assert achieved >= target - 1e-9, (
            f"batch_size={batch_size} blundered on {state}: chose {m.best_action()} "
            f"(value {achieved}) < optimal {target}"
        )
        tested += 1
    return tested


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("solved", sorted(SOLVED))
def test_batched_mcts_plays_optimally_on_solved_endgames(solved, batch_size):
    make_game, at_least = SOLVED[solved]
    tested = _assert_plays_optimally_on_endgames(
        make_game(), 3, lambda size: min(2000, max(240, 30 * size)), batch_size
    )
    assert tested > at_least


@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [1, 4])
def test_batched_mcts_plays_optimally_on_ttt_deep(batch_size):
    _assert_plays_optimally_on_endgames(
        TicTacToe(), 5, lambda size: min(3000, max(300, 16 * size)), batch_size
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    ("moves", "sims", "best"),
    [([0, 1, 0, 1, 0, 1], 1500, 0), ([0, 1, 2, 1, 0, 1], 2000, 1)],
    ids=["takes-the-win", "blocks-the-loss"],
)
def test_batched_mcts_finds_the_connect4_tactic(moves, sims, best, batch_size):
    game = Connect4()
    m = MCTS(game)
    m.run(sims, root_state=game.from_moves(moves), batch_size=batch_size)
    assert m.best_action() == best
