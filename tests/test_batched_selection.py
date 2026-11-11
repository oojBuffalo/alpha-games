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

import random
from collections import Counter

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
