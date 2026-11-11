"""Batched parallel selection: API-level smoke tests (design doc §6.2, §12 M5).

The serial engine is pinned first: a seeded golden of every node's ``{N, W, Q}``,
captured from the engine *before* batching touched it, so "``batch_size = 1`` reproduces
the serial engine bit for bit" is checked against that engine and not against itself.
"""

from __future__ import annotations

import hashlib
import random

import pytest

from core import MCTS
from games.connect4 import Connect4
from games.tictactoe import TicTacToe

EPS = 0.25  # D7
ALPHA_NUMERATOR = 10.8  # D7


def _all_nodes(root):
    """Return every materialized node under ``root`` in deterministic preorder."""
    nodes, stack = [], [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        nodes.append(node)
        stack.extend(reversed(node.children))
    return nodes


def _digest(root) -> str:
    """Return a hash of every node's exact ``{N, W, Q}`` (floats compared bit for bit)."""
    rows = [
        (list(node.N), [w.hex() for w in node.W], [q.hex() for q in node.Q])
        for node in _all_nodes(root)
    ]
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def _seeded_evaluator(seed: int):
    """Return a deterministic evaluator: value and logits are a pure function of the state."""

    def evaluate(game, state):
        rng = random.Random(f"{seed}:{state!r}")
        value = rng.uniform(-1.0, 1.0)
        return value, {a: rng.uniform(-2.0, 2.0) for a in game.legal_moves(state)}

    return evaluate


def _seeded_search(game, seed: int | None, **run_kwargs) -> tuple[str, int, str]:
    """Search, reuse the chosen subtree, search again; return what the golden pins.

    Args:
        game: The adapter to search from its initial state.
        seed: Seeds the evaluator and the D7 noise stream; ``None`` runs the M0 default
            (no evaluator, no noise).
        **run_kwargs: Forwarded to both :meth:`MCTS.run` calls.

    Returns:
        ``(first-search digest, best action, post-advance digest)``.
    """
    if seed is None:
        m = MCTS(game)
    else:
        noise = (EPS, ALPHA_NUMERATOR, random.Random(seed))
        m = MCTS(game, evaluate=_seeded_evaluator(seed), root_noise=noise)
    first = _digest(m.run(300, root_state=game.initial_state(), **run_kwargs))
    action = m.best_action()
    m.advance(action)
    return first, action, _digest(m.run(150, **run_kwargs))


# Captured from the serial engine as it stood before batched selection was added.
SERIAL_GOLDEN = {
    "ttt-m0": (
        "e2b444ea265559ac670e00bc57738ccd8bbda1a0d0c351e0536299795b9570d9",
        0,
        "e6912681331065cee7f3420c6dcc67e81ae204d4f5e624fe2dacd054bff9bac5",
    ),
    "ttt-seeded": (
        "1c2a3f211913aa05256d78ad6eab16611ef4b4afdfde63d9e3f05a496b83b1ad",
        2,
        "f1eef20b33ab4bee7c7ec9f0faa4542b6a74b2d3cb784ee8ec8c23b78e8b679e",
    ),
    "c4-m0": (
        "aa08986251a0120ee017e478dd8dd599f9ef1da543bfb80e9e4279abc758a0e0",
        0,
        "05722434f77fd65d6ab1062d9f4a601fd3c2f8c06aa37ee91d7130dfc3ec7be6",
    ),
    "c4-seeded": (
        "f9750cbfe351749f8c3f05b25f51c5711540a1abfc78ce50494b18e9d0a032af",
        4,
        "a0e377bafd2fe94257ef73edbca08e68787bf3b88dcdfb76cc91deda1d9dc7d3",
    ),
}

GOLDEN_CASES = {
    "ttt-m0": (TicTacToe, None),
    "ttt-seeded": (TicTacToe, 20261110),
    "c4-m0": (Connect4, None),
    "c4-seeded": (Connect4, 20261110),
}


@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_serial_search_matches_the_pre_batching_golden(case):
    make_game, seed = GOLDEN_CASES[case]
    assert _seeded_search(make_game(), seed) == SERIAL_GOLDEN[case]


# --- bookkeeping primitives: completed-simulations counter + in-flight marker ----------


@pytest.mark.parametrize("make_game", [TicTacToe, Connect4])
def test_run_counts_completed_simulations_and_leaves_nothing_in_flight(make_game):
    game = make_game()
    m = MCTS(game)
    assert m.completed_simulations == 0

    root = m.run(120, root_state=game.initial_state())
    assert m.completed_simulations == 120
    assert not any(node.in_flight for node in _all_nodes(root))

    # The counter is engine-lifetime: subtree reuse and incremental runs only add to it.
    m.advance(m.best_action())
    m.run(30)
    assert m.completed_simulations == 150
    assert not any(node.in_flight for node in _all_nodes(m.root))


# --- expansion split: defer (reserve the leaf) then fill (install the evaluation) ------


def test_deferred_leaf_is_a_collision_target_until_it_is_filled():
    game = TicTacToe()
    m = MCTS(game)
    m.set_root(game.initial_state())
    node = m.root

    m._defer_expand(node)
    # Reserved, not expanded: a second descent reaching it must see a collision, never
    # re-expand it or select through edges that have no priors yet.
    assert node.in_flight and not node.is_expanded
    assert node.actions == list(game.legal_moves(node.state))
    assert node.P == []

    assert m._fill_expand(node, 0.25, {4: 3.0}) == 0.25
    n = len(node.actions)
    assert node.is_expanded
    assert abs(sum(node.P) - 1.0) < 1e-12
    assert max(node.P) == node.P[node.actions.index(4)]  # the one raised logit
    assert node.N == [0] * n and node.W == [0.0] * n and node.Q == [0.0] * n
    assert node.vloss == [0] * n and node.children == [None] * n


def test_descent_reports_a_collision_at_a_leaf_that_is_already_in_flight():
    game = TicTacToe()
    m = MCTS(game)
    m.set_root(game.initial_state())

    path, leaf, collided = m._descend(apply_vloss=True)
    assert (path, leaf, collided) == ([], m.root, False)
    assert leaf.in_flight and not leaf.is_expanded  # deferred: no evaluation yet

    assert m._descend(apply_vloss=True) == ([], m.root, True)
