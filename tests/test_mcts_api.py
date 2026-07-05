"""MCTS public-API contract edge cases (design doc §6.2, §12 M0).

Covers the argument-validation and tie-break guarantees documented on ``MCTS.advance``
and ``MCTS.best_action`` — the paths the invariant/search suites don't exercise directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core import MCTS
from games.tictactoe import TicTacToe


def test_advance_rejects_illegal_action_on_unexpanded_root():
    # A fresh (never-searched) root has no edge list, so advance must validate legality
    # itself instead of handing the action to apply() and building a malformed state
    # (TTT's apply would grow the board from 9 to 18 cells on action -1).
    game = TicTacToe()
    m = MCTS(game)
    m.set_root(game.initial_state())
    with pytest.raises(ValueError):
        m.advance(-1)


def test_advance_rejects_illegal_action_on_expanded_root():
    game = TicTacToe()
    m = MCTS(game)
    m.run(50, root_state=game.initial_state())  # expands the root
    with pytest.raises(ValueError):
        m.advance(-1)


def test_advance_accepts_legal_action_on_unexpanded_root():
    game = TicTacToe()
    m = MCTS(game)
    m.set_root(game.initial_state())
    m.advance(4)  # center is legal from the initial position
    assert m.root is not None
    assert m.root.state == game.apply(game.initial_state(), 4)


def test_best_action_breaks_ties_by_lowest_action_id():
    # Equal visit counts with actions in non-ascending id order: the documented contract
    # is lowest action id wins, independent of adapter/index order. best_action reads only
    # node.actions and node.N, so a duck-typed node exercises the tie-break in isolation.
    m = MCTS(TicTacToe())
    node = SimpleNamespace(actions=[9, 2], N=[5, 5])
    assert m.best_action(node) == 2
