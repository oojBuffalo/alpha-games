"""Connect Four rules + solved-position value tests (design doc §12 M0).

Win-detection and legality use the standard board; solver-grounded value assertions use
small variants (solving the standard board is infeasible and is not the point here).
"""

from __future__ import annotations

import pytest

from games.connect4 import Connect4
from tests.reference.minimax import optimal_actions, optimal_values


def test_vertical_win_detection():
    g = Connect4()
    s = g.from_moves([0, 1, 0, 1, 0, 1, 0])  # player 0 stacks col 0 four high
    assert g.is_terminal(s)
    assert g.terminal_utility(s, 0) == 1.0
    assert g.terminal_utility(s, 1) == -1.0


def test_horizontal_win_detection():
    g = Connect4()
    s = g.from_moves([0, 0, 1, 1, 2, 2, 3])  # player 0 fills the bottom row across cols 0..3
    assert g.is_terminal(s)
    assert g.terminal_utility(s, 0) == 1.0


def test_diagonal_down_right_win_detection():
    g = Connect4(4, 4, 3)
    s = g.from_grid(["X...", ".X..", "..X.", "...."], to_play=1)
    assert g.is_terminal(s)
    assert g.terminal_utility(s, 0) == 1.0


def test_diagonal_down_left_win_detection():
    g = Connect4(4, 4, 3)
    s = g.from_grid(["..X.", ".X..", "X...", "...."], to_play=1)
    assert g.is_terminal(s)
    assert g.terminal_utility(s, 0) == 1.0


def test_full_column_is_not_legal():
    g = Connect4(3, 3, 3)
    s = g.from_moves([0, 0, 0])  # col 0 is now full
    assert 0 not in g.legal_moves(s)
    assert set(g.legal_moves(s)) == {1, 2}


def test_full_board_without_a_line_is_a_draw():
    g = Connect4(2, 3, 3)
    s = g.from_grid(["OXO", "XOX"], to_play=0)
    assert g.is_terminal(s)
    assert g.terminal_utility(s, 0) == 0.0
    assert g.terminal_utility(s, 1) == 0.0


def test_small_board_immediate_win_value():
    g = Connect4(3, 3, 3)
    s = g.from_moves([0, 1, 0, 1])  # player 0 has two stacked in col 0 and is to move
    assert optimal_values(g, s)[0] == 1.0  # completing col 0 wins
    assert 0 in optimal_actions(g, s)


@pytest.mark.parametrize(
    "rows, cols, connect",
    [
        (3, 3, 0),  # non-positive connect: first stone would be a vacuous "win"
        (0, 3, 3),  # non-positive rows
        (3, 0, 3),  # non-positive cols
        (-1, 3, 3),  # negative rows
        (0, 0, 0),  # fully degenerate: empty board reported as terminal
    ],
)
def test_rejects_non_positive_dimensions(rows, cols, connect):
    with pytest.raises(ValueError):
        Connect4(rows, cols, connect)


def test_rejects_connect_exceeding_board():
    # Guards the pre-existing check against being shadowed by the positivity checks.
    with pytest.raises(ValueError):
        Connect4(3, 3, 4)
