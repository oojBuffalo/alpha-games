"""Tic-Tac-Toe rules + solved-position value tests (design doc §12 M0)."""

from __future__ import annotations

from games.tictactoe import TicTacToe
from tests.reference.minimax import optimal_actions, optimal_values


def test_initial_position_is_a_draw():
    g = TicTacToe()
    assert optimal_values(g, g.initial_state()) == (0.0, 0.0)


def test_win_detection_and_utility_row():
    g = TicTacToe()
    s = g.from_grid(["XXX", "OO.", "..."], to_play=1)
    assert g.is_terminal(s)
    assert g.terminal_utility(s, 0) == 1.0
    assert g.terminal_utility(s, 1) == -1.0


def test_win_detection_diagonal():
    g = TicTacToe()
    s = g.from_grid(["X..", "OX.", "OOX"], to_play=1)
    assert g.is_terminal(s)
    assert g.terminal_utility(s, 0) == 1.0


def test_full_board_no_line_is_draw():
    g = TicTacToe()
    s = g.from_grid(["XOX", "XOO", "OXX"], to_play=0)
    assert g.is_terminal(s)
    assert g.terminal_utility(s, 0) == 0.0
    assert g.terminal_utility(s, 1) == 0.0


def test_immediate_win_is_the_only_optimal_move():
    g = TicTacToe()
    # X wins at cell 2 (top row); if X plays anything else, O completes the middle row
    # at cell 5 and wins. So taking the win now is the unique optimal move.
    s = g.from_grid(["XX.", "OO.", "..."], to_play=0)
    assert optimal_values(g, s) == (1.0, -1.0)
    assert optimal_actions(g, s) == [2]


def test_must_block_is_the_only_optimal_move():
    g = TicTacToe()
    s = g.from_grid(["X..", "OO.", "..X"], to_play=0)  # O threatens the middle row at cell 5
    assert optimal_actions(g, s) == [5]
