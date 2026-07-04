"""Tic-Tac-Toe: a small, fully-solvable strictly-alternating reference game (M0).

State is ``(board, to_play)`` with ``board`` a length-9 tuple over ``{EMPTY, 0, 1}``
(row-major, cells 0..8) and ``to_play`` in ``{0, 1}``. Immutable and hashable so the
reference solver can memoize on it.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.game import Action, PlayerId, State, ValueTargetSpec
from core.game import Game as _Game

EMPTY = -1

# Row-major indices of the eight winning lines.
_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # columns
    (0, 4, 8),
    (2, 4, 6),  # diagonals
)

TTTState = tuple  # (tuple[int, ...], int)


class TicTacToe(_Game):
    """Standard 3x3 Tic-Tac-Toe. Perfect play is a draw."""

    @property
    def num_players(self) -> int:
        return 2

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def is_perfect_information(self) -> bool:
        return True

    @property
    def symmetry_group(self) -> Sequence:
        return ()

    @property
    def value_targets(self) -> ValueTargetSpec:
        return ValueTargetSpec(primary_name="z")

    def initial_state(self) -> State:
        return ((EMPTY,) * 9, 0)

    def current_player(self, state: State) -> PlayerId:
        return state[1]

    def legal_moves(self, state: State) -> Sequence[Action]:
        board = state[0]
        return [i for i in range(9) if board[i] == EMPTY]

    def apply(self, state: State, action: Action) -> State:
        board, to_play = state
        if board[action] != EMPTY:
            raise ValueError(f"cell {action} is occupied")
        new_board = board[:action] + (to_play,) + board[action + 1 :]
        return (new_board, 1 - to_play)

    def is_terminal(self, state: State) -> bool:
        board = state[0]
        return self._winner(board) is not None or EMPTY not in board

    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        winner = self._winner(state[0])
        if winner is None:
            return 0.0  # draw (or full board with no line)
        return 1.0 if winner == player_id else -1.0

    @staticmethod
    def _winner(board: tuple[int, ...]) -> PlayerId | None:
        for a, b, c in _LINES:
            v = board[a]
            if v != EMPTY and v == board[b] == board[c]:
                return v
        return None

    # --- test convenience ------------------------------------------------------

    def from_grid(self, rows: Sequence[str], to_play: PlayerId) -> State:
        """Build a state from a 3-row ASCII grid (``X`` -> 0, ``O`` -> 1, ``.`` -> empty).

        Args:
            rows: Three strings of three chars each.
            to_play: The player to move in the constructed position.

        Returns:
            The corresponding ``(board, to_play)`` state.
        """
        mapping = {"X": 0, "O": 1, ".": EMPTY}
        cells = [mapping[ch] for row in rows for ch in row]
        if len(cells) != 9:
            raise ValueError("grid must be 3x3")
        return (tuple(cells), to_play)
