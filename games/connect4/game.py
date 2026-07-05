"""Connect Four: a strictly-alternating reference game, parameterized by board size (M0).

Standard is 7 columns x 6 rows, connect 4. The board size and win length are
constructor parameters so tests can use a small, fully-solvable variant (the same
adapter, a smaller tree) alongside tactical positions on the standard board — and so
M2.5's config-parameterized correctness net has a precedent.

State is ``(board, to_play)`` with ``board`` a length ``rows*cols`` tuple over
``{EMPTY, 0, 1}`` in row-major order (row 0 = top), and ``to_play`` in ``{0, 1}``.
Action ids are column indices ``0..cols-1``.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.game import Action, PlayerId, State, ValueTargetSpec
from core.game import Game as _Game

EMPTY = -1


class Connect4(_Game):
    """Connect Four on a ``rows x cols`` board, ``connect`` in a row to win.

    Args:
        rows: Number of rows (default 6).
        cols: Number of columns (default 7).
        connect: Run length required to win (default 4).
    """

    def __init__(self, rows: int = 6, cols: int = 7, connect: int = 4):
        if rows <= 0 or cols <= 0:
            raise ValueError(f"rows and cols must be positive (got {rows}x{cols})")
        if connect <= 0:
            raise ValueError(f"connect length must be positive (got {connect})")
        if connect > max(rows, cols):
            raise ValueError("connect length exceeds both board dimensions")
        self.rows = rows
        self.cols = cols
        self.connect = connect

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
        return ((EMPTY,) * (self.rows * self.cols), 0)

    def current_player(self, state: State) -> PlayerId:
        return state[1]

    def legal_moves(self, state: State) -> Sequence[Action]:
        board = state[0]
        # A column is playable iff its top cell (row 0) is empty.
        return [c for c in range(self.cols) if board[c] == EMPTY]

    def apply(self, state: State, action: Action) -> State:
        board, to_play = state
        row = self._drop_row(board, action)
        if row is None:
            raise ValueError(f"column {action} is full")
        idx = row * self.cols + action
        new_board = board[:idx] + (to_play,) + board[idx + 1 :]
        return (new_board, 1 - to_play)

    def is_terminal(self, state: State) -> bool:
        board = state[0]
        return self._winner(board) is not None or EMPTY not in board

    def terminal_utility(self, state: State, player_id: PlayerId) -> float:
        winner = self._winner(state[0])
        if winner is None:
            return 0.0
        return 1.0 if winner == player_id else -1.0

    # --- internals -------------------------------------------------------------

    def _drop_row(self, board: tuple[int, ...], col: int) -> int | None:
        """Return the lowest empty row in ``col`` (largest row index), or None if full."""
        for r in range(self.rows - 1, -1, -1):
            if board[r * self.cols + col] == EMPTY:
                return r
        return None

    def _winner(self, board: tuple[int, ...]) -> PlayerId | None:
        rows, cols, k = self.rows, self.cols, self.connect
        for r in range(rows):
            for c in range(cols):
                v = board[r * cols + c]
                if v == EMPTY:
                    continue
                # Check the four forward directions from (r, c): →, ↓, ↘, ↙.
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    er, ec = r + (k - 1) * dr, c + (k - 1) * dc
                    if not (0 <= er < rows and 0 <= ec < cols):
                        continue
                    if all(board[(r + i * dr) * cols + (c + i * dc)] == v for i in range(1, k)):
                        return v
        return None

    # --- test convenience ------------------------------------------------------

    def from_moves(self, columns: Sequence[Action]) -> State:
        """Build a state by playing ``columns`` in order from the initial state.

        Args:
            columns: Column indices to drop into, alternating players from player 0.

        Returns:
            The resulting state (raises if any move is illegal).
        """
        state = self.initial_state()
        for c in columns:
            state = self.apply(state, c)
        return state

    def from_grid(self, rows: Sequence[str], to_play: PlayerId) -> State:
        """Build a state from an ASCII grid (``X`` -> 0, ``O`` -> 1, ``.`` -> empty).

        Row 0 is the top. Useful for exercising win detection directly (the grid need
        not be physically reachable — floating stones are fine for testing ``_winner``).

        Args:
            rows: ``self.rows`` strings of ``self.cols`` chars each.
            to_play: The player to move in the constructed position.

        Returns:
            The corresponding ``(board, to_play)`` state.
        """
        mapping = {"X": 0, "O": 1, ".": EMPTY}
        cells = [mapping[ch] for row in rows for ch in row]
        if len(cells) != self.rows * self.cols:
            raise ValueError(f"grid must be {self.rows}x{self.cols}")
        return (tuple(cells), to_play)
